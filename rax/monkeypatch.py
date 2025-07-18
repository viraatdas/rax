"""Monkeypatch JAX operations for eager mode shape validation."""

import functools
from typing import Any, Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np

from .validator import RAXShapeError


# Store original operations for restoration
_original_ops: Dict[str, Callable] = {}


def validate_shapes_binary(op_name: str, a: Any, b: Any) -> None:
    """Validate shapes for binary operations."""
    a_shape = getattr(a, 'shape', None)
    b_shape = getattr(b, 'shape', None)
    
    if a_shape is None or b_shape is None:
        return  # Can't validate non-array inputs
    
    # Operation-specific validation
    if op_name == 'dot':
        # For dot product: a.shape[-1] must equal b.shape[0]
        if len(a_shape) == 0 or len(b_shape) == 0:
            raise RAXShapeError(
                f"dot requires non-scalar inputs, got shapes {a_shape} and {b_shape}"
            )
        
        if len(b_shape) == 1:
            # Vector dot product
            if a_shape[-1] != b_shape[0]:
                raise RAXShapeError(
                    f"dot shape mismatch: {a_shape} · {b_shape} "
                    f"(last dimension {a_shape[-1]} != {b_shape[0]})"
                )
        else:
            # Matrix multiplication
            if a_shape[-1] != b_shape[-2]:
                raise RAXShapeError(
                    f"dot shape mismatch: {a_shape} · {b_shape} "
                    f"(dimension {a_shape[-1]} != {b_shape[-2]})"
                )
    
    elif op_name == 'matmul':
        # Matrix multiplication rules
        if len(a_shape) == 0 or len(b_shape) == 0:
            raise RAXShapeError(
                f"matmul requires non-scalar inputs, got shapes {a_shape} and {b_shape}"
            )
        
        # Check compatibility
        if a_shape[-1] != b_shape[-2]:
            raise RAXShapeError(
                f"matmul shape mismatch: {a_shape} @ {b_shape} "
                f"(dimension {a_shape[-1]} != {b_shape[-2]})"
            )


def wrap_binary_op(op_name: str, original_op: Callable) -> Callable:
    """Wrap a binary operation with shape validation."""
    @functools.wraps(original_op)
    def wrapped(a, b, **kwargs):
        try:
            validate_shapes_binary(op_name, a, b)
        except RAXShapeError as e:
            raise RAXShapeError(f"[RAX] {e}")
        
        return original_op(a, b, **kwargs)
    
    return wrapped


def validate_reshape(array: Any, new_shape: Tuple[int, ...]) -> None:
    """Validate reshape operation."""
    if not hasattr(array, 'shape'):
        return
    
    old_shape = array.shape
    old_size = np.prod(old_shape)
    
    # Handle -1 in new shape
    if -1 in new_shape:
        known_size = 1
        unknown_idx = None
        for i, dim in enumerate(new_shape):
            if dim == -1:
                if unknown_idx is not None:
                    raise RAXShapeError(
                        "Can only have one -1 dimension in reshape"
                    )
                unknown_idx = i
            else:
                known_size *= dim
        
        if unknown_idx is not None:
            inferred_dim = old_size // known_size
            if known_size * inferred_dim != old_size:
                raise RAXShapeError(
                    f"Cannot reshape array of size {old_size} to {new_shape}"
                )
    else:
        new_size = np.prod(new_shape)
        if old_size != new_size:
            raise RAXShapeError(
                f"Cannot reshape array of shape {old_shape} (size {old_size}) "
                f"to shape {new_shape} (size {new_size})"
            )


def wrap_reshape(original_op: Callable) -> Callable:
    """Wrap reshape with validation."""
    @functools.wraps(original_op)
    def wrapped(array, new_shape, **kwargs):
        try:
            validate_reshape(array, new_shape)
        except RAXShapeError as e:
            raise RAXShapeError(f"[RAX] {e}")
        
        return original_op(array, new_shape, **kwargs)
    
    return wrapped


def validate_einsum(subscripts: str, *operands) -> None:
    """Basic validation for einsum operations."""
    # This is a simplified validation - full einsum validation is complex
    # For now, just check that we have the right number of operands
    
    # Count number of commas to determine expected operands
    if '->' in subscripts:
        input_part = subscripts.split('->')[0]
    else:
        input_part = subscripts
    
    expected_operands = input_part.count(',') + 1
    if len(operands) != expected_operands:
        raise RAXShapeError(
            f"einsum expects {expected_operands} operands for '{subscripts}', "
            f"got {len(operands)}"
        )


def wrap_einsum(original_op: Callable) -> Callable:
    """Wrap einsum with validation."""
    @functools.wraps(original_op)
    def wrapped(subscripts, *operands, **kwargs):
        try:
            validate_einsum(subscripts, *operands)
        except RAXShapeError as e:
            raise RAXShapeError(f"[RAX] {e}")
        
        return original_op(subscripts, *operands, **kwargs)
    
    return wrapped


def apply_monkeypatch(verbose: bool = False) -> None:
    """Apply RAX monkeypatches to JAX operations."""
    global _original_ops
    
    if _original_ops:
        # Already patched
        return
    
    if verbose:
        print("[RAX] Applying operation monkeypatches...")
    
    # Patch binary operations
    binary_ops = [
        ('dot', jnp.dot),
        ('matmul', jnp.matmul),
    ]
    
    for op_name, op in binary_ops:
        if op is not None:
            _original_ops[op_name] = op
            wrapped = wrap_binary_op(op_name, op)
            setattr(jnp, op_name, wrapped)
            if verbose:
                print(f"  - Patched jnp.{op_name}")
    
    # Patch reshape
    if hasattr(jnp, 'reshape'):
        _original_ops['reshape'] = jnp.reshape
        setattr(jnp, 'reshape', wrap_reshape(jnp.reshape))
        if verbose:
            print("  - Patched jnp.reshape")
    
    # Patch einsum
    if hasattr(jnp, 'einsum'):
        _original_ops['einsum'] = jnp.einsum
        setattr(jnp, 'einsum', wrap_einsum(jnp.einsum))
        if verbose:
            print("  - Patched jnp.einsum")
    
    # Also patch the @ operator by patching ndarray.__matmul__
    # This is trickier and may not work in all cases


def remove_monkeypatch(verbose: bool = False) -> None:
    """Remove RAX monkeypatches."""
    global _original_ops
    
    if not _original_ops:
        return
    
    if verbose:
        print("[RAX] Removing operation monkeypatches...")
    
    for op_name, original_op in _original_ops.items():
        setattr(jnp, op_name, original_op)
        if verbose:
            print(f"  - Restored jnp.{op_name}")
    
    _original_ops.clear() 