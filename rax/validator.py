"""Function validation and wrapping for RAX safety guarantees."""

import functools
import inspect
import traceback
from typing import Any, Callable, Optional, get_type_hints
import re

import jax
import jax.numpy as jnp
from beartype import beartype
from beartype.door import is_bearable


class RAXValidationError(Exception):
    """Raised when RAX validation fails."""
    pass


class RAXShapeError(RAXValidationError):
    """Raised when shape validation fails."""
    pass


class RAXMathError(RAXValidationError):
    """Raised when mathematical operation validation fails."""
    pass


def format_shape_error(func_name: str, param_name: str, expected_shape: str, actual_shape: tuple, func_file: str = None, func_line: int = None) -> str:
    """Format a shape error message with clear information."""
    location = ""
    if func_file and func_line:
        location = f"\n  Location: {func_file}:{func_line}"
    
    # Extract dimension info from jaxtyping annotation
    dims_match = re.search(r"'([^']+)'", expected_shape)
    expected_dims = dims_match.group(1) if dims_match else expected_shape
    
    return (
        f"\n[RAX] Shape mismatch in function '{func_name}'{location}\n"
        f"\n  Parameter '{param_name}' has incompatible shape:"
        f"\n    Expected: {expected_dims}"
        f"\n    Actual:   {actual_shape}"
        f"\n\n  This means the input tensor doesn't match the declared type annotation."
    )


def extract_shape_info_from_error(error_msg: str) -> tuple[str, tuple, str]:
    """Extract parameter name, actual shape, and expected shape from error message."""
    # Look for parameter name in different formats
    # The error contains: "Function __main__.linear() parameter x="
    # Let's be more flexible with the regex
    param_match = re.search(r'parameter\s+(\w+)\s*=', error_msg)
    param_name = param_match.group(1) if param_match else "x"
    
    # Look for actual shape in Traced<float32[...]>
    shape_match = re.search(r'Traced<\w+\[([^\]]+)\]>', error_msg)
    if shape_match:
        shape_str = shape_match.group(1)
        # Convert to tuple, handling both numeric and named dimensions
        parts = [x.strip() for x in shape_str.split(',')]
        actual_shape = []
        for part in parts:
            if part.isdigit():
                actual_shape.append(int(part))
            else:
                actual_shape.append(part)
        actual_shape = tuple(actual_shape)
    else:
        actual_shape = "unknown"
    
    # Look for expected shape in jaxtyping format
    # Handle: <class 'jaxtyping.Float32[Array, 'batch 128']'>
    type_match = re.search(r"<class 'jaxtyping\.\w+\[Array, '([^']+)'\]'>", error_msg)
    if type_match:
        expected_shape = type_match.group(1)
    else:
        # Fallback
        expected_shape = "unknown"
    
    return param_name, actual_shape, expected_shape


def has_type_annotations(func: Callable) -> bool:
    """Check if a function has type annotations for all parameters and return type."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    
    # Check if all parameters have type hints
    for param_name, param in sig.parameters.items():
        if param_name not in hints and param.default is inspect.Parameter.empty:
            return False
    
    # Check if return type is annotated
    return 'return' in hints


def validate_jaxpr(func: Callable, *args, **kwargs) -> None:
    """
    Validate a function by tracing it with jax.make_jaxpr.
    
    This catches shape mismatches, invalid operations, etc. at compile time.
    """
    # Get the original function if it's wrapped
    original_func = getattr(func, '_rax_original', func)
    func_name = original_func.__name__
    
    # Get source info
    try:
        source_file = inspect.getsourcefile(original_func)
        source_lines, start_line = inspect.getsourcelines(original_func)
    except:
        source_file = "unknown"
        start_line = 0
    
    try:
        # Use jax.make_jaxpr to trace the function
        jaxpr = jax.make_jaxpr(func)(*args, **kwargs)
        # If we get here, the function traced successfully
    except Exception as e:
        error_msg = str(e)
        
        # Check if this is a beartype validation error
        if "violates type hint" in error_msg and "dimension size" in error_msg:
            # Extract information from error
            param_name, actual_shape, expected_shape = extract_shape_info_from_error(error_msg)
            
            # Format the error nicely
            formatted_error = format_shape_error(
                func_name, param_name, expected_shape, actual_shape,
                source_file, start_line
            )
            raise RAXShapeError(formatted_error) from None
        
        # Broadcasting errors
        if "incompatible shapes for broadcasting" in error_msg:
            shapes_match = re.search(r'broadcasting: (.+?)\.?$', error_msg)
            shapes = shapes_match.group(1) if shapes_match else "unknown shapes"
            raise RAXShapeError(
                f"\n[RAX] Broadcasting error in function '{func_name}'\n"
                f"  Location: {source_file}:{start_line}\n"
                f"\n  Cannot broadcast shapes: {shapes}"
                f"\n\n  Ensure arrays have compatible dimensions for element-wise operations."
            ) from None
        
        # Reshape errors
        if "cannot reshape array" in error_msg:
            # Extract shape info
            from_match = re.search(r'shape \(([^)]+)\) \(size (\d+)\)', error_msg)
            to_match = re.search(r'into shape \(([^)]+)\) \(size (\d+)\)', error_msg)
            if from_match and to_match:
                from_shape, from_size = from_match.groups()
                to_shape, to_size = to_match.groups()
                raise RAXShapeError(
                    f"\n[RAX] Reshape error in function '{func_name}'\n"
                    f"  Location: {source_file}:{start_line}\n"
                    f"\n  Cannot reshape array:"
                    f"\n    From: ({from_shape}) with {from_size} elements"
                    f"\n    To:   ({to_shape}) with {to_size} elements"
                    f"\n\n  The total number of elements must remain the same."
                ) from None
        
        # Einsum errors
        if "einsum" in error_msg.lower() or "Size of label" in error_msg:
            if "does not match" in error_msg:
                label_match = re.search(r"label '(\w)' for operand (\d+) \((\d+)\) does not match previous terms \((\d+)\)", error_msg)
                if label_match:
                    label, operand, size1, size2 = label_match.groups()
                    raise RAXShapeError(
                        f"\n[RAX] Einsum dimension mismatch in function '{func_name}'\n"
                        f"  Location: {source_file}:{start_line}\n"
                        f"\n  Dimension '{label}' has conflicting sizes:"
                        f"\n    Operand {operand}: {size1}"
                        f"\n    Previous term: {size2}"
                        f"\n\n  All occurrences of dimension '{label}' must have the same size."
                    ) from None
        
        # Check for JAX shape mismatch in operations
        if isinstance(e, TypeError) and "requires contracting dimensions" in error_msg:
            # Extract shapes from error
            shape_match = re.search(r'got \((\d+),?\) and \((\d+),?\)', error_msg)
            if shape_match:
                shape1, shape2 = shape_match.groups()
                
                raise RAXShapeError(
                    f"\n[RAX] Shape mismatch in function '{func_name}'\n"
                    f"  Location: {source_file}:{start_line}\n"
                    f"\n  Matrix multiplication has incompatible dimensions:"
                    f"\n    Left dimension:  {shape1}"
                    f"\n    Right dimension: {shape2}"
                    f"\n\n  These dimensions must match for matrix multiplication."
                ) from None
        
        # Other type errors
        if isinstance(e, TypeError):
            if "missing required positional argument" in error_msg:
                raise RAXValidationError(f"Invalid arguments for function '{func_name}': {e}")
            raise RAXMathError(f"Type error in function '{func_name}': {e}")
        
        # Shape/dimension errors
        if isinstance(e, ValueError):
            if "shape" in error_msg.lower() or "dimension" in error_msg.lower():
                raise RAXShapeError(f"Shape/dimension error in function '{func_name}': {e}")
            raise RAXMathError(f"Math/logic error in function '{func_name}': {e}")
        
        # Catch-all for other errors
        raise RAXValidationError(f"Validation failed for function '{func_name}': {e}")


def validate_function(func: Callable, enable_jit: bool = True, require_annotations: bool = True) -> Callable:
    """
    Wrap a function with RAX safety guarantees.
    
    This applies:
    1. Type/shape validation via beartype (if annotations exist)
    2. Compile-time validation via jax.make_jaxpr
    3. JIT compilation (if enabled)
    
    Args:
        func: The function to validate
        enable_jit: Whether to apply jax.jit
        require_annotations: Whether to require type annotations
        
    Returns:
        The wrapped function with safety guarantees
    """
    func_name = func.__name__
    original_func = func  # Keep reference to original function
    
    # Check for type annotations
    has_annotations = has_type_annotations(func)
    if require_annotations and not has_annotations:
        raise RAXValidationError(
            f"Function '{func_name}' is missing type annotations. "
            "All function parameters and return types must be annotated with jaxtyping shapes."
        )
    
    # Get source information for better errors
    try:
        source_file = inspect.getsourcefile(original_func)
        source_lines, start_line = inspect.getsourcelines(original_func)
    except:
        source_file = "unknown"
        start_line = 0
    
    # Apply beartype if annotations exist
    if has_annotations:
        func = beartype(func)
        # Store the original function reference on the beartype wrapper
        func._rax_original = original_func
    
    @functools.wraps(original_func)
    def wrapped(*args, **kwargs):
        try:
            # First, validate with jax.make_jaxpr to catch shape/math errors
            validate_jaxpr(func, *args, **kwargs)
            
            # If validation passed, execute the function
            return func(*args, **kwargs)
        except RAXShapeError:
            # Re-raise our shape errors directly (they're already formatted)
            raise
        except RAXValidationError:
            # Re-raise our validation errors directly
            raise
        except Exception as e:
            # Check if this is a beartype/jaxtyping error
            error_msg = str(e)
            
            # Handle beartype validation errors
            if "violates type hint" in error_msg and "dimension size" in error_msg:
                # Extract information from error
                param_name, actual_shape, expected_shape = extract_shape_info_from_error(error_msg)
                
                # Extract specific dimension mismatch
                dim_match = re.search(r'dimension size (\d+) does not equal (\d+)', error_msg)
                if dim_match:
                    actual_dim, expected_dim = dim_match.groups()
                    
                    # Format the error nicely
                    formatted_error = format_shape_error(
                        func_name, param_name, expected_shape, actual_shape,
                        source_file, start_line
                    )
                    raise RAXShapeError(formatted_error) from None
            
            # For other errors, wrap them
            raise RAXValidationError(
                f"\n[RAX] Error in function '{func_name}'"
                f"\n  Location: {source_file}:{start_line}"
                f"\n  {type(e).__name__}: {error_msg}"
            ) from None
    
    # Apply JIT if enabled
    if enable_jit:
        wrapped = jax.jit(wrapped)
    
    # Preserve the original function for introspection
    wrapped._rax_original = original_func
    wrapped._rax_validated = True
    
    return wrapped


def should_validate_function(func: Callable, module_name: str) -> bool:
    """
    Determine if a function should be validated by RAX.
    
    Skip:
    - Built-in functions
    - Functions from standard library
    - Functions from third-party modules (unless in __main__)
    - Already validated functions
    - Class methods (for now)
    """
    # Skip if already validated
    if hasattr(func, '_rax_validated'):
        return False
    
    # Skip built-ins
    if inspect.isbuiltin(func):
        return False
    
    # Skip if not a regular function
    if not inspect.isfunction(func):
        return False
    
    # Skip lambdas (usually don't have useful names/annotations)
    if func.__name__ == '<lambda>':
        return False
    
    # Get the module where the function is defined
    func_module = inspect.getmodule(func)
    if func_module is None:
        return False
    
    func_module_name = func_module.__name__
    
    # Only validate functions from __main__ or the script being run
    if func_module_name != '__main__' and func_module_name != module_name:
        return False
    
    # Skip private functions (starting with _)
    if func.__name__.startswith('_'):
        return False
    
    return True 