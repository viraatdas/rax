"""Function validation and wrapping for RAX safety guarantees."""

import functools
import inspect
from typing import Any, Callable, Optional, get_type_hints

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
    try:
        # Use jax.make_jaxpr to trace the function
        jaxpr = jax.make_jaxpr(func)(*args, **kwargs)
        # If we get here, the function traced successfully
    except TypeError as e:
        if "missing required positional argument" in str(e):
            # This is likely from calling with wrong number of args
            raise RAXValidationError(f"Invalid arguments for function '{func.__name__}': {e}")
        raise RAXMathError(f"Type error in function '{func.__name__}': {e}")
    except ValueError as e:
        # Common for shape mismatches
        error_msg = str(e)
        if "shape" in error_msg.lower() or "dimension" in error_msg.lower():
            raise RAXShapeError(f"Shape/dimension error in function '{func.__name__}': {e}")
        raise RAXMathError(f"Math/logic error in function '{func.__name__}': {e}")
    except Exception as e:
        # Catch-all for other JAX errors
        raise RAXValidationError(f"Validation failed for function '{func.__name__}': {e}")


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
    
    # Check for type annotations
    has_annotations = has_type_annotations(func)
    if require_annotations and not has_annotations:
        raise RAXValidationError(
            f"Function '{func_name}' is missing type annotations. "
            "All function parameters and return types must be annotated with jaxtyping shapes."
        )
    
    # Apply beartype if annotations exist
    if has_annotations:
        func = beartype(func)
    
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        # First, validate with jax.make_jaxpr to catch shape/math errors
        try:
            validate_jaxpr(func, *args, **kwargs)
        except RAXValidationError as e:
            # Re-raise with more context
            raise type(e)(f"[RAX] {e}")
        
        # If validation passed, execute the function
        return func(*args, **kwargs)
    
    # Apply JIT if enabled
    if enable_jit:
        wrapped = jax.jit(wrapped)
    
    # Preserve the original function for introspection
    wrapped._rax_original = func
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