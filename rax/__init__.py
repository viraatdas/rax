"""RAX: A safer, compiler-like JAX frontend with shape/type safety and memory correctness guarantees."""

__version__ = "0.1.0"

from .runner import run_script
from .validator import validate_function

__all__ = ["run_script", "validate_function"] 