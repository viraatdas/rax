"""Script runner with RAX safety validation."""

import functools
import inspect
import sys
from pathlib import Path
from types import FunctionType
from typing import Any, Dict

from .validator import validate_function, RAXValidationError
from .monkeypatch import apply_monkeypatch, remove_monkeypatch


class LazyValidator:
    """Wrapper that validates a function on first call."""
    
    def __init__(self, func: FunctionType, enable_jit: bool = True, verbose: bool = False):
        self._func = func
        self._validated_func = None
        self._enable_jit = enable_jit
        self._verbose = verbose
        self.__name__ = func.__name__
        self.__doc__ = func.__doc__
        
    def __call__(self, *args, **kwargs):
        # Validate on first call
        if self._validated_func is None:
            if self._verbose:
                print(f"[RAX] Validating function '{self.__name__}' on first call...")
            
            try:
                self._validated_func = validate_function(
                    self._func, 
                    enable_jit=self._enable_jit
                )
                if self._verbose:
                    print(f"[RAX] ✓ Function '{self.__name__}' validated successfully")
            except RAXValidationError as e:
                raise RAXValidationError(f"[RAX] {e}")
        
        return self._validated_func(*args, **kwargs)
    
    def __getattr__(self, name):
        # Forward attribute access to the original function
        return getattr(self._func, name)


class RAXNamespace(dict):
    """Custom namespace that wraps functions with RAX validation."""
    
    def __init__(self, enable_jit: bool = True, verbose: bool = False):
        super().__init__()
        self._enable_jit = enable_jit
        self._verbose = verbose
        self._wrapped_functions = set()
    
    def __setitem__(self, key: str, value: Any):
        # If it's a user-defined function, wrap it
        if isinstance(value, FunctionType) and not key.startswith('_') and key not in self._wrapped_functions:
            # Skip main function - it's usually for orchestration, not computation
            if key == 'main':
                super().__setitem__(key, value)
                return
                
            # Check if it's a user-defined function (not from imports)
            if hasattr(value, '__module__'):
                module = value.__module__
                # Wrap if it's from __main__ or None (defined in exec)
                if module == '__main__' or module is None:
                    if self._verbose:
                        print(f"[RAX] Found function '{key}' (module: {module})")
                    
                    value = LazyValidator(value, self._enable_jit, self._verbose)
                    self._wrapped_functions.add(key)
        
        super().__setitem__(key, value)


def run_script(
    script_path: str,
    enable_jit: bool = True,
    enable_monkeypatch: bool = True,
    verbose: bool = False
) -> None:
    """
    Run a Python script with RAX safety guarantees.
    
    Args:
        script_path: Path to the script to run
        enable_jit: Whether to JIT-compile functions
        enable_monkeypatch: Whether to monkeypatch JAX operations
        verbose: Whether to print verbose output
    """
    script_path = Path(script_path).resolve()
    
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    if verbose:
        print(f"[RAX] Running script: {script_path}")
        print(f"[RAX] JIT compilation: {'enabled' if enable_jit else 'disabled'}")
        print(f"[RAX] Operation monkeypatching: {'enabled' if enable_monkeypatch else 'disabled'}")
    
    # Apply monkeypatch if enabled
    if enable_monkeypatch:
        apply_monkeypatch(verbose=verbose)
    
    try:
        # Add the script's directory to sys.path
        script_dir = str(script_path.parent)
        if script_dir not in sys.path:
            sys.path.insert(0, script_dir)
        
        # Read the script
        with open(script_path, 'r') as f:
            source_code = f.read()
        
        # Create RAX namespace that auto-validates functions
        namespace = RAXNamespace(enable_jit=enable_jit, verbose=verbose)
        namespace.update({
            '__name__': '__main__',
            '__file__': str(script_path),
            '__doc__': None,
            '__package__': None,
        })
        
        if verbose:
            print("[RAX] Executing script with RAX validation...")
        
        # Compile and execute
        compiled_code = compile(source_code, str(script_path), 'exec')
        exec(compiled_code, namespace)
        
    finally:
        # Clean up monkeypatch
        if enable_monkeypatch:
            remove_monkeypatch(verbose=verbose)
        
        # Remove script directory from sys.path
        if script_dir in sys.path:
            sys.path.remove(script_dir) 