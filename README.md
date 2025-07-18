# RAX: A Safe JAX Frontend

RAX is a compiler-like frontend for JAX that adds shape/type safety and memory correctness guarantees to your JAX code, without requiring any code changes beyond adding type annotations.

## Installation

### From GitHub
```bash
pip install git+https://github.com/viraatdas/rax.git
```

### For Development
```bash
git clone -b rax https://github.com/viraatdas/rax.git
cd rax
pip install -e .
```

## Quick Start

1. **Annotate your functions** with jaxtyping:

```python
# my_model.py
import jax.numpy as jnp
from jaxtyping import Float, Array

def dense(x: Float[Array, "batch 128"], w: Float[Array, "128 64"]) -> Float[Array, "batch 64"]:
    return jnp.dot(x, w)

def model(x: Float[Array, "batch 128"]) -> Float[Array, "batch 64"]:
    w = jnp.ones((128, 64))
    return dense(x, w)

# Run your model
x = jnp.ones((32, 128))
output = model(x)
print(f"Output shape: {output.shape}")
```

2. **Run with RAX**:

```bash
rax run my_model.py
```

## What RAX Does

When you run `rax run script.py`, RAX:

1. **Scans for user-defined functions** in your script
2. **Validates type annotations** - Requires all functions to have jaxtyping annotations
3. **Enforces shapes at runtime** via beartype integration
4. **Traces functions symbolically** using `jax.make_jaxpr` to catch shape/math errors
5. **JIT-compiles by default** for performance
6. **Monkeypatches JAX operations** to validate shapes 

## Safety Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| Type & shape correctness | `@beartype` + jaxtyping |
| Math logic validation | `jax.make_jaxpr(fn)(*args)` |
| Compilation before execution | `jax.jit` (automatic) |
| Early failure on errors | Compile-time trace validation |
| Memory predictability | Enforced size-consistent ops |

## Examples

### Valid Code
```python
# This runs successfully
def add(x: Float[Array, "10"], y: Float[Array, "10"]) -> Float[Array, "10"]:
    return x + y

result = add(jnp.ones(10), jnp.ones(10))
```

### Invalid Code (Caught by RAX)
```python
# This fails at validation time
def bad_dot(x: Float[Array, "32 100"], w: Float[Array, "128 64"]) -> Float[Array, "32 64"]:
    return jnp.dot(x, w)  # Shape mismatch!

# RAX output:
# [RAX] Shape/dimension error in function 'bad_dot': 
# dot shape mismatch: (32, 100) · (128, 64)
```

## CLI Options

```bash
rax run script.py [options]

Options:
  --no-jit           Disable automatic JIT compilation
  --no-monkeypatch   Disable JAX operation monkeypatching
  --verbose          Enable verbose output
```

## Requirements

- JAX code must use jaxtyping annotations for all function parameters and returns
- No other code changes needed - RAX is fully compatible with existing JAX code

## Future Extensions

- `rax compile` - Validate without execution
- `rax trace` - Output compiled JAXPR for debugging
- `rax lint` - Detect missing annotations
- `rax export` - Export to XLA/MLIR format