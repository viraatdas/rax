# RAX

**RAX** is a compiler-like frontend for JAX that enforces shape/type safety and mathematical correctness before your code runs.

It adds:

- Type and shape validation for all functions
- Compile-time detection of invalid matrix operations
- Automatic `jax.jit` compilation
- Memory safety by design
- Full compatibility with existing JAX code

No need to rewrite your code—just add shape/type annotations using [`jaxtyping`](https://github.com/patrick-kidger/jaxtyping).

---

## What It Does

When you run:

```bash
rax run my_model.py
```

RAX will:
1. Scan your script for all user-defined functions
2. Ensure each function is annotated with jaxtyping
3. Wrap each function in:
  - beartype for runtime type/shape enforcement
  - jax.make_jaxpr(fn)(*args) to statically validate shape/math logic
  - jax.jit for automatic compilation

4. Fail early if:
  - A function is missing annotations
  - Shapes don’t match
  - Matrix operations are invalid
5. Monkeypatch math ops like jnp.dot, reshape, einsum for eager-mode validation

## Installation
```bash
pip install rax
```

## Usage
Write your JAX code like normal:
```python
import jax.numpy as jnp
from jaxtyping import Float, Array

def dense(x: Float[Array, "batch 128"], w: Float[Array, "128 64"]) -> Float[Array, "batch 64"]:
    return jnp.dot(x, w)

out = dense(jnp.ones((32, 128)), jnp.ones((128, 64)))
```

Then run it with:

```bash
rax run my_model.py
```

If you pass the wrong shape:
```python
out = dense(jnp.ones((32, 100)), jnp.ones((128, 64)))
```

RAX will catch the error before execution:

```
[RAX] Shape/math error in `dense`: dot shape mismatch: (32, 100) · (128, 64)
```

## Goals
- Compatible with any existing JAX project
- Only requirement: add jaxtyping annotations
- Memory-safe by construction
- Detects logic errors before they hit XLA

## Planned Features
- `rax run`: compile + run (default)
- `rax trace`: show JAXPR trace
- `rax lint`: check annotation coverage
- `rax export`: emit MLIR, XLA, or serialized IR