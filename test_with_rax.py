import jax.numpy as jnp
from jaxtyping import Float, Array

def linear(x: Float[Array, "batch 128"], w: Float[Array, "128 64"]) -> Float[Array, "batch 64"]:
    return jnp.dot(x, w)

# RAX will fail *before* compiling:
# [RAX] Shape/math error in `linear`: dot shape mismatch: (32, 100) · (128, 64)
output = linear(jnp.ones((32, 100)), jnp.ones((128, 64)))

