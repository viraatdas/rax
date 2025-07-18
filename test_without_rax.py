import jax.numpy as jnp

def linear(x, w):
    return jnp.dot(x, w)

# Silent bug until runtime — crashes deep in XLA
output = linear(jnp.ones((32, 100)), jnp.ones((128, 64)))

