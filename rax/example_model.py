"""Example neural network model demonstrating RAX safety features.

This script shows how RAX enforces shape/type safety for JAX code.
Run with: rax run example_model.py
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float, Array


def relu(x: Float[Array, "..."]) -> Float[Array, "..."]:
    """ReLU activation function."""
    return jnp.maximum(0, x)


def dense(
    x: Float[Array, "batch features_in"],
    w: Float[Array, "features_in features_out"],
    b: Float[Array, "features_out"]
) -> Float[Array, "batch features_out"]:
    """Fully connected layer."""
    return jnp.dot(x, w) + b


def layer_norm(x: Float[Array, "batch features"]) -> Float[Array, "batch features"]:
    """Simple layer normalization."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    std = jnp.std(x, axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-6)


def mlp_forward(
    x: Float[Array, "batch 784"],
    w1: Float[Array, "784 256"],
    b1: Float[Array, "256"],
    w2: Float[Array, "256 128"],
    b2: Float[Array, "128"],
    w3: Float[Array, "128 10"],
    b3: Float[Array, "10"]
) -> Float[Array, "batch 10"]:
    """Simple 3-layer MLP for MNIST."""
    # First layer
    h1 = dense(x, w1, b1)
    h1 = layer_norm(h1)
    h1 = relu(h1)
    
    # Second layer
    h2 = dense(h1, w2, b2)
    h2 = layer_norm(h2)
    h2 = relu(h2)
    
    # Output layer
    logits = dense(h2, w3, b3)
    return logits


def softmax(logits: Float[Array, "batch classes"]) -> Float[Array, "batch classes"]:
    """Softmax activation."""
    exp_logits = jnp.exp(logits - jnp.max(logits, axis=-1, keepdims=True))
    return exp_logits / jnp.sum(exp_logits, axis=-1, keepdims=True)


def main():
    """Run a forward pass with example data."""
    print("🚀 RAX Example: Safe Neural Network Forward Pass")
    print("=" * 50)
    
    # Initialize parameters with correct shapes
    key = jax.random.PRNGKey(42)
    keys = jax.random.split(key, 6)
    
    # Weight initialization (Xavier/Glorot)
    w1 = jax.random.normal(keys[0], (784, 256)) * jnp.sqrt(2.0 / 784)
    b1 = jnp.zeros(256)
    
    w2 = jax.random.normal(keys[1], (256, 128)) * jnp.sqrt(2.0 / 256)
    b2 = jnp.zeros(128)
    
    w3 = jax.random.normal(keys[2], (128, 10)) * jnp.sqrt(2.0 / 128)
    b3 = jnp.zeros(10)
    
    # Create batch of random input data
    batch_size = 32
    x = jax.random.normal(keys[3], (batch_size, 784))
    
    print(f"📊 Input shape: {x.shape}")
    print(f"🧮 Total parameters: {w1.size + b1.size + w2.size + b2.size + w3.size + b3.size:,}")
    
    # Forward pass
    print("\n🔄 Running forward pass...")
    logits = mlp_forward(x, w1, b1, w2, b2, w3, b3)
    probs = softmax(logits)
    
    print(f"✅ Output logits shape: {logits.shape}")
    print(f"✅ Output probabilities shape: {probs.shape}")
    
    # Extract value for printing (to avoid JIT tracing issues)
    prob_sum = float(jnp.sum(probs, axis=-1)[0])
    print(f"📈 Probability sum per sample: {prob_sum:.6f}")
    
    # Get predictions
    predictions = jnp.argmax(probs, axis=-1)
    print(f"\n🎯 Predictions for first 5 samples: {predictions[:5]}")
    
    print("\n✨ Success! All shape validations passed.")
    print("💡 Try modifying the shapes to see RAX's safety features in action!")


if __name__ == "__main__":
    main() 