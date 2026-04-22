"""Smoke test: confirm JAX sees the GPU and a matmul actually runs on it."""
import jax
import jax.numpy as jnp

print("jax version:", jax.__version__)
print("devices:", jax.devices())
print("default backend:", jax.default_backend())

x = jnp.ones((4096, 4096))
y = x @ x
y.block_until_ready()
print("matmul shape:", y.shape, "device:", y.device)
print("OK" if jax.default_backend() == "gpu" else "FAIL: JAX fell back to CPU")
