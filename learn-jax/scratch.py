import jax

key = jax.random.PRNGKey(0)

init = jax.nn.initializers.variance_scaling(
    scale=1, mode="fan_in", distribution="uniform"
)
tp = init(key=key, shape=(8,))
