from pathlib import Path
import jax
import jax.numpy as jnp
import pretty_traceback
import optax
from tqdm import tqdm
from compare.datagen import make_ndarray, Batch

pretty_traceback.install()

# Hyperparams
LR = 0.003
N_EPOCHS = 2
BATCH_SIZE = 32

# Data config
DATAROOT = Path.home() / "mldata" / "binclass"

key = jax.random.PRNGKey(0)


def init_linear(key, in_features, out_features):
    W = jax.nn.initializers.glorot_normal()(key, (in_features, out_features))
    var = jnp.sqrt(out_features)
    b = jax.random.uniform(key, (out_features,), minval=-var, maxval=var)
    if out_features == 1:
        W = W.squeeze()
        b = b.squeeze()
    return {"W": W, "b": b}


def net(params, X):
    for param in params[:-1]:
        W, b = param["W"], param["b"]
        X = jax.nn.relu(X @ W + b)
    W, b = params[-1]["W"], params[-1]["b"]
    # p = jax.nn.sigmoid(X @ W.T + b)
    logits = X @ W + b
    return logits


# @jax.jit
def loss(params, batch):
    X, y = batch.X, batch.y
    logits = net(params, X)
    bce = optax.sigmoid_binary_cross_entropy(logits, y)
    mean_bce = jnp.mean(bce)
    return mean_bce, logits


def batches(X, y, batch_size):
    stop = 0
    while stop < X.shape[0]:
        start = stop
        stop = start + batch_size
        batch_X = X[start:stop]
        batch_y = y[start:stop]
        yield Batch(batch_X, batch_y)


def train():
    X_train, X_val, y_train, y_val = make_ndarray(DATAROOT)
    X_train, X_val, y_train, y_val = (
        jnp.array(X_train),
        jnp.array(X_val),
        jnp.array(y_train),
        jnp.array(y_val),
    )
    params = [init_linear(key, 20, 32), init_linear(key, 32, 1)]
    optim = optax.adam(learning_rate=LR)
    opt_state = optim.init(params)

    @jax.jit
    def step(params, opt_state, batch):
        (train_loss, logits), grads = jax.value_and_grad(loss, has_aux=True)(
            params, batch
        )
        updates, opt_state = optim.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, train_loss, logits

    for epoch in range(1, N_EPOCHS + 1):
        train_losses, train_accuracies = [], []
        print("\nTRAIN:")
        for batch in tqdm(
            batches(X_train, y_train, BATCH_SIZE), total=X_train.shape[0] // BATCH_SIZE
        ):
            params, opt_state, train_loss, logits = step(params, opt_state, batch)
            train_losses.append(train_loss)

            probs = jax.nn.sigmoid(logits)
            y_hat = jnp.where(probs > 0.5, 1, 0)
            train_accuracy = jnp.mean(jnp.where(y_hat == batch.y, 1, 0))
            train_accuracies.append(train_accuracy)

        avg_train_loss = jnp.mean(jnp.array(train_losses))
        avg_train_acc = jnp.mean(jnp.array(train_accuracies))

        val_losses, val_accuracies = [], []
        print("\nVALIDATE:")
        for batch in tqdm(
            batches(X_val, y_val, 10_000), total=X_val.shape[0] // 10_000
        ):
            val_loss, logits = loss(params, batch)
            val_losses.append(val_loss)
            probs = jax.nn.sigmoid(logits)
            y_hat = jnp.where(probs > 0.5, 1, 0)
            val_accuracy = jnp.mean(jnp.where(y_hat == batch.y, 1, 0))
            val_accuracies.append(val_accuracy)

        avg_val_loss = jnp.mean(jnp.array(val_losses))
        avg_val_acc = jnp.mean(jnp.array(val_accuracies))

        print(
            f"\n{epoch}: Val Loss={avg_val_loss:.5f}, Train Loss={avg_train_loss:.5f}, Val Acc = {avg_val_acc:.3f}, Train Acc = {avg_train_acc:.3f}\n"
        )


def main():
    train()


if __name__ == "__main__":
    main()
