"""
Three things I can try tomorrow -
1. Use optax loss function
2. Standardize X
3. Examine the jaxpr of grad(loss)
"""
from collections import namedtuple
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
import jax.numpy as jnp
import jax
import optax
from torch.utils.data import Dataset, DataLoader
import pretty_traceback

pretty_traceback.install()

Batch = namedtuple("Batch", ["X", "y"])

key = jax.random.PRNGKey(0)


class BinDataset(Dataset):
    def __init__(self, X, y):
        self._X = X
        self._y = y

    def __getitem__(self, idx):
        return self._X[idx], self._y[idx]

    def __len__(self):
        return self._X.shape[0]


def collate(samples):
    xs, ys = zip(*samples)
    X = jnp.vstack(jnp.array(x) for x in xs)
    y = jnp.array([y for y in ys])
    return Batch(X, y)


def build_model():
    def model(params, x):
        for param in params[:-1]:
            W, b = param["W"], param["b"]
            x = jax.nn.relu(W @ x + b)
        W, b = params[-1]["W"], params[-1]["b"]
        p = jax.nn.sigmoid(W @ x + b)
        if jnp.isnan(p):
            print("Got it!")
        return p

    def vmodel(params, X):
        for param in params[:-1]:
            W, b = param["W"], param["b"]
            X = jax.nn.relu(X @ W.T + b)
        W, b = params[-1]["W"], params[-1]["b"]
        # p = jax.nn.sigmoid(X @ W.T + b)
        logits = X @ W.T + b
        return logits

    return vmodel

    # return jax.vmap(model, in_axes=(None, 0))


def init_linear(key, in_features, out_features):
    W = jax.nn.initializers.glorot_normal()(key, (out_features, in_features))
    var = jnp.sqrt(out_features)
    b = jax.random.uniform(key, (out_features,), minval=-var, maxval=var)
    if out_features == 1:
        W = W.squeeze()
        b = b.squeeze()
    return {"W": W, "b": b}


def build_loss(model):
    def loss(params, batch):
        X, y = batch.X, batch.y
        logits = model(params, X)
        # logp = jnp.log(p)
        # logp = jnp.where(logp < -100, -100, logp)
        # logq = jnp.log(1 - p)
        # logq = jnp.where(logq < -100, -100, logq)
        # bce = -(y * logp + (1 - y) * logq)
        # mean_bce = jnp.mean(bce)
        # if jnp.isnan(mean_bce):
        #     raise RuntimeError()
        # return mean_bce
        bce = optax.sigmoid_binary_cross_entropy(logits, y)
        mean_bce = jnp.mean(bce)
        return mean_bce

    return loss


def wirecheck(traindl, params, model, loss):
    batch = next(iter(traindl))
    model(params, batch.X)
    loss(params, batch)


def train(hyperparams, trainset, valset):
    model = build_model()
    optim = optax.adam(learning_rate=hyperparams["learning_rate"])
    init_params = [init_linear(key, 20, 8), init_linear(key, 8, 1)]
    loss = build_loss(model)
    traindl = DataLoader(
        trainset,
        batch_size=hyperparams["batch_size"],
        shuffle=True,
        drop_last=True,
        collate_fn=collate,
    )

    wirecheck(traindl, init_params, model, loss)
    print("Wirecheck successful!")

    @jax.jit
    def step(params, opt_state, batch):
        loss_val, grads = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        # if jnp.isnan(grads[0]["W"][0, 0]) or jnp.isnan(grads[1]["W"][0]):
        #     print("Loss: ", loss_val)
        #     print("Max W: ", jnp.max(params[0]["W"]), " ", jnp.max(params[1]["W"]))
        #     print("Min W: ", jnp.min(params[0]["W"]), " ", jnp.min(params[1]["W"]))
        #     raise RuntimeError()
        return new_params, opt_state, loss_val, grads

    opt_state = optim.init(init_params)
    params = init_params
    for epoch in range(hyperparams["n_epochs"]):
        for i, batch in enumerate(traindl):
            new_params, opt_state, loss_val, grads = step(params, opt_state, batch)
            # print(
            #     f"Max grad = {jnp.max(grads[0]['W'])}    Min grad = {jnp.min(grads[0]['W'])}"
            # )
            # print(
            #     f"Max grad = {jnp.max(grads[1]['W'])}    Min grad = {jnp.min(grads[1]['W'])}"
            # )
            if i % 1000 == 0:
                print(f"Epoch {epoch} Step {i}: Loss = {loss_val:.5f}")

            params = new_params

    return params


def main():
    hyperparams = {"learning_rate": 0.005, "n_epochs": 2, "batch_size": 32}

    X, y = make_classification(
        n_classes=2,
        n_samples=1_000_000,
        random_state=0,
        n_features=20,
        n_informative=10,
        n_redundant=7,
        n_repeated=3,
        flip_y=0.05,
        class_sep=0.5,
    )
    print("max: ", np.max(X))
    print("min: ", np.min(X))
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
    trainset = BinDataset(X_train, y_train)
    params = train(hyperparams, trainset, None)  # noqa


if __name__ == "__main__":
    main()
