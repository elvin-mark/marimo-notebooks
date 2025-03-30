import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Simple MLP in written in numpy""")
    return


@app.cell
def _():
    import math
    import numpy as np
    from sklearn.datasets import load_digits
    return load_digits, math, np


@app.cell
def _(load_digits):
    raw_data = load_digits()

    train_x = raw_data["data"] / 16.0
    train_y = raw_data["target"]
    return raw_data, train_x, train_y


@app.cell
def _(np):
    def relu(x):
        return np.maximum(x, 0)


    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))


    def tanh(x):
        return (1 - np.exp(-2 * x)) / (1 + np.exp(-2 * x))
    return relu, sigmoid, tanh


@app.cell
def _(np, sigmoid, tanh):
    def model_forward(x, params):
        w1, b1, w2, b2, w3, b3 = params
        o1 = x @ w1 + b1
        o2 = tanh(o1)
        o3 = o2 @ w2 + b2
        o4 = tanh(o3)
        o5 = o4 @ w3 + b3
        return [o1, o2, o3, o4, o5]


    def model_backward(x, params, o, dL):
        w1, b1, w2, b2, w3, b3 = params
        o1, o2, o3, o4, o5 = o

        db3 = np.sum(dL, axis=0)
        dw3 = o4.T @ dL
        dL4 = dL @ w3.T

        # dL3 = dL4 * (o4 > 0) # relu
        # dL3 = dL4 * o4 * (1 - o4) # sigmoid
        dL3 = dL4 * (1 - o4**2)

        db2 = np.sum(dL3, axis=0)
        dw2 = o2.T @ dL3
        dL2 = dL3 @ w2.T

        # dL1 = dL2 * (o2 > 0) # relu
        # dL1 = dL2 * o2 * (1 - o2) # sigmoid
        dL1 = dL2 * (1 - o2**2)

        db1 = np.sum(dL1, axis=0)
        dw1 = x.T @ dL1
        return [dw1, db1, dw2, db2, dw3, db3]


    def mse_loss(o, y):
        o = sigmoid(o)
        y_ = np.zeros(o.shape)
        y_[list(range(len(y))), y] = 1
        L = np.mean((o - y_) ** 2) / 2
        dL = (o - y_) / (o.shape[0] * o.shape[1])
        return L, o * (1 - o) * dL


    def cross_entropy_loss(o, y):
        pass


    def optim(params, dparams, lr=1):
        return [w - lr * dw for w, dw in zip(params, dparams)]
    return cross_entropy_loss, model_backward, model_forward, mse_loss, optim


@app.cell
def _(
    math,
    model_backward,
    model_forward,
    mse_loss,
    np,
    optim,
    train_x,
    train_y,
):
    w1 = np.random.randn(64, 32)
    b1 = np.random.randn(
        32,
    )
    w2 = np.random.randn(32, 32)
    b2 = np.random.randn(
        32,
    )

    w3 = np.random.randn(32, 10)
    b3 = np.random.randn(
        10,
    )

    params = [w1, b1, w2, b2, w3, b3]
    lr = 1
    batch_size = 64
    for epoch in range(50):
        tot_loss = 0
        for i in range(math.ceil(len(train_y) / batch_size)):
            x = train_x[i : batch_size * (i + 1)]
            y = train_y[i : batch_size * (i + 1)]
            o = model_forward(x, params)
            L, dL = mse_loss(o[-1], y)
            dparams = model_backward(x, params, o, dL)
            params = optim(params, dparams, lr=lr)
            tot_loss += L
        print(tot_loss)
    return (
        L,
        b1,
        b2,
        b3,
        batch_size,
        dL,
        dparams,
        epoch,
        i,
        lr,
        o,
        params,
        tot_loss,
        w1,
        w2,
        w3,
        x,
        y,
    )


@app.cell
def _(model_forward, np, params, train_x, train_y):
    tmp_ = model_forward(train_x, params)
    idxs = np.argmax(tmp_[-1], axis=1)
    print(idxs[:40])
    print(train_y[:40])
    return idxs, tmp_


if __name__ == "__main__":
    app.run()
