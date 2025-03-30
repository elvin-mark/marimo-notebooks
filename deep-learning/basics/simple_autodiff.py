import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""# Simple Autodiff engine""")
    return


@app.cell
def _():
    class AddBackward:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __call__(self, loss):
            self.x.backward(loss)
            self.y.backward(loss)


    class SubBackward:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __call__(self, loss):
            self.x.backward(loss)
            self.y.backward(-loss)


    class MulBackward:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __call__(self, loss):
            self.x.backward(self.y.data * loss)
            self.y.backward(self.x.data * loss)


    class DivBackward:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __call__(self, loss):
            self.x.backward(loss / self.y.data)
            self.y.backward(-self.x.data * loss / self.y.data**2)


    class Variable:
        def __init__(self, data):
            self.data = data
            self.grad = 0
            self.backward_fn = None

        def __add__(self, x):
            o = Variable(self.data + x.data)
            o.backward = AddBackward(self, x)
            return o

        def __sub__(self, x):
            o = Variable(self.data - x.data)
            o.backward = SubBackward(self, x)
            return o

        def __mul__(self, x):
            o = Variable(self.data * x.data)
            o.backward = MulBackward(self, x)
            return o

        def __truediv__(self, x):
            o = Variable(self.data / x.data)
            o.backward = DivBackward(self, x)
            return o

        def backward(self, loss):
            if self.backward_fn is None:
                self.grad += loss
            else:
                self.backward_fn(loss)
    return AddBackward, DivBackward, MulBackward, SubBackward, Variable


@app.cell
def _(Variable):
    x = Variable(2)
    y = Variable(3)
    z = Variable(4)
    w = Variable(5)

    o = x*y -z/w + x*w
    o.backward(1)
    return o, w, x, y, z


@app.cell
def _(w, x, y, z):
    print(x.grad)
    print(y.grad)
    print(z.grad)
    print(w.grad)
    return


if __name__ == "__main__":
    app.run()
