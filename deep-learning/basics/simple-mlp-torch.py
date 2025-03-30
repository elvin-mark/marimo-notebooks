import marimo

__generated_with = "0.11.31"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md("""# PyTorch Basics for Deep Learning""")
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    from sklearn.datasets import load_digits
    import matplotlib.pyplot as plt
    import marimo as mo
    return load_digits, mo, nn, plt, torch


@app.cell
def _(load_digits):
    # Loading the data 
    raw_data = load_digits()
    return (raw_data,)


@app.cell
def _(mo):
    idx = mo.ui.slider(0,1000,1)
    idx
    return (idx,)


@app.cell
def _(idx, plt, raw_data):
    # Visualizing one sample of the data (image)
    plt.imshow(raw_data["data"][idx.value].reshape(8,8))
    plt.show()
    # Printing the label of the image
    print(raw_data["target"][idx.value])
    return


@app.cell
def _(raw_data, torch):
    # Transforming the data into tensors
    image_data = torch.tensor(raw_data["data"] / 16).float()
    target_data = torch.tensor(raw_data["target"]).long()

    # Creating the dataset and dataloader
    train_ds = torch.utils.data.TensorDataset(image_data, target_data)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=32)
    return image_data, target_data, train_dl, train_ds


@app.cell
def _(nn):
    # Defining the model
    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    return (model,)


@app.cell
def _(model, nn, torch):
    # Defining the loss function
    crit = nn.CrossEntropyLoss()

    # And the optimizer
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    return crit, optim


@app.cell
def _(crit, model, optim, train_dl):
    # Training the model for 10 epochs
    for epoch in range(10):
        tot_loss = 0
        for x, y in train_dl:
            optim.zero_grad()
            o = model(x)
            l = crit(o, y)
            l.backward()
            optim.step()
            tot_loss += l.item()
        print("Loss: {}".format(tot_loss / len(train_dl)))
    return epoch, l, o, tot_loss, x, y


if __name__ == "__main__":
    app.run()
