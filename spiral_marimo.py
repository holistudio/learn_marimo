import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import torch
    from torch import nn, optim

    from torch.utils.data import Dataset
    from torch.utils.data import DataLoader
    from torch.utils.data import random_split

    import torch.functional as F

    import fastbook
    from fastbook import plt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return nn, plt, torch


@app.cell
def _(torch):
    seed = 12345
    torch.manual_seed(seed)
    N = 1000  # num_samples_per_class
    K = 3     # num_classes
    return K, N


@app.cell
def _(K, N, torch):
    # Generate spirals

    t = torch.linspace(0, 1.5, N)
    a = 0.8 * t + 0.2  # amplitude 0.2 → 1.0
    X = list()
    y = list()
    for k in range(K):
        θ = (2 * t + k) * 2 * torch.pi / K + 0.2 * torch.randn(N)
        X.append(torch.stack((a * θ.sin(), a * θ.cos()), dim=1))
        y.append(torch.zeros(N, dtype=torch.long).fill_(k))
    X = torch.cat(X)
    y = torch.cat(y)
    return X, y


@app.cell
def _(X, plt, y):
    plt.figure(figsize=(4.5, 4))
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xticks([-1, -0.5, 0, 0.5, 1])
    plt.yticks([-1, -0.5, 0, 0.5, 1])

    plt.show()
    return


@app.cell
def _(nn):
    class DeepLinear(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),  # BREAD
                nn.Linear(hidden_size, hidden_size), # BREAD            <= NOT A SANDWICH!
                nn.Linear(hidden_size,output_size)   # BREAD
            )
    
        def forward(self, xb):
            score = self.net(xb)
            return score

    return


@app.cell
def _(nn):
    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),  # BREAD
                nn.ReLU(),                           # STUFF
                nn.Linear(hidden_size, hidden_size), # BREAD           <= THAT'S A SANDWICH!
                nn.ReLU(),                           # STUFF
                nn.Linear(hidden_size,output_size)   # BREAD
            )
        def forward(self, xb):
            score = self.net(xb)
            return score

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
