import torch
from torch import nn
from torch.autograd import Variable

from torch.optim import Adam, LBFGS
from torch.utils.data import Dataset, DataLoader

temp = [1,2,3,4,5]
list(zip(temp, temp[:-1]))


class NNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes, loss, sigmoid=False):
        super().__init__()

        self.input_dim = input_dim
        self.layer_sizes = hidden_layer_sizes
        self.iter = 0
        # The loss function could be MSE or BCELoss depending on the problem
        self.lossFct = loss

        # We leave the optimizer empty for now to assign flexibly
        self.optim = None

        hidden_layer_sizes = [input_dim] + hidden_layer_sizes
        last_layer = nn.Linear(hidden_layer_sizes[-1], 1)
        self.layers = \
            [nn.Sequential(nn.Linear(input_, output_), nn.ReLU())
             for input_, output_ in
             zip(hidden_layer_sizes, hidden_layer_sizes[1:])] + \
            [last_layer]

        # The output activation depends on the problem
        if sigmoid:
            self.layers = self.layers + [nn.Sigmoid()]

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.layers(x)
        return x

    def train(self, data_loader, epochs, validation_data=None):

        for epoch in range(epochs):
            running_loss = self._train_iteration(data_loader)
            val_loss = None
            if validation_data is not None:
                y_hat = self(validation_data['X'])
                val_loss = self.lossFct(input=y_hat, target=validation_data['y']).detach().numpy()
                print('[%d] loss: %.3f | validation loss: %.3f' %
                      (epoch + 1, running_loss, val_loss))
            else:
                print('[%d] loss: %.3f' %
                      (epoch + 1, running_loss))

    def _train_iteration(self, data_loader):
        running_loss = 0.0
        for i, (X, y) in enumerate(data_loader):

            X = X.float()
            y = y.unsqueeze(1).float()

            X_ = Variable(X, requires_grad=True)
            y_ = Variable(y)

            ### Comment out the typical gradient calculation
            #             pred = self(X)
            #             loss = self.lossFct(pred, y)

            #             self.optim.zero_grad()
            #             loss.backward()

            ### Add the closure function to calculate the gradient.
            def closure():
                if torch.is_grad_enabled():
                    self.optim.zero_grad()
                output = self(X_)
                loss = self.lossFct(output, y_)
                if loss.requires_grad:
                    loss.backward()
                return loss

            self.optim.step(closure)

            # calculate the loss again for monitoring
            output = self(X_)
            loss = closure()
            running_loss += loss.item()

        return running_loss

    # I like to include a sklearn like predict method for convenience
    def predict(self, X):
        X = torch.Tensor(X)
        return self(X).detach().numpy().squeeze()


class ExperimentData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

X,y = make_classification(n_samples=20000, n_features=100, n_informative=80, n_redundant=0, n_clusters_per_class=20, class_sep=1, random_state=123)

X, X_val, y, y_val = train_test_split(X, y, test_size=0.5, random_state=123)

# Don't forget to prepare the data for the DataLoader
data = ExperimentData(X,y)

INPUT_SIZE = X.shape[1]
EPOCHS=5 # Few epochs to avoid overfitting

pred_val = {}

HIDDEN_LAYER_SIZE = []



data_loader = DataLoader(data, batch_size=X.shape[0])
net = NNet(INPUT_SIZE, HIDDEN_LAYER_SIZE, loss = nn.BCELoss(), sigmoid=True)
net.optim = LBFGS(net.parameters(), history_size=10, max_iter=4)
net.train(data_loader, EPOCHS, validation_data = {"X":torch.Tensor(X_val), "y":torch.Tensor(y_val).unsqueeze(1) })

print(net.predict(X_val))
