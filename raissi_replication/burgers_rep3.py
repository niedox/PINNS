import torch
import itertools

from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from torch.nn import MSELoss
from torch import nn
from torch.utils.data import Dataset, DataLoader


"""class ExperimentData(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx]"""

class PhysicsInformedNN(nn.Module):

    def __init__(self, X, u, layers_struc, lb, ub):
        super(PhysicsInformedNN, self).__init__()

        #bounds
        self.lb = torch.tensor(lb, dtype=torch.float32)
        self.ub = torch.tensor(ub, dtype=torch.float32)

        #data
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(X[:, 1:2], dtype=torch.float32, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32)  # target data

        """self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(X[:, 1:2], dtype=torch.float32, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32, requires_grad=True) #target data"""

        #layer struct
        self.layers_struc = layers_struc


        last_layer = nn.Linear(layers_struc[-2], 1)
        self.layers_u = \
            [nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
             for input_, output_ in
             zip(layers_struc[:-2], layers_struc[1:-2])] + \
            [last_layer]

        self.net_u = nn.Sequential(*self.layers_u)


        # Initialize parameters
        tensor = torch.ones((2,), dtype=torch.float32)
        lambda_1 = tensor.new_tensor([0.0], requires_grad = False)
        lambda_2 = tensor.new_tensor([-6.0], requires_grad = False)
        self.lambda_1 = torch.nn.Parameter(lambda_1)
        self.lambda_2 = torch.nn.Parameter(lambda_2)



        """self.inputLayer = nn.Linear(layers[0], layers[1])
        self.hiddenLayers = nn.ModuleList([nn.Linear(layers[i+1], layers[i+2]) for i in range(len(layers)-2)]) ##TODO check nb layes
        self.outputLayer = nn.Linear(layers[-2], layers[-1])
        self.xavier_init()
        self.model = nn.Sequential(self.inputLayer, self.hiddenLayers, self.outputLayer)"""

        self.optimizer = None

    def forward(self, x, t):
        # Normalize
        x = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        t = 2.0 * (t - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0

        # net u
        u = self.net_u(torch.cat((x, t)))

        # net f
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        
        u_t = torch.autograd.grad(u, t, retain_graph=True, allow_unused=True)[0]
        u_x = torch.autograd.grad(u, x, retain_graph=True, create_graph=True, allow_unused=True)[0]  # TODO check gradients
        u_xx = torch.autograd.grad(u_x, x, retain_graph=True, allow_unused=True)[0]

        f = u_t + lambda_1 * u * u_x - lambda_2 * u_xx

        return u, f

    def loss_fn(self, u_out, f_out, target):
        loss = nn.MSELoss()
        MSE_u = loss(u_out, target)
        MSE_f = loss(f_out, torch.tensor([0], dtype=torch.float32))
        return MSE_u + MSE_f

    def train(self, nIter):

        for i in range(nIter):
            running_loss = 0.0

            for (x_in, t_in, target) in zip(self.x, self.t, self.u):
                def closure():
                    self.optimizer.zero_grad()
                    u_est, f_est = self(x_in, t_in)
                    loss = self.loss_fn(u_est, f_est, target)
                    loss.backward()

                    return loss

                self.optimizer.step(closure)

                loss = closure()
                print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, self.lambda_1.item(), np.exp(self.lambda_2.item())))
                running_loss += loss.item()

            print("iter_loss:", running_loss)

def xavier_init_routine(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0)

if __name__ == "__main__":
    nu = 0.01 / np.pi

    N_u = 2000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('../Data/burgers_shock.mat')

    t = data['t'].flatten()[:, None]
    x = data['x'].flatten()[:, None]
    Exact = np.real(data['usol']).T

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Domain bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    ######################################################################
    ######################## Noiseless Data ###############################
    ######################################################################
    noise = 0.0

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]


    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.apply(xavier_init_routine)

    # model.optimizer = torch.optim.SGD(par, lr=0.01)
    for p in model.parameters():
        print(p)

    model.optimizer = torch.optim.LBFGS(model.parameters(), max_iter=50000, max_eval=50000,
                                        tolerance_grad=1.0 * np.finfo(float).eps)

    model.train(1)

    u_pred, f_pred = model.predict(X_star)

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    lambda_2_value = np.exp(lambda_2_value)

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))