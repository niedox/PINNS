import torch
import pickle
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

from torch import nn
from scipy.interpolate import griddata

import time

np.random.seed(1234)
torch.manual_seed(1234)

#Global Variables
PLOT = 1
MAX_ITER_LBFGS = 50000
NOISE = 0.01

class PhysicsInformedNN(nn.Module):

    def __init__(self, X, u, layers_struc, lb, ub, max_iter_lbfgs = 50000):
        super(PhysicsInformedNN, self).__init__()

        #bounds
        self.lb = torch.tensor(lb, dtype=torch.float64)
        self.ub = torch.tensor(ub, dtype=torch.float64)

        #data
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float64, requires_grad=True)
        self.t = torch.tensor(X[:, 1:2], dtype=torch.float64, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float64, requires_grad=True)  # target data

        #layer struct
        self.layers_struc = layers_struc

        self.ux = None
        self.ut = None
        self.uxx = None

        #Neural net
        last_layer = nn.Linear(layers_struc[-2], 1)
        self.layers_u = \
            [nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
             for input_, output_ in
             zip(layers_struc[:-1], layers_struc[1:-1])] + \
            [last_layer]

        self.neural_net = nn.Sequential(*self.layers_u).double()


        # Initialize parameters
        tensor = torch.ones((2,), dtype=torch.float64)
        lambda_1 = tensor.new_tensor([0.0])
        lambda_2 = tensor.new_tensor([-6.0])
        self.lambda_1 = torch.nn.Parameter(lambda_1)
        self.lambda_2 = torch.nn.Parameter(lambda_2)


        #optimizer
        self.optimizer = torch.optim.LBFGS(self.parameters(), max_iter = max_iter_lbfgs,
                                        tolerance_grad=1.0 * np.finfo(float).eps)

        self.adam_optimizer = torch.optim.Adam(self.parameters())

        #Histories for debugging
        self.lossHist = []
        self.lambda1Hist = []
        self.lambda2Hist = []
        self.uxxHist = []
        self.uxHist = []
        self.utHist = []
        self.fLossHist = []
        self.uLossHist = []
        self.wHist = []

    def net_u(self, x, t):
        """Forward pass in the u net"""

        # Normalize input
        x_n = 2.0 * (x - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        t_n = 2.0 * (t - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0

        u = self.neural_net(torch.cat((x_n, t_n), -1))

        return u


    def net_f(self, x, t):
        """forward pass in the f net"""

        #u net fast forward pass
        u = self.net_u(x, t)

        #get lambdas
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)

        #differentiation
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones(len(u), 1),retain_graph=True,
                                  only_inputs=True, create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones(len(u), 1), retain_graph=True,
                                  only_inputs=True, create_graph=True)[0]
        u_xx= torch.autograd.grad(u_x, x, grad_outputs=torch.ones(len(u_x), 1), retain_graph = True,
                                  only_inputs=True, create_graph=True)[0]

        #Burger's equation
        f =  u_t + lambda_1 * u * u_x - lambda_2 * u_xx

        #store derivatives
        self.uxxHist.append(u_xx.detach())
        self.uxHist.append(u_x.detach())
        self.utHist.append(u_t.detach())

        return f

    def loss_fn(self, u_out, f_out, target):
        loss = nn.MSELoss()

        MSE_u = loss(torch.squeeze(u_out), torch.squeeze(target))
        MSE_f = loss(torch.squeeze(f_out), torch.zeros((len(f_out)), dtype=torch.float64))

        #store Losses
        self.uLossHist.append(MSE_u.item())
        self.fLossHist.append(MSE_f.item())

        return MSE_u + MSE_f

    def train_adam(self, iterations):

        for it in range(iterations):

            u_est = self.net_u(self.x, self.t)
            f_est = self.net_f(self.x, self.t)
            loss = self.loss_fn(u_est, f_est, self.u)

            loss.backward()

            self.adam_optimizer.step()
            self.adam_optimizer.zero_grad()

            #reset input gradients
            self.t.grad.zero_()
            self.x.grad.zero_()

            if (it%10) == 0:
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
                 (it, loss.item(), self.lambda_1.item(), np.exp(self.lambda_2.item())))

                #Store parameters history
                self.lossHist.append(loss.item())
                self.lambda1Hist.append(self.lambda_1.item())
                self.lambda2Hist.append(np.exp(self.lambda_2.item()))
                hh = self.optimizer.param_groups[0]['params'][18].T.detach().numpy()
                self.wHist.append(hh.copy())

    def train_lbfgs(self):

        """Function for lbfgs optimizer"""

        def closure():
            self.optimizer.zero_grad()

            if self.x.grad is not None:
                self.x.grad.zero_()
                self.t.grad.zero_()

            u_est = self.net_u(self.x, self.t)
            f_est = self.net_f(self.x, self.t)
            loss = self.loss_fn(u_est, f_est, self.u)

            if loss.requires_grad:
                loss.backward()

            print('Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.6f' %
             (loss.item(), self.lambda_1.item(), np.exp(self.lambda_2.item())))

            # Store parameters history
            self.lossHist.append(loss.item())
            self.lambda1Hist.append(self.lambda_1.item())
            self.lambda2Hist.append(np.exp(self.lambda_2.item()))
            hh = self.optimizer.param_groups[0]['params'][18].T.detach().numpy()
            self.wHist.append(hh.copy())

            return loss

        self.optimizer.step(closure)


    def train(self, adam_it):
        self.train_adam(adam_it)
        self.train_lbfgs()

    def predict(self, x, t):
        x = torch.unsqueeze(torch.tensor(x, dtype=torch.float64, requires_grad=True), -1)
        t = torch.unsqueeze(torch.tensor(t, dtype=torch.float64, requires_grad=True), -1)
        u = self.net_u(x, t)
        f = self.net_f(x, t)
        return u.detach().numpy(), f.detach().numpy()

def xavier_init_routine(m):
    """function for weight initialization"""
    if type(m) == nn.Linear:

        #manual xavier initialization
        xavier_stddev = np.sqrt(2 / (m.in_features + m.out_features))
        torch.nn.init.trunc_normal_(m.weight, mean = 0, std = xavier_stddev)

        m.bias.data.fill_(0)


if __name__ == "__main__":

    nu = 0.01 / np.pi #lambda 2 true value
    N_u = 2000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    #load data
    print("loading data")
    data = scipy.io.loadmat('./Data/burgers_shock.mat')
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

    #Build model
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub, max_iter_lbfgs=MAX_ITER_LBFGS)
    model.apply(xavier_init_routine) #initialize nn

    #TRAIN
    model.train(0)

    if PLOT:
        #Plot results
        plt.plot(model.lossHist)
        plt.title("Loss pytorch")
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.show()

        plt.plot(model.lambda1Hist)
        plt.title("l1 pytorch")
        plt.xlabel("iteration")
        plt.ylabel("lambda 1")
        plt.show()

        plt.plot(model.lambda2Hist)
        plt.title("l2 pytorch")
        plt.xlabel("iteration")
        plt.ylabel("lambda 2")
        plt.show()

        plt.plot(model.uLossHist)
        plt.title("u loss pytorch")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("u loss")
        plt.show()

        plt.plot(model.fLossHist)
        plt.title("f loss pytorch")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("f loss")
        plt.show()


    u_pred, f_pred = model.predict(X_star[:, 0], X_star[:, 1])

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    lambda_1_value = model.lambda_1.item()
    lambda_2_value = np.exp(model.lambda_2.item())

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))


    ######################################################################
    ######################## Noisy Data ###############################
    ######################################################################

    noise = NOISE
    u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])

    # Build model
    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub, max_iter_lbfgs=MAX_ITER_LBFGS)
    model.apply(xavier_init_routine)  # initialize nn

    # TRAIN
    model.train(1000)

    if PLOT:
        #Plot results
        plt.plot(model.lossHist)
        plt.title("Loss pytorch")
        plt.xlabel("iteration")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.show()

        plt.plot(model.lambda1Hist)
        plt.title("l1 pytorch")
        plt.xlabel("iteration")
        plt.ylabel("lambda 1")
        plt.show()

        plt.plot(model.lambda2Hist)
        plt.title("l2 pytorch")
        plt.xlabel("iteration")
        plt.ylabel("lambda 2")
        plt.show()

        plt.plot(model.uLossHist)
        plt.title("u loss pytorch")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("u loss")
        plt.show()

        plt.plot(model.fLossHist)
        plt.title("f loss pytorch")
        plt.yscale("log")
        plt.xlabel("iteration")
        plt.ylabel("f loss")
        plt.show()



    u_pred, f_pred = model.predict(X_star[:, 0], X_star[:, 1])

    error_u = np.linalg.norm(u_star - u_pred, 2) / np.linalg.norm(u_star, 2)

    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')

    lambda_1_value = model.lambda_1.item()
    lambda_2_value = np.exp(model.lambda_2.item())

    error_lambda_1 = np.abs(lambda_1_value - 1.0) * 100
    error_lambda_2 = np.abs(lambda_2_value - nu) / nu * 100

    print('Error u: %e' % (error_u))
    print('Error l1: %.5f%%' % (error_lambda_1))
    print('Error l2: %.5f%%' % (error_lambda_2))

