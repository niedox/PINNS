import sys
#sys.path.insert(0, '../../Utilities/')

import torch
import itertools


from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from torch.nn import MSELoss
from torch import nn
#from plotting import newfig, savefig
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import matplotlib.gridspec as gridspec
import time



class PhysicsInformedNN:

    def __init__(self, X, u, layers, lb, ub):
        super(PhysicsInformedNN, self).__init__()

        #bounds
        self.lb = lb
        self.ub = ub

        #variables
        self.x = torch.tensor(X[:, 0:1], dtype=torch.float32, requires_grad=True)
        self.t = torch.tensor(X[:, 1:2], dtype=torch.float32, requires_grad=True)
        self.u = torch.tensor(u, dtype=torch.float32) #target data

        self.layers = layers

        # Initialize NNs
        #self.weights, self.biases = self.initialize_NN(layers)

        # Initialize parameters
        tensor = torch.ones((2,), dtype=torch.float32)
        self.lambda_1 = tensor.new_tensor([0.0], requires_grad = False, dtype=torch.float32)
        self.lambda_2 = tensor.new_tensor([-6.0], requires_grad = False, dtype=torch.float32)


        ####MODEL
        last_layer = nn.Linear(layers[-2], 1)
        self.model_layers = []
        for (input_, output_) in zip(layers[0:-1], layers[1:-1]):
            self.model_layers.append(nn.Linear(input_, output_))
            self.model_layers.append(nn.Tanh())
        self.model_layers.append(last_layer)


        self.model = nn.Sequential(*self.model_layers)
        self.model.float()
        print(self.model)
        """self.inputLayer = nn.Linear(layers[0], layers[1])
        self.hiddenLayers = nn.ModuleList([nn.Linear(layers[i+1], layers[i+2]) for i in range(len(layers)-2)]) ##TODO check nb layes
        self.outputLayer = nn.Linear(layers[-2], layers[-1])
        self.xavier_init()
        self.model = nn.Sequential(self.inputLayer, self.hiddenLayers, self.outputLayer)"""

        self.optimizer = None

        """self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))"""

        """# 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)"""

    def xavier_init(self):

        def xavier_init_routine(m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform(m.weight)
                m.bias.data.fill_(0)

        #initialisation
        self.model.apply(xavier_init_routine)


    def neural_net(self, X):
        Y = self.model(X)
        return Y


    def net_u(self, x, t):
        b_x = torch.cat((x, t))
        u = self.neural_net(b_x)
        return u


    def net_f(self, x, t):
        lambda_1 = self.lambda_1
        lambda_2 = torch.exp(self.lambda_2)
        u = self.net_u(x, t)
        u_t = torch.autograd.grad(u, t, retain_graph=True)
        u_x = torch.autograd.grad(u, x, retain_graph=True, create_graph=True) #TODO check gradients
        u_xx = torch.autograd.grad(u_x, x, retain_graph=True)
        f = u_t[0] + lambda_1 * u * u_x[0] - lambda_2 * u_xx[0]

        return f

    def loss_fn(self, u_est, f_est, target):
        loss = nn.MSELoss()
        MSE_u = loss(u_est, target)
        MSE_f = loss(f_est, torch.tensor(0, dtype=torch.float32))
        print("MSEU: ", MSE_u)
        print("MSEF: ", MSE_f)

        return MSE_u + MSE_f

    def train(self):
        #tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}

        for (x_in, t_in, target) in zip(self.x, self.t, self.u):
            def closure():
                self.optimizer.zero_grad()
                u_est = self.net_u(x_in, t_in)
                f_est = self.net_f(x_in, t_in)
                loss = self.loss_fn(u_est, f_est, target)
                print("LOSS: ", loss.item())
                #for param in self.model.parameters():
                 #   print("W: ", param.data)

                loss.backward()
                print("lambdas: ", self.lambda_1, " // ", self.lambda_2)

                return loss

            self.optimizer.step(closure)



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
    aa = model.model.parameters()
    print("!! ", model.model.parameters())
    par = list(model.model.parameters()) + list(model.lambda_1) + list(model.lambda_2)
    model.optimizer = torch.optim.LBFGS(par, max_iter=50000, max_eval=50000,
                                           tolerance_grad=1.0 * np.finfo(float).eps)

    model.train()

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

    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
    noise = 0.01
    u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])

    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    model.train(10000)

    u_pred, f_pred = model.predict(X_star)

    lambda_1_value_noisy = model.sess.run(model.lambda_1)
    lambda_2_value_noisy = model.sess.run(model.lambda_2)
    lambda_2_value_noisy = np.exp(lambda_2_value_noisy)

    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0) * 100
    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu) / nu * 100

    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))

    ######################################################################
    ############################# Plotting ###############################
    ######################################################################

    """fig, ax = newfig(1.0, 1.4)
    ax.axis('off')

    ####### Row 0: u(t,x) ##################
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1.0 / 3.0 + 0.06, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow',
                  extent=[t.min(), t.max(), x.min(), x.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=2,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
    ax.set_title('$u(t,x)$', fontsize=10)

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1.0 / 3.0 - 0.1, bottom=1.0 - 2.0 / 3.0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, Exact[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.set_title('$t = 0.25$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, Exact[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, Exact[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, U_pred[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(t,x)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75$', fontsize=10)

    ####### Row 3: Identified PDE ##################
    gs2 = gridspec.GridSpec(1, 3)
    gs2.update(top=1.0 - 2.0 / 3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)

    ax = plt.subplot(gs2[:, :])
    ax.axis('off')
    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
    s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
    s3 = r'Identified PDE (1\% noise) & '
    s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
    s5 = r'\end{tabular}$'
    s = s1 + s2 + s3 + s4 + s5
    ax.text(0.1, 0.1, s)

    # savefig('./figures/Burgers_identification')"""