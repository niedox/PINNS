import numpy as np
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from torch import nn
from utilities import weighted_mse_loss, xavier_init_routine

np.random.seed(1234)
torch.manual_seed(1234)


class PINN_single(nn.Module):
    def __init__(
        self,
        t,
        I,
        layers_struc,
        lb,
        ub,
        v,
        linesearch="strong_wolfe",
        device="cpu",
        max_iter_lbfgs=50000,
        Ek=-96.2,
        gk=0.1,
    ):
        super(PINN_single, self).__init__()

        self.dev = device
        # bounds
        self.lb = torch.tensor(lb, dtype=torch.float64, device=device)
        self.ub = torch.tensor(ub, dtype=torch.float64, device=device)

        # ion_channels
        self.Ek = Ek
        self.gk = gk

        # data
        self.t = torch.tensor(t, dtype=torch.float64, requires_grad=True, device=device)
        self.I = torch.tensor(
            I, dtype=torch.float64, requires_grad=True, device=device
        )  # target data
        self.act = torch.tensor(
            v, dtype=torch.float64, requires_grad=True, device=device
        )

        # layer struct
        self.layers_struc = layers_struc

        # Neural nets
        last_layer = nn.Linear(layers_struc[-2], 1)
        layers = [
            nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
            for input_, output_ in zip(layers_struc[:-1], layers_struc[1:-1])
        ] + [last_layer]

        self.net_i = nn.Sequential(*layers).double().to(device)

        last_layer = nn.Linear(layers_struc[-2], 1)
        layers = [
            nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
            for input_, output_ in zip(layers_struc[:-1], layers_struc[1:-1])
        ] + [last_layer]

        self.net_m = nn.Sequential(*layers).double().to(device)

        last_layer = nn.Linear(layers_struc[-2], 1)
        layers = [
            nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
            for input_, output_ in zip(layers_struc[:-1], layers_struc[1:-1])
        ] + [last_layer]

        self.net_h = nn.Sequential(*layers).double().to(device)

        # Initialize parameters
        tensor = torch.ones((2,), dtype=torch.float64, device=device)
        m_tau = tensor.new_tensor([1.0], device=device)
        m_inf = tensor.new_tensor([1.0], device=device)
        h_tau = tensor.new_tensor([1.0], device=device)
        h_inf = tensor.new_tensor([0.0], device=device)

        self.m_tau = torch.nn.Parameter(m_tau)
        self.h_tau = torch.nn.Parameter(h_tau)
        self.m_inf = torch.nn.Parameter(m_inf)
        self.h_inf = torch.nn.Parameter(h_inf)

        # optimizer
        self.optimizer = torch.optim.LBFGS(
            self.parameters(),
            max_iter=max_iter_lbfgs,
            tolerance_grad=1.0 * np.finfo(float).eps,
            line_search_fn=linesearch,
        )  # , tolerance_change=1e-10/6)

        self.adam_optimizer = torch.optim.Adam(self.parameters())

        # Histories for debugging
        self.lossHist = []

        self.f1LossHist = []
        self.f2LossHist = []
        self.f3LossHist = []
        self.f4LossHist = []
        self.f5LossHist = []
        self.iLossHist = []
        self.m0Hist = []

    def nets(self, t):
        """Forward pass in the u net"""

        # Normalize input
        t_n = 2.0 * (t - self.lb) / (self.ub - self.lb) - 1.0

        I = self.net_i(torch.unsqueeze(t_n, -1))
        m = self.net_m(torch.unsqueeze(t_n, -1))
        h = self.net_h(torch.unsqueeze(t_n, -1))

        return I, m, h

    def normalize_i(self, inp):
        inp_n = 2.0 * (inp - self.I.min()) / (self.I.max() - self.I.min()) - 1.0
        return inp_n

    def net_f(self, t):
        """computes the ODEs and functions to be used in the loss"""

        I, m, h = self.nets(t)

        _, m0, h0 = self.nets(torch.tensor([0], dtype=torch.float64, device=self.dev))

        f1 = self.normalize_i(self.gk * m * m * h * (self.act - self.Ek)) - I  #

        m_t = torch.autograd.grad(
            m,
            t,
            grad_outputs=torch.ones(len(m), 1, device=self.dev),
            retain_graph=True,
            only_inputs=True,
            create_graph=True,
        )[0]
        h_t = torch.autograd.grad(
            h,
            t,
            grad_outputs=torch.ones(len(h), 1, device=self.dev),
            retain_graph=True,
            only_inputs=True,
            create_graph=True,
        )[0]

        if self.m_tau == 0:
            self.m_tau = torch.nn.Parameter(CTensor([0.01]))
        if self.h_tau == 0:
            self.h_tau = torch.nn.Parameter(CTensor([0.01]))

        f2 = 5 * ((self.m_inf - torch.squeeze(m)) / (self.m_tau * 5) - m_t)
        f3 = 500 * ((self.h_inf - torch.squeeze(h)) / (self.h_tau * 500) - h_t)

        f4 = m0 ** 2

        f5 = (h0 - torch.tensor([1], dtype=torch.float64, device=self.dev)) ** 2

        return f1, f2, f3, f4, f5

    def loss_fn(self, f1, f2, f3, f4, f5, Ik_out, target):
        target = self.normalize_i(target)
        loss = nn.MSELoss()

        MSE_i = loss(torch.squeeze(Ik_out), torch.squeeze(target))
        MSE_f1 = loss(
            torch.squeeze(f1),
            torch.zeros(f1.size(), dtype=torch.float64, device=self.dev).squeeze(),
        )
        MSE_f2 = loss(
            torch.squeeze(f2),
            torch.zeros(f2.size(), dtype=torch.float64, device=self.dev).squeeze(),
        )
        MSE_f3 = loss(
            torch.squeeze(f3),
            torch.zeros(f3.size(), dtype=torch.float64, device=self.dev).squeeze(),
        )

        # store Losses
        self.iLossHist.append(MSE_i.item())
        self.f1LossHist.append(MSE_f1.item())
        self.f2LossHist.append(MSE_f2.item())
        self.f3LossHist.append(MSE_f3.item())
        self.f4LossHist.append(f4.item())
        self.f5LossHist.append(f5.item())

        return MSE_i + MSE_f1 + 10 * MSE_f2 + 10 * MSE_f3 + f4 + f5

    def train_adam(self, iterations):
        """train using ADAM optimizer. Nb of iterations and batch size can be specified"""

        for it in range(iterations):

            i_est, _, _ = self.nets(self.t)
            f1, f2, f3, f4, f5 = self.net_f(self.t)
            loss = self.loss_fn(f1, f2, f3, f4, f5, i_est, self.I)

            loss.backward()

            self.adam_optimizer.step()
            self.adam_optimizer.zero_grad()

            # reset input gradients
            self.t.grad.zero_()

            if (it % 10) == 0:
                print("It: %d, Loss: %.3e" % (it, loss.item()))

    def train_lbfgs(self):

        """Function for lbfgs optimizer"""

        def closure():
            self.optimizer.zero_grad()

            if self.I.grad is not None:
                self.I.grad.zero_()
                self.t.grad.zero_()

            i_est, _, _ = self.nets(self.t)
            f1, f2, f3, f4, f5 = self.net_f(self.t)
            loss = self.loss_fn(f1, f2, f3, f4, f5, i_est, self.I)

            if loss.requires_grad:
                loss.backward()

            print("Loss: %.3e" % loss.item())

            # Store parameters history
            self.lossHist.append(loss.item())

            return loss

        self.optimizer.step(closure)

    def train(self, adam_it):
        self.train_adam(adam_it)
        self.train_lbfgs()

    def predict(self, t):
        """forward pass into the model"""
        t = torch.tensor(t, dtype=torch.float64, requires_grad=True, device=self.dev)
        i, _, _ = self.nets(t)
        if i.is_cuda:
            i = i.cpu()
        return i.cpu().detach().numpy()
