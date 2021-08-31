#######
####### Fits model on data with different activation voltages

import time
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch.cuda as tc
from torch import nn
from utilities import xavier_init_routine, weighted_mse_loss


# seed for reproducibility
np.random.seed(1234)
torch.manual_seed(1234)


class PINN_multi(nn.Module):
    def __init__(
        self,
        t,
        v,
        I,
        layers_struc,
        lb,
        ub,
        max_iter_lbfgs=50000,
        line_sh="strong_wolfe",
        Ek=-96.2,
        gk=0.1,
        device="cpu",
    ):

        super(PINN_multi, self).__init__()

        # device
        self.dev = device

        # bounds
        self.lb = torch.tensor(lb, device=self.dev)
        self.ub = torch.tensor(ub, device=self.dev)
        # ion channel
        self.Ek = torch.tensor(Ek, device=self.dev)
        self.gk = torch.tensor(gk, device=self.dev)

        # data
        self.v = torch.tensor(
            v, dtype=torch.float64, requires_grad=True, device=self.dev
        ).unsqueeze(-1)
        self.t = torch.tensor(
            t, dtype=torch.float64, requires_grad=True, device=self.dev
        ).unsqueeze(-1)
        self.I = torch.tensor(
            I, dtype=torch.float64, requires_grad=True, device=self.dev
        )

        # Build data structure
        self.t_rep, self.v_rep, self.input = self.construct_input(self.t, self.v)

        # layer structure
        self.layers_struc = layers_struc

        # Build neural nets
        last_layer = nn.Linear(layers_struc[-2], 1)
        layers = [
            nn.Sequential(nn.Linear(input_, output_), nn.Tanh()).double()
            for input_, output_ in zip(layers_struc[:-1], layers_struc[1:-1])
        ] + [last_layer]

        self.net_i = nn.Sequential(*layers).double().to(self.dev)

        last_layer = nn.Linear(layers_struc[-2], 1)
        layers = [
            nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
            for input_, output_ in zip(layers_struc[:-1], layers_struc[1:-1])
        ] + [last_layer]

        self.net_m = nn.Sequential(*layers).double().to(self.dev)

        last_layer = nn.Linear(layers_struc[-2], 1)
        layers = [
            nn.Sequential(nn.Linear(input_, output_), nn.Tanh())
            for input_, output_ in zip(layers_struc[:-1], layers_struc[1:-1])
        ] + [last_layer]

        self.net_h = nn.Sequential(*layers).double().to(self.dev)

        # Initialize parameters
        tensor = torch.ones((2,), dtype=torch.float64)
        m_tau = tensor.new_tensor(np.full(len(v), 1.0), device=self.dev)
        m_inf = tensor.new_tensor(np.full(len(v), 1.0), device=self.dev)
        h_tau = tensor.new_tensor(np.full(len(v), 1.0), device=self.dev)
        h_inf = tensor.new_tensor(np.full(len(v), 0.0), device=self.dev)

        self.m_tau = torch.nn.Parameter(m_tau)
        self.h_tau = torch.nn.Parameter(h_tau)
        self.m_inf = torch.nn.Parameter(m_inf)
        self.h_inf = torch.nn.Parameter(h_inf)

        # optimizer
        self.optimizer = torch.optim.LBFGS(
            self.parameters(),
            max_iter=max_iter_lbfgs,
            tolerance_grad=1.0 * np.finfo(float).eps,
            line_search_fn=line_sh,
        )

        self.adam_optimizer = torch.optim.Adam(self.parameters())

        # Histories for debugging
        self.lossHist = []
        self.f1LossHist = []
        self.f2LossHist = []
        self.f3LossHist = []
        self.f4LossHist = []
        self.f5LossHist = []
        self.iLossHist = []

    def construct_input(self, t, v):
        """construct input structure from time and voltage data. t and v have to be torch tensors"""

        # normalize input
        # t = 2.0 * (t - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        # v = 2.0 * (v - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0

        t_rep = torch.tile(t, (1, len(v))).transpose(0, 1)  # replicate t vector
        v_rep = torch.tile(v, (1, len(t)))  # replicate v vector

        t_n = 2.0 * (t_rep - self.lb[0]) / (self.ub[0] - self.lb[0]) - 1.0
        v_n = 2.0 * (v_rep - self.lb[1]) / (self.ub[1] - self.lb[1]) - 1.0

        input = torch.stack((t_n, v_n)).transpose(0, 2)  # 2D input

        return t_rep, v_rep, input

    def nets(self, input):
        """Forward pass into the 3 neural nets"""

        I = self.net_i(input)
        m = self.net_m(input)
        h = self.net_h(input)

        return I, m, h

    def net_f(self, input):
        """computes the ODEs and functions to be used in the loss"""

        # forward pass
        I, m, h = self.nets(input)

        # f1
        f1 = self.normalize_i(self.gk * m * m * h * (self.v - self.Ek)) - I

        # f2-f3
        m = m.squeeze()
        h = h.squeeze()
        # derivative
        m_t = torch.autograd.grad(
            m,
            self.t_rep,
            grad_outputs=torch.ones(
                len(h), len(self.v), device=self.dev, dtype=torch.float64
            ).squeeze(),
            retain_graph=True,
            only_inputs=True,
            create_graph=True,
        )[0]

        h_t = torch.autograd.grad(
            h,
            self.t_rep,
            grad_outputs=torch.ones(
                len(h), len(self.v), dtype=torch.float64, device=self.dev
            ).squeeze(),
            retain_graph=True,
            only_inputs=True,
            create_graph=True,
        )[0]

        f2 = 50 * (
            (self.m_inf.unsqueeze(-1) - m.transpose(0, 1))
            / (self.m_tau.unsqueeze(-1) * 50)
            - m_t
        )
        f3 = 500 * (
            (self.h_inf.unsqueeze(-1) - h.transpose(0, 1))
            / (self.h_tau.unsqueeze(-1) * 500)
            - h_t
        )

        # f4-f5

        _, _, input0 = self.construct_input(
            torch.tensor([0], dtype=torch.float64, device=self.dev), self.v
        )
        _, m0, h0 = self.nets(input0)

        f4 = m0
        f5 = h0 - torch.tensor([1], dtype=torch.float64, device=self.dev)

        return f1, f2, f3, f4, f5, I

    def loss_fn(self, f1, f2, f3, f4, f5, Ik_out, target):
        loss = nn.MSELoss()

        MSE_i = loss(torch.squeeze(Ik_out), torch.squeeze(self.normalize_i(target)))
        MSE_f1 = loss(f1, torch.zeros(f1.shape, dtype=torch.float64, device=self.dev))
        MSE_f2 = loss(f2, torch.zeros(f2.shape, dtype=torch.float64, device=self.dev))
        MSE_f3 = loss(f3, torch.zeros(f3.shape, dtype=torch.float64, device=self.dev))
        MSE_f4 = loss(f4, torch.zeros(f4.shape, dtype=torch.float64, device=self.dev))
        MSE_f5 = loss(f5, torch.zeros(f5.shape, dtype=torch.float64, device=self.dev))

        # store Losses
        self.iLossHist.append(MSE_i.item())
        self.f1LossHist.append(MSE_f1.item())
        self.f2LossHist.append(MSE_f2.item())
        self.f3LossHist.append(MSE_f3.item())

        return MSE_i + MSE_f1 + 10 * MSE_f2 + 10 * MSE_f3 + MSE_f4 + MSE_f5

    def normalize_i(self, inp):
        inp_n = 2.0 * (inp - self.I.min()) / (self.I.max() - self.I.min()) - 1.0
        return inp_n

    def train_adam(self, iterations):
        """train using ADAM optimizer"""

        for it in range(iterations):

            # forward pass
            f1, f2, f3, f4, f5, i_est = self.net_f(self.input)
            loss = self.loss_fn(f1, f2, f3, f4, f5, i_est, self.I)

            # backward pass
            loss.backward(retain_graph=True)
            self.adam_optimizer.step()

            self.adam_optimizer.zero_grad()  # reset gradients

            # display losses
            if (it % 10) == 0:
                print("It: %d, Loss: %.3e" % (it, loss.item()))
                # Store parameters history
                self.lossHist.append(loss.item())

    def train_lbfgs(self):
        """Function for lbfgs optimizer"""

        def closure():

            # reset gradients
            self.optimizer.zero_grad()
            if self.I.grad is not None:
                self.I.grad.zero_()

            # forward pass
            f1, f2, f3, f4, f5, i_est = self.net_f(self.input)
            loss = self.loss_fn(f1, f2, f3, f4, f5, i_est, self.I)

            # backward pass
            if loss.requires_grad:
                loss.backward(retain_graph=True)

            print("Loss: %.3e" % loss.item())

            # Store parameters history
            self.lossHist.append(loss.item())

            return loss

        self.optimizer.step(closure)

    def train(self, adam_it):
        self.train_adam(adam_it)
        self.train_lbfgs()
        print("training over")

    def predict(self, Xin):
        """feed-forward pass into the model. Xin has to be a stacked 3D tensor (as built in self._init_)"""
        i, _, _ = self.nets(Xin)

        return i.cpu().detach().numpy()
