import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

import matplotlib.pyplot as plt

import Black_scholes
class NeuralNet(torch.nn.Module):
    def __init__(self, d, q1, q2):
        super(NeuralNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d, q1),
            nn.ReLU(),
            nn.Linear(q1, q2),
            nn.ReLU(),
            nn.Linear(q2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        propagation = self.net(x)
        return propagation


class Neural_Network_Pricer:
    def __init__(self, mod, opt, nb_samples, epochs=100):
        self.mod = mod
        self.opt = opt
        self.nb_samples = nb_samples
        #self.network = network
        self.epochs = epochs


    def simulate_samples(self, rounds):
        samples = []
        for _ in range(rounds):
            path = self.mod.asset(self.opt.T, self.opt.dates)
            samples.append(path)
        fitted = np.stack([samples[round] for round in range(len(samples))], axis=1)
        return torch.FloatTensor(fitted)
    
        

    def simulate_conditionally(self, n, samples, J):
        z_n = []
        for stocks in samples[n]:
            all_continuations = []
            for _ in range(J):
                continuation = self.mod.asset_conditionally(self.opt.T, self.opt.dates, n, stocks)
                all_continuations.append(continuation)
            z_n.append(all_continuations)
        return np.array(z_n)




    def initiate_values(self, samples):
        """Initiates the option values at date N before backward computing"""
        values = np.zeros((self.opt.dates + 1, self.nb_samples))
        for round in range(len(samples)):
            values[self.opt.dates, round] = self.opt.payoff(samples[self.opt.dates][round], self.opt.dates, self.mod.r)
        return values



    def train(self, samples):
        """Trains the neural network models for the optimal stopping problem, using the input samples."""
        f = np.zeros((self.opt.dates + 1, self.nb_samples))
        stopping = np.array([self.opt.dates for _ in range(self.nb_samples)])
        f[self.opt.dates, :] = 1
        trained_models = [None] * self.opt.dates
        losses = []
        for n in range(self.opt.dates - 1, -1, -1):
            neural_network = NeuralNet(self.mod.size, self.mod.size + 40, self.mod.size + 40)
            optimizer = torch.optim.Adam(neural_network.parameters(), lr = 0.0001)
            for epoch in range(self.epochs):
                F = neural_network.forward(samples[n])
                optimizer.zero_grad()
                reward = torch.zeros((self.nb_samples))
                for round in range(self.nb_samples):
                    reward[round] = -self.opt.payoff(samples[n][round], n, self.mod.r) * F[round] - self.opt.payoff(samples[int(stopping[round])][round], stopping[round], self.mod.r) * (1 - F[round])
                back_propagation_reward = reward.mean()
                back_propagation_reward.backward()
                optimizer.step() 
            trained_models[n] = neural_network
            adjusted_F = F.detach().numpy().reshape(self.nb_samples)
            f[n, :] = (adjusted_F > 0.5).astype(float)
            stopping = np.array([max(range(len(f[:,col])), key=lambda row: f[row,col]) for col in range(f.shape[1])])
            losses.append((np.min(adjusted_F) + np.max(adjusted_F))/2)
            print('n = {}, losses : {}'.format(n, losses[-1]))
        #plt.plot(losses[1:])
        #plt.title("Training Loss Over Time")
        #plt.xlabel("Dates")
        #plt.ylabel("Loss")
        #plt.show()
        return trained_models
