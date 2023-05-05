import numpy as np
from scipy.special import hermite
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as functional

import matplotlib.pyplot as plt

import Black_scholes
import Neural_network

def predict_values(pricer, samples, trained_models):
        """ Computes the predicted mean of all simulated payoffs at corresponding stopping times. Useful for lower bound."""
        f = np.zeros((pricer.opt.dates + 1, len(samples[0])))
        stopping = np.array([pricer.opt.dates for _ in range(len(samples[0]))])
        f[pricer.opt.dates, :] = 1
        values = pricer.initiate_values(samples)
        for n in range(pricer.opt.dates - 1, -1, -1):
            network = trained_models[n]
            F = network(samples[n])
            adjusted_F = F.detach().numpy().reshape(len(samples[0]))
            f[n, :] = (adjusted_F > 0.5).astype(float)
            stopping = np.array([max(range(len(f[:,col])), key=lambda row: f[row,col]) for col in range(f.shape[1])])
            for round in range(len(samples[0])):
                values[n, round] = np.exp((pricer.mod.r * pricer.opt.T / float(pricer.opt.dates)) * (stopping[round] - n)) * pricer.opt.payoff(samples[int(stopping[round])][round], stopping[round], pricer.mod.r)
        return values, f
    

def compute_lower_bound(pricer, trained_models):
    """ Uses the predicted mean of all simulated payoffs at corresponding stopping times to compute the lower bound."""
    samples = pricer.simulate_samples(pricer.nb_samples)
    values = predict_values(pricer, samples, trained_models)[0]
    lower_bound = np.mean(values, axis=1)
    return lower_bound[0]



def doob_increments(noise, doob):
    """Computes the incrementation for the doob matrix."""
    for i in range(len(doob)):
        for j in range(len(doob[0])):
            if i == 0:
                doob[i, j] = noise[i, j]
            else:
                doob[i, j] = doob[i - 1, j] + noise[i, j]

def get_proper_continuation(z_n, m, round, J):
    """Extracts the current conditionally simulated trajectory"""
    z_n_m = []
    for j in range(J):
        z_n_m.append(z_n[round][j][m])
    return torch.from_numpy(np.array(z_n_m)).float()




def compute_upper_bound(pricer, trained_models, Ku, J):
    """Computes the upper bound for the expected payoff using the Doob-Meyer decomposition."""
    z = pricer.simulate_samples(Ku)
    f = np.array([np.zeros((pricer.opt.dates + 1, J)) for _ in range(Ku)])
    f_theta = np.zeros((pricer.opt.dates + 1, Ku)) 
    stopping = np.array([np.zeros((pricer.opt.dates + 1, J)) for _ in range(Ku)])
    for round in range(Ku):
        f[round][pricer.opt.dates, :] = 1
        stopping[round][pricer.opt.dates, :] = pricer.opt.dates
    f_theta[pricer.opt.dates, :] = 1
    noisy_estimates = np.zeros((pricer.opt.dates + 1, Ku))
    doob = np.zeros((pricer.opt.dates + 1, Ku))
    z_N = pricer.simulate_conditionally(pricer.opt.dates, z, J)
    current_C = torch.from_numpy(np.array([np.mean(np.array([pricer.opt.payoff(z_N[round][j][-1], pricer.opt.dates, pricer.mod.r) for j in range(J)])) for round in range(Ku)]))
    
    for n in range(pricer.opt.dates - 1, -1, -1):
        z_n = pricer.simulate_conditionally(n, z, J)
        #fitted_z_n = torch.from_numpy(np.stack([z_n[round] for round in range(len(z_n))], axis=1))
        network = trained_models[n]
        F = network(z[n])
        adjusted_F = F.detach().numpy().reshape(Ku)
        f_theta = np.zeros((pricer.opt.dates + 1, Ku))
        f_theta[n, :] = (adjusted_F > 0.5).astype(float)
        for m in range(n, pricer.opt.dates):
            network = trained_models[m]
            for round in range(Ku):
                z_n_m = get_proper_continuation(z_n, m - n, round, J)
                F = network(z_n_m)
                adjusted_F = F.detach().numpy().reshape(J)
                f[round][m, :] = (adjusted_F > 0.5).astype(float)
                stopping[round][m, :] = np.array([max(range(len(f[round][:,col])), key=lambda row: f[round][row,col]) for col in range(f[round].shape[1])])
        previous_C = current_C
        current_C = torch.from_numpy(np.array([np.mean(np.array([pricer.opt.payoff(z_n[round][j][int(stopping[round][n + 1, j]) - n], stopping[round][n + 1, j], pricer.mod.r) for j in range(J)])) for round in range(Ku)]))
        for round in range(Ku):
            noisy_estimates[n + 1, round] = f_theta[n + 1, round] * pricer.opt.payoff(z[n + 1][round], n + 1, pricer.mod.r)  + (1 - f_theta[n + 1, round]) * previous_C[round] - current_C[round]
    
    doob_increments(noisy_estimates, doob)
    upper = np.array([[pricer.opt.payoff(z[n][round], n, pricer.mod.r) - doob[n, round] for round in range(Ku)] for n in range(pricer.opt.dates + 1)])
    maxed_upper = np.amax(upper, axis=0)
    return np.mean(maxed_upper)




