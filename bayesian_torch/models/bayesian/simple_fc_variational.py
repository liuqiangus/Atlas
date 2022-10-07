from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import LinearReparameterization

prior_mu = 0.0
prior_sigma = 1.0
posterior_mu_init = 0.0
posterior_rho_init = -3.0


class SFC(nn.Module):
    def __init__(self, input_dim=14, output_dim=1, activation=F.relu):
        super(SFC, self).__init__()

        self.activation = activation

        self.fc1 = LinearReparameterization(
            in_features=input_dim,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc2 = LinearReparameterization(
            in_features=128,
            out_features=256,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc3 = LinearReparameterization(
            in_features=256,
            out_features=256,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc4 = LinearReparameterization(
            in_features=256,
            out_features=256,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc5 = LinearReparameterization(
            in_features=256,
            out_features=128,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

        self.fc6 = LinearReparameterization(
            in_features=128,
            out_features=output_dim,
            prior_mean=prior_mu,
            prior_variance=prior_sigma,
            posterior_mu_init=posterior_mu_init,
            posterior_rho_init=posterior_rho_init,
        )

    def forward(self, x):
        kl_sum = 0
        x, kl = self.fc1(x)
        kl_sum += kl
        x = F.relu(x)
        x, kl = self.fc2(x)
        kl_sum += kl
        x = F.relu(x)
        # x, kl = self.fc3(x)
        # kl_sum += kl
        # x = F.relu(x)
        # x, kl = self.fc4(x)
        # kl_sum += kl
        # x = F.relu(x)
        x, kl = self.fc5(x)
        kl_sum += kl
        x = F.relu(x)        
        x, kl = self.fc6(x)
        kl_sum += kl
        if self.activation is None:
            output = x
        else:
            output = self.activation(x) # attention, this only regress non-negative values TODO XXX
        return torch.squeeze(output), kl_sum 
