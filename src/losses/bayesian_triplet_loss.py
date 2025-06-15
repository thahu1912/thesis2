import torch
import torch.nn as nn
from ..models.layers import functional as LF

class BayesianTripletLoss(nn.Module):
    def __init__(self, margin, varPrior, kl_scale_factor=1e-6, distribution='gauss'):
        super(BayesianTripletLoss, self).__init__()
        
        self.margin = margin
        self.varPrior = varPrior
        self.kl_scale_factor = kl_scale_factor
        self.distribution = distribution

    def forward(self, x, label):
        # divide x into anchor, positive, negative based on labels
        D, N = x.shape
        nq = torch.sum(label.data == -1).item()  # number of tuples
        S = x.size(1) // nq  # number of images per tuple including query: 1+1+n
        A = x[:, label.data == -1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        P = x[:, label.data == 1].permute(1, 0).repeat(1, S - 2).view((S - 2) * nq, D).permute(1, 0)
        N = x[:, label.data == 0]

        varA = A[-1:, :]
        varP = P[-1:, :]
        varN = N[-1:, :]

        muA = A[:-1, :]
        muP = P[:-1, :]
        muN = N[:-1, :]

        # calculate nll
        nll = LF.negative_loglikelihood(muA, muP, muN, varA, varP, varN, margin=self.margin)

        kl = torch.tensor(0.0, device=x.device)

        # KL(anchor|| prior) + KL(positive|| prior) + KL(negative|| prior)
        if self.distribution == 'gauss':
            muPrior = torch.zeros_like(muA, requires_grad=False)
            varPrior = torch.ones_like(varA, requires_grad=False) * self.varPrior

            kl = (LF.kl_div_gauss(muA, varA, muPrior, varPrior) + \
                  LF.kl_div_gauss(muP, varP, muPrior, varPrior) + \
                  LF.kl_div_gauss(muN, varN, muPrior, varPrior))

        elif self.distribution == 'vMF':
            kl = (LF.kl_div_vMF(muA, varA) + \
                  LF.kl_div_vMF(muP, varP) + \
                  LF.kl_div_vMF(muN, varN))

        return nll + self.kl_scale_factor * kl

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'margin=' + '{:.4f}'.format(self.margin) + ')' 