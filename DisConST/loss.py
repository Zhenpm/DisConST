import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

        
class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-5), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        result = torch.mean(result)
        result = nan2inf(result)
        return result

def nan2zero(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x), x)

def nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x)+float('inf'), x)

def nelem(x):
    nelem = torch.sum(~torch.isnan(x)).float()
    return torch.where(nelem == 0., torch.tensor(1.), nelem).to(x.device)

def reduce_mean(x):
    nelem = nelem(x)
    x = nan2zero(x)
    return torch.sum(x) / nelem

def mse_loss(y_true, y_pred):
    ret = torch.square(y_pred - y_true)
    return reduce_mean(ret)

def poisson_loss(y_true, y_pred):
    y_pred = y_pred.float()
    y_true = y_true.float()

    nelem = nelem(y_true)
    y_true = nan2zero(y_true)

    ret = y_pred - y_true*torch.log(y_pred+1e-10) + torch.lgamma(y_true+1.0)
    return torch.sum(ret) / nelem

class NB:
    def __init__(self, theta=None, masking=False, scale_factor=1.0, debug=False):
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.debug = debug
        self.masking = masking
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        
        y_true = y_true
        y_pred = y_pred * scale_factor
        #y_pred = nan2zero(y_pred)
        y_pred = torch.clamp(y_pred,min=0)
        if self.masking:
            nelem = nelem(y_true)
            y_true = nan2zero(y_true)

        theta = torch.minimum(self.theta, torch.tensor(1e6))

        t1 = torch.lgamma(theta+eps) + torch.lgamma(y_true+1.0) - torch.lgamma(y_true+theta+eps)
        t2 = (theta+y_true) * torch.log(1.0 + (y_pred/(theta+eps))) + (y_true * (torch.log(theta+eps) - torch.log(y_pred+eps)))
        if self.debug:
            assert_ops = [
                torch.all(torch.isfinite(y_pred), 'y_pred has inf/nans'),
                torch.all(torch.isfinite(t1), 't1 has inf/nans'),
                torch.all(torch.isfinite(t2), 't2 has inf/nans')
            ]

            # Add tensorboard logging if you are using it
            # tensorboard.add_histogram('t1', t1, global_step)
            # tensorboard.add_histogram('t2', t2, global_step)

            with torch.no_grad():
                for assert_op in assert_ops:
                    assert_op()

            final = t1 + t2

        else:
            final = t1 + t2
        final = nan2inf(final)

        if mean:
            if self.masking:
                final = torch.sum(final) / nelem
            else:
                final = reduce_mean(final)
        return final

class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        scale_factor = scale_factor[:, None]
        eps = self.eps

        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0-self.pi+eps)
        y_true = y_true
        y_pred = y_pred * scale_factor

        y_pred = torch.clamp(y_pred,min=0)

        theta = torch.minimum(self.theta, torch.tensor(1e6))
        zero_nb = torch.pow(theta/(theta+y_pred+eps), theta)
        zero_case = -torch.log(self.pi + ((1.0-self.pi)*zero_nb)+eps)
        result = torch.where(y_true < 1e-5, zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge

        if mean:
            if self.masking:
                result = reduce_mean(result)
                print('mask')
            else:
                result = torch.mean(result)

        result = nan2inf(result)
        '''
        if self.debug:
            # Add tensorboard logging if you are using it
            # tensorboard.add_histogram('nb_case', nb_case, global_step)
            # tensorboard.add_histogram('zero_nb', zero_nb, global_step)
            # tensorboard.add_histogram('zero_case', zero_case, global_step)
            # tensorboard.add_histogram('ridge', ridge, global_step)
            pass
        '''
        return result