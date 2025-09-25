import torch
import numpy as np
from scipy.stats import norm

class QuantileLoss:
    def __init__(self, quantiles, bandwidth=0.05):
        self.quantiles = quantiles
        self.bandwidth = bandwidth
        self.h_tensor = torch.tensor(bandwidth)
        
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))
        self.sqrt_2pi = torch.sqrt(torch.tensor(2.0 * torch.pi))
        self.sqrt_2_over_pi = torch.sqrt(torch.tensor(2.0 / torch.pi))

    def origin(self, yhat, y, requires_grad=True):
        z = y - yhat
        q = self.quantiles
        out1 = torch.max(q[None]*z, (q[None]-1)*z)
        out2 = q[None]*(z >= 0) + (q[None]-1)*(z < 0)
        if requires_grad:
            return out1.mean(), out2
        return out1.mean()

    def ols(self, yhat, y, requires_grad=True):
        z = y - yhat
        out1 = torch.square(z) 
        out2 = 2*z
        if requires_grad:
            return out1.mean(), out2
        return out1.mean()

    def gaussian(self, yhat, y, requires_grad=True):
        z = y - yhat
        q = self.quantiles
        h = self.h_tensor
        
        normalized_z = z / h
        pnorm_val = 0.5 * (1 + torch.erf(normalized_z / self.sqrt_2.to(z.device)))
        exp_term = torch.exp(-0.5 * normalized_z**2)
        
        out1 = 0.5 * h * (self.sqrt_2_over_pi.to(z.device) * exp_term + normalized_z * (2 * pnorm_val - 1)) + (q[None] - 0.5) * z
        out2 = q[None] - 1 + pnorm_val
        
        if requires_grad:
            return out1.mean(), out2
        return out1.mean()

    def logistic(self, yhat, y, requires_grad=True):
        z = y - yhat
        q = self.quantiles
        h = self.h_tensor

        normalized_z = z / h
        tmp = torch.exp(-normalized_z)
        log_term = torch.log1p(tmp)

        out1 = 0.5 * h * (normalized_z + 2 * log_term) + (q[None] - 0.5) * z
        out2 = q[None] + (1 - tmp) / (1 + tmp)
        
        if requires_grad:
            return out1.mean(), out2
        return out1.mean()

    def uniform(self, yhat, y, requires_grad=True):
        z = y - yhat
        q = self.quantiles
        h = self.h_tensor
        
        normalized_z = z / h
        abs_normalized_z = torch.abs(normalized_z)
        sign_normalized_z = torch.sign(normalized_z)
        
        out1_term = torch.where(
            abs_normalized_z <= 1,
            0.5 * normalized_z**2 + 0.5,
            abs_normalized_z
        )
        out1 = 0.5 * h * out1_term + (q[None] - 0.5) * z
        
        out2_term = torch.where(
            abs_normalized_z <= 1,
            normalized_z,
            sign_normalized_z
        )
        out2 = q[None] - 0.5 + 0.5 * out2_term
        
        if requires_grad:
            return out1.mean(), out2
        return out1.mean()

    def epanechnikov(self, yhat, y, requires_grad=True):
        z = y - yhat
        q = self.quantiles
        h = self.h_tensor
        
        normalized_z = z / h
        abs_normalized_z = torch.abs(normalized_z)
        sign_normalized_z = torch.sign(normalized_z)
        
        out1_term = torch.where(
            abs_normalized_z <= 1,
            0.75 * normalized_z**2 - 0.125 * normalized_z**4 + 0.375,
            abs_normalized_z
        )
        out1 = 0.5 * h * out1_term + (q[None] - 0.5) * z
        
        out2_term = torch.where(
            abs_normalized_z <= 1,
            1.5 * normalized_z - 0.5 * normalized_z**3,
            sign_normalized_z
        )
        out2 = q[None] - 0.5 + 0.5 * out2_term
        
        if requires_grad:
            return out1.mean(), out2
        return out1.mean()

    def triangular(self, yhat, y, requires_grad=True):
        z = y - yhat
        q = self.quantiles
        h = self.h_tensor
        
        normalized_z = z / h
        abs_normalized_z = torch.abs(normalized_z)
        sign_normalized_z = torch.sign(normalized_z)
        
        out1_term = torch.where(
            abs_normalized_z <= 1,
            normalized_z**2 - abs_normalized_z**3/3 + 0.375,
            abs_normalized_z
        )
        out1 = 0.5 * h * out1_term + (q[None] - 0.5) * z
        
        out2_term = torch.where(
            abs_normalized_z <= 1,
            (2 * normalized_z - sign_normalized_z * normalized_z**2),
            sign_normalized_z
        )
        out2 = q[None] - 0.5 + 0.5 * out2_term
        
        if requires_grad:
            return out1.mean(), out2
        return out1.mean()