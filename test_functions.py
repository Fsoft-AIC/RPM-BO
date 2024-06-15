import torch
import numpy as np
import warnings
import LassoBench
from pathlib import Path
import os
warnings.filterwarnings("ignore")

root_dir = str(Path(os.getcwd()).parent.absolute())



class TestFunction:
    def evaluate(self,x):
        pass

def l2cost(x, point):
        return 10 * np.linalg.norm(x - point, 1)


class ackley_proposed_sphere(TestFunction):
    def __init__(self, low_dim, device, radius=1):
        self.low_dim = low_dim
        self.radius = radius
        self.device = device
    def evaluate(self, x):
        z = x[0][:self.low_dim+1]
        z = self.radius * z/torch.linalg.norm(z,ord=2)
        n = len(z)
        S = 0
        M = 0
        for i in range(n):
            S += z[i]**2
        S = -0.2*torch.sqrt(S/n)
        for i in range(n):
            M += torch.cos(2*torch.pi*z[i])
        M = torch.exp(M/n)
        return -20 * torch.exp(S) - M + torch.exp(torch.tensor(1,device=self.device)) + 20 + torch.normal(0,0.1,(1,),device=self.device)

class ackley_proposed_mix(TestFunction):
    def __init__(self, low_dim, device):
        self.low_dim = low_dim
        self.device = device
    def evaluate(self, x):
        z = x[0][:20]
        for i in range(0,10,2):
            norm_z = torch.linalg.norm(z[i:i+2], ord = 2)
            z[i:i+2] = z[i:i+2]/norm_z
        n = len(z)
        S = 0
        M = 0
        for i in range(n):
            S += z[i]**2
        S = -0.2*torch.sqrt(S/n)
        for i in range(n):
            M += torch.cos(2*torch.pi*z[i])
        M = torch.exp(M/n)
        return -20 * torch.exp(S) - M + torch.exp(torch.tensor(1,device=self.device)) + 20 + torch.rand(1, device=self.device)*0.01

class hyper_proposed_sphere(TestFunction):
    def __init__(self, low_dim, device, radius=1):
        self.low_dim = low_dim
        self.radius = radius
        self.device = device
    def evaluate(self, x):
        z = x[0][:self.low_dim+1]
        z = self.radius * z/torch.linalg.norm(z,ord=2)
        n = len(z)
        S = 0
        for i in range(n):
            for j in range(i):
                S += z[j]**2
        return S

class hyper_proposed_mix(TestFunction):
    def __init__(self, low_dim, device):
        self.low_dim = low_dim
        self.device = device
    def evaluate(self, x):
        z = x[0][:20]
        for i in range(0,10,2):
            norm_z = torch.linalg.norm(z[i:i+2], ord = 2)
            z[i:i+2] = z[i:i+2]/norm_z
        n = len(z)
        S = 0
        for i in range(n):
            for j in range(i):
                S += z[j]**2
        return S
    
class lasso_hard(TestFunction):
    def __init__(self, device):
        self.device = device
    
    def evaluate(self,x):
        # dimension of x is an even number
        synt_bench = LassoBench.SyntheticBenchmark(pick_bench="synt_hard", noise=True)
        z = x[0].cpu().numpy()
        loss = synt_bench.evaluate(z)
        return torch.tensor(loss).to(device=self.device)

class lasso_real_proposed(TestFunction):
    def __init__(self,  device):
        self.device = device
    

    def evaluate(self,x):
        # dimension of x is an even number
        synt_bench = LassoBench.RealBenchmark(pick_data="leukemia")
        #d = synt_bench.n_features
        z = x[0].cpu().numpy()
        loss = synt_bench.evaluate(z)
        #f = Mopta.MoptaSoftConstraints()
        #z = (x[0].cpu().numpy() + 1)/2
        return torch.tensor(loss).to(device=self.device)

    