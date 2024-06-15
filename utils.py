from __future__ import annotations
from numbers import Number
import gpytorch
import torch
from torch.distributions.exp_family import ExponentialFamily
from botorch.models import SingleTaskGP
from botorch import fit_gpytorch_model
from fit_model import fit_gpytorch_model_torch
from BoManifolds.manifold_optimization.manifold_gp_fit import fit_gpytorch_manifold
from torch.distributions.utils import broadcast_all
from torch.nn import Module as TModule
from torch.distributions import constraints

from gpytorch.priors import Prior
from gpytorch.priors.utils import _bufferize_attributes

import pymanopt.optimizers as pyman_solvers
import numpy as np


def _standard_invgamma(concentration):
    return 1/torch._standard_gamma(concentration)


class InverseGamma(ExponentialFamily):
    r"""
    Creates a Gamma distribution parameterized by shape :attr:`concentration` and :attr:`rate`.

    Example::

        >>> m = Gamma(torch.tensor([1.0]), torch.tensor([1.0]))
        >>> m.sample()  # Gamma distributed with concentration=1 and rate=1
        tensor([ 0.1046])

    Args:
        concentration (float or Tensor): shape parameter of the distribution
            (often referred to as alpha)
        rate (float or Tensor): rate = 1 / scale of the distribution
            (often referred to as beta)
    """
    arg_constraints = {'concentration': constraints.positive, 'rate': constraints.positive}
    support = constraints.nonnegative
    has_rsample = True
    _mean_carrier_measure = 0

    @property
    def mean(self):
        return self.rate / (self.concentration - 1)
        #return self.concentration / self.rate

    @property
    def mode(self):
        return self.rate / (self.concentration + 1)
        #return ((self.concentration - 1) / self.rate).clamp(min=0)

    @property
    def variance(self):
        return self.rate.pow(2) / ((self.concentration-1).pow(2) * (self.concentration - 2))
        #return self.concentration / self.rate.pow(2)

    def __init__(self, concentration, rate, validate_args=None):
        self.concentration, self.rate = broadcast_all(concentration, rate)
        if isinstance(concentration, Number) and isinstance(rate, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.concentration.size()
        super(InverseGamma, self).__init__(batch_shape, validate_args=validate_args)

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        batch_shape = torch.Size(batch_shape)
        new.concentration = self.concentration.expand(batch_shape)
        new.rate = self.rate.expand(batch_shape)
        super(InverseGamma, new).__init__(batch_shape, validate_args=False)
        new._validate_args = self._validate_args
        return new

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        value = _standard_invgamma(self.concentration.expand(shape)) / self.rate.expand(shape)
        value.detach().clamp_(min=torch.finfo(value.dtype).tiny)  # do not record in autograd graph
        return value

    def log_prob(self, value):
        value = torch.as_tensor(value, dtype=self.rate.dtype, device=self.rate.device)
        if self._validate_args:
            self._validate_sample(value)
        return (torch.xlogy(self.concentration, self.rate) + torch.xlogy(-self.concentration-1,value) - self.rate/value - torch.lgamma(self.concentration))
        #return (torch.xlogy(self.concentration, self.rate) +
        #        torch.xlogy(self.concentration - 1, value) -
        #        self.rate * value - torch.lgamma(self.concentration))

    def entropy(self):
        return (self.concentration + torch.log(self.rate) + torch.lgamma(self.concentration) - (1+self.concentration)*torch.digamma(self.concentration))
        #return (self.concentration - torch.log(self.rate) + torch.lgamma(self.concentration) +
        #        (1.0 - self.concentration) * torch.digamma(self.concentration))

    @property
    def _natural_params(self):
        return (self.concentration - 1, -self.rate)

    def _log_normalizer(self, x, y):
        return torch.lgamma(x + 1) + (x + 1) * torch.log(-y.reciprocal())


class InvGammaPrior(Prior, InverseGamma):
    """Gamma Prior parameterized by concentration and rate

    1/pdf(x) = beta^alpha / Gamma(alpha) * x^(alpha - 1) * exp(-beta * x)

    were alpha > 0 and beta > 0 are the concentration and rate parameters, respectively.
    """

    def __init__(self, concentration, rate, validate_args=False, transform=None):
        TModule.__init__(self)
        InverseGamma.__init__(self, concentration=concentration, rate=rate, validate_args=validate_args)
        _bufferize_attributes(self, ("concentration", "rate"))
        self._transform = transform


    def expand(self, batch_shape):
        batch_shape = torch.Size(batch_shape)
        return InvGammaPrior(self.concentration.expand(batch_shape), self.rate.expand(batch_shape))

    def __call__(self, *args, **kwargs):
        return super(InverseGamma, self).__call__(*args, **kwargs)

    '''
    Sample in a low-dimensional linear embedding, to initialize ALEBO.
    Generates points on a linear subspace of [-1, 1]^D by generating points in
    [-b, b]^D, projecting them down with a matrix B, and then projecting them
    back up with the pseudoinverse of B. Thus points thus all lie in a linear
    subspace defined by B. Points whose up-projection falls outside of [-1, 1]^D
    are thrown out, via rejection sampling.
    To generate n points, we start with nsamp points in [-b, b]^D, which are
    mapped down to the embedding and back up as described above. If >=n points
    fall within [-1, 1]^D after being mapped up, then the first n are returned.
    If there are less than n points in [-1, 1]^D, then b is constricted
    (halved) and the process is repeated until there are at least n points in
    [-1, 1]^D. There exists a b small enough that all points will project to
    [-1, 1]^D, so this is guaranteed to terminate, typically after few rounds.
    Args:
        B: A (dxD) projection down.
        nsamp: Number of samples to use for rejection sampling.
        init_bound: b for the initial sampling space described above.
        seed: seed for UniformGenerator
'''

def get_fitted_model(train_x, train_obj, covar_module, likelihood, update_param, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj, covar_module=covar_module, likelihood=likelihood)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    # Define the marginal log-likelihood
    if update_param:
        mll_fct = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)

        # Define a solver on the manifold
        projection_solver = pyman_solvers.ConjugateGradient(max_iterations=200)

        mll_fct.to(train_x)
        fit_gpytorch_model(mll=mll_fct, optimizer=fit_gpytorch_manifold, solver = projection_solver, nb_init_candidates = 20)

    return model

def get_fitted_model_torch(train_x, train_obj, covar_module, likelihood, update_param, state_dict=None):
    # initialize and fit model
    model = SingleTaskGP(train_X=train_x, train_Y=train_obj, covar_module=covar_module, likelihood=likelihood)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    # Define the marginal log-likelihood
    if update_param:
        mll_fct = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        mll_fct.to(train_x)
        fit_gpytorch_model_torch(mll=mll_fct, options = {"disp": False, "maxiter":1000, "lr":0.1})

    return model


def re_order(x):
    a = x
    for i in range(1,len(x)):
        if a[i-1] < a[i]:
            a[i] = a[i-1]
    return a

def create_file_name(test_func, low_dim, high_dim, initial_n, start_rep, stop_rep, seed):
    return str(test_func) + '_results_CS_d' + str(low_dim) + '_D' + str(high_dim) + '_n' + str(
                        initial_n) + '_rep_' + str(start_rep) + '_' + str(stop_rep) + '_seed' + str(seed) 
    
def transform_dataframe(dataframe):
    df_grouped = (
    dataframe[['index', 'Value']].groupby(['index'])
    .agg(['mean', 'std', 'count'])
)
    df_grouped = df_grouped.droplevel(axis=1, level=0).reset_index()
# Calculate a confidence interval as well.
    df_grouped['ci'] = 1.03 * df_grouped['std'] / np.sqrt(df_grouped['count'])
    df_grouped['ci_lower'] = df_grouped['mean'] - df_grouped['ci']
    df_grouped['ci_upper'] = df_grouped['mean'] + df_grouped['ci']
    return df_grouped

def ensure_not_1D(x):
    """
    Ensure x is not 1D (i.e. make size (D,) data into size (1,D))
    :param x: torch.Tensor
    :return:
    """
    if x.ndim == 1:
        if isinstance(x, np.ndarray):
            x = np.expand_dims(x, axis=0)
        elif isinstance(x, torch.Tensor):
            x = x.unsqueeze(0)
    return x


