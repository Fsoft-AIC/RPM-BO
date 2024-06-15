import torch
import gpytorch
from gpytorch.constraints import GreaterThan



class OrthogonalProjectionNNGaussianKernel(gpytorch.kernels.Kernel):
    def __init__(self, dim, hidden_units, beta_min, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the ambient high-dimensional space
        :param hidden_units: dimension of the hidden layer
        :param beta_min: minimum value of the inverse square lengthscale parameter beta
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(OrthogonalProjectionNNGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.beta_min = beta_min
        self.dim = dim
        self.hidden_units = hidden_units

        # Add beta parameter, corresponding to the inverse of the lengthscale parameter.
        beta_num_dims = 1
        self.register_parameter(name="raw_beta", parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1,
                                                                                          beta_num_dims)))

        if beta_prior is not None:
            self.register_prior("beta_prior", beta_prior, lambda module: self.beta, lambda module, v: self._set_beta(v))

        # A GreaterThan constraint is defined on the lengthscale parameter to guarantee the positive-definiteness of the
        #  kernel.
        # The value of beta_min can be determined e.g. experimentally.
        self.register_constraint("raw_beta", GreaterThan(self.beta_min))


        # Add a first matrix weight and first  bias
        b1 = torch.rand(1, self.hidden_units).repeat(*self.batch_shape,1,1)
        self.register_parameter(name = "raw_W1", parameter=torch.nn.Parameter(torch.rand(*self.batch_shape, self.hidden_units,
                                                                                          self.dim)))
        self.register_parameter(name = 'raw_b1', parameter=torch.nn.Parameter(b1))


        # Add a second matrix weight and first  bias
        b2 = torch.rand(1, self.dim).repeat(*self.batch_shape,1,1)
        self.register_parameter(name = "raw_W2", parameter=torch.nn.Parameter(torch.rand(*self.batch_shape, self.dim,
                                                                                          self.hidden_units)))
        self.register_parameter(name = 'raw_b2', parameter=torch.nn.Parameter(b2))


    @property
    def beta(self):
        return self.raw_beta_constraint.transform(self.raw_beta)

    @beta.setter
    def beta(self, value):
        self._set_beta(value)

    def _set_beta(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_beta)
        self.initialize(raw_beta=self.raw_beta_constraint.inverse_transform(value))

    @property
    def W1(self):
        return self.raw_W1

    @W1.setter
    def W1(self, value):
        self._set_W1(value)

    def _set_W1(self, value):
        self.initialize(raw_W1=value)


    @property
    def W2(self):
        return self.raw_W2

    @W2.setter
    def W2(self, value):
        self._set_W2(value)

    def _set_W2(self, value):
        self.initialize(raw_W2=value)

    @property
    def b1(self):
        return self.raw_b1

    @b1.setter
    def b1(self, value):
        self._set_b1(value)

    def _set_b1(self, value):
        self.initialize(raw_b1=value)


    @property
    def b2(self):
        return self.raw_b2

    @b2.setter
    def b2(self, value):
        self._set_b2(value)

    def _set_b2(self, value):
        self.initialize(raw_b2=value)


    def forward(self, x1, x2, diagonal_distance=False, **params):
            px1 = torch.relu(self.W1 @ x1.T + self.b1.T)
            px2 = torch.relu(self.W1 @ x2.T + self.b1.T)

            px1 = self.W2 @ px1 + self.b2.T
            px2 = self.W2 @ px2 + self.b2.T

            px1 = px1.T / torch.max(torch.abs(px1.T), dim = 1)[0].reshape(-1,1)
            px2 = px2.T / torch.max(torch.abs(px2.T), dim = 1)[0].reshape(-1,1)

            # Compute distance
            distance = torch.cdist(px1, px2, p = 2) #TODO: check the dimension of the output
            distance2 = torch.mul(distance, distance)

            exp_component = torch.exp(- distance2.mul(self.beta.double()))

            return exp_component



