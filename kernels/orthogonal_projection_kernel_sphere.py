import torch
import gpytorch
from gpytorch.constraints import GreaterThan, Interval

import pymanopt.manifolds as pyman_man

class OrthogonalProjectionSphereGaussianKernel(gpytorch.kernels.Kernel):
    def __init__(self, dim, latent_dim, boundaries, beta_min, beta_prior=None, **kwargs):
        """
        Initialisation.

        Parameters
        ----------
        :param dim: dimension of the ambient high-dimensional space
        :param latent_dim: dimension of the embedded sphere
        :param boundaries: boundary of original search space
        :param beta_min: minimum value of the inverse square lengthscale parameter beta
        :param beta_prior: prior on the parameter beta
        :param kwargs: additional arguments
        """
        super(OrthogonalProjectionSphereGaussianKernel, self).__init__(has_lengthscale=False, **kwargs)
        self.beta_min = beta_min
        self.dim = dim
        self.latent_dim = latent_dim
        self.boundaries = boundaries

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

        # Add projection matrix parameters stands for the basis of subspace
        self.raw_projection_matrix_manifold = pyman_man.Stiefel(self.dim, self.latent_dim)
        self.register_parameter(name="raw_projection_matrix",
                                parameter=torch.nn.Parameter(torch.Tensor(self.raw_projection_matrix_manifold.random_point()).
                                                             repeat(*self.batch_shape, 1, 1)))

        # Add a center parameter of the effective sphere
        centroid = torch.rand(1,self.latent_dim).repeat(*self.batch_shape,1,1)
        self.register_parameter(name="raw_centroid",
                                parameter=torch.nn.Parameter(centroid))
       
        # Add a radius of the sphere
        radius = torch.rand(1,1).repeat(*self.batch_shape,1,1)
        self.register_parameter(name="raw_radius",
                                parameter=torch.nn.Parameter(radius))
        self.register_constraint("raw_radius", Interval(lower_bound=10e-2, upper_bound=torch.sqrt(((self.boundaries[:,0][:self.latent_dim] - self.boundaries[:,1][:self.latent_dim])**2).sum())/2))
       
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
    def projection_matrix(self):
        return self.raw_projection_matrix

    @projection_matrix.setter
    def projection_matrix(self, value):
        self._set_projection_matrix(value)

    def _set_projection_matrix(self, value):
        self.initialize(raw_projection_matrix=value)


    @property
    def centroid(self):
        return self.raw_centroid
        #return self.raw_centroid_constraint.transform(self.raw_centroid)

    @centroid.setter
    def centroid(self, value):
        self._set_centroid(value)

    def _set_centroid(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_centroid)
        #self.initialize(raw_beta=value)
        self.initialize(raw_beta=self.raw_centroid_constraint.inverse_transform(value))


    @property
    def radius(self):
        return self.raw_radius_constraint.transform(self.raw_radius)

    @radius.setter
    def radius(self, value):
        self._set_radius(value)
    
    def _set_radius(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_radius)
        self.initialize(raw_beta=self.raw_radius_constraint.inverse_transform(value))


    def forward(self, x1, x2, diagonal_distance=False, **params):
            px1 = self.projection_matrix @ (self.projection_matrix.T @ x1.T - self.centroid.T)
            px2 = self.projection_matrix @ (self.projection_matrix.T @ x2.T - self.centroid.T)

            px1 = torch.clamp(self.radius * px1 / torch.norm(px1) + self.projection_matrix @ self.centroid.T, -1,1)
            px2 = torch.clamp(self.radius * px2 / torch.norm(px2) + self.projection_matrix @ self.centroid.T, -1,1)
            
            # Compute distance
            distance = torch.cdist(px1.T, px2.T, p = 2) #TODO: check the dimension of the output
            distance2 = torch.mul(distance, distance)

            exp_component = torch.exp(- distance2.mul(self.beta.double()))

            return exp_component



