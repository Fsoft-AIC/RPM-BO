import torch
import gpytorch
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import ortho_group
from torch.autograd import Variable
from utils import get_fitted_model, InvGammaPrior, get_fitted_model_torch
from kernels.orthogonal_projection_kernel_sphere import OrthogonalProjectionSphereGaussianKernel
from pyDOE import lhs
import scipy




class GA_sphere_model():
    """
    Maximize a black-box objective function.
    """

    def __init__(self, d_orig, d_embedding, d_embedding_sphere, device,
                 initial_random_samples=10,
                 dtype = torch.double,
                 box_size = 1,
                 random_embedding_seed=0,
                 initial_points_list = None):
        """
        Parameters
        ----------
        original_boundaries ((D, 2) np.array): Boundaries of the original search
            space (of dimension D). This is used for rescaling. The first column
            is the minimum value for the corresponding dimension/row, and the
            second column is the maximum value.
        n_keep_dims (int): Number of dimensions in the original space that are
            preserved in the embedding. This is used if certain dimensions are
            expected to be independently relevant. Assume that these dimensions
            are the first parameters of the input space.
        d_embedding: int
            Number of dimensions for the lower dimensional subspace
        box_size (int): the boundary of the search space
        """
        if initial_points_list is None:
            self.initial_random_samples = initial_random_samples
        else:
            self.initial_random_samples = int(initial_points_list.shape[0])
        self.rng = check_random_state(random_embedding_seed)
        self.initial_points_list = initial_points_list
        self.box_size = box_size
        self.dtype = dtype
        self.device=device
        # Dimension of the embedded space
        self.d_embedding = d_embedding

        # Dimension of the d_true_embedding
        self.d_embedding_sphere = d_embedding_sphere

        # Dimensions of the original space
        self.d_orig = d_orig

        # Draw an orthogonal matrix with the shape is (d_embedding,d_orig) A.A^T  = I
        if self.d_orig < 2000:
            m = ortho_group.rvs(dim = self.d_orig)  # Draw a random (d_orig,d_orig) orthogonal matrix
            self.A = torch.tensor(m[:self.d_embedding,:]).to(dtype=self.dtype, device=self.device)
        else:
            m = np.random.random(size=(self.d_orig, self.d_embedding))
            q, _ = np.linalg.qr(m)
            self.A = torch.tensor(q.T).to(dtype=self.dtype, device=self.device)

        # Produces (d_embedding, 2) array
        self.embedding_boundaries = torch.tensor(
            [[-np.sqrt(self.box_size * self.d_embedding),
                np.sqrt(self.box_size * self.d_embedding)]] * self.d_embedding).to(dtype=self.dtype, device=self.device)

        self.best_value = -1000000  # The process return the maximum point
        self.best_x = np.array([])

        self.X = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of data
        self.X_embedded = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of embedded data
        self.y = torch.tensor([], dtype = self.dtype, device=self.device)  # running list of function evaluations

        self.model = None
        self.latent_model = None

        # Create the covariance for the original GP model
        boundaries = torch.stack([-self.box_size * torch.ones(d_orig, dtype=dtype, device = device), self.box_size * torch.ones(d_orig, dtype=dtype, device=device)]).T
        self.k_fct = gpytorch.kernels.ScaleKernel(OrthogonalProjectionSphereGaussianKernel(dim=self.d_orig, latent_dim=self.d_embedding_sphere,
                                        boundaries = boundaries, beta_min=0.21, beta_prior=gpytorch.priors.GammaPrior(2.0, 0.15)))
        self.k_fct.to(dtype=self.dtype, device=self.device)

        # Define the projection matrix which forms a basis of the latent space
        self.projection_matrix = Variable(self.k_fct.base_kernel.projection_matrix.data.clone(), requires_grad=False)

        # Define the centroid of the effective subspace
        self.centroid = Variable(self.k_fct.base_kernel.centroid.data.clone(), requires_grad = False)

        # Define the radius of the effective subspace
        self.radius = Variable(self.k_fct.base_kernel.radius.data.clone(), requires_grad = False)
        # Create the covariance for the projected GP model
        self.latent_k_fct = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lengthscale_prior=InvGammaPrior(2.0,0.15)))
        self.latent_k_fct.to(dtype=self.dtype, device=self.device, non_blocking=False) ## Cast to type of x_data

    
        # Define the likelihood function of the original GP model
        self.noise_prior = gpytorch.priors.torch_priors.GammaPrior(1.1, 0.05)
        self.noise_prior_mode = (self.noise_prior.concentration - 1) / self.noise_prior.rate
        self.lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=self.noise_prior,
                                                                          noise_constraint=
                                                                          gpytorch.constraints.GreaterThan(1e-8),
                                                                          initial_value=self.noise_prior_mode)
        self.lik_fct.to(dtype=self.dtype,device=self.device)

        # Define the likelihood function of the projected GP model
        self.latent_lik_fct = gpytorch.likelihoods.gaussian_likelihood.GaussianLikelihood(noise_prior=self.noise_prior,
                                                                                 noise_constraint=
                                                                                 gpytorch.constraints.GreaterThan(1e-8),
                                                                                 initial_value=self.noise_prior_mode)
        self.latent_lik_fct.to(dtype=self.dtype, device=self.device)

    def projection_to_sphere(self, x):
        X = x.to(dtype=self.dtype,device=self.device)
        X_query = (X @ self.projection_matrix - self.centroid)@self.projection_matrix.T
        X_query = X_query / torch.norm(X_query, dim = 1).reshape(-1,1)
        X_query = torch.clamp(self.radius * X_query + self.centroid @ self.projection_matrix.T, -self.box_size, self.box_size)
        return X_query.reshape(-1,self.d_orig)

    def proposed_EI(self, z):
        Z_torch = torch.tensor(z, requires_grad=True).to(dtype=self.dtype,device=self.device)
        X = Z_torch @ self.A
        Z_proj = self.projection_to_sphere(X) @ self.A.T
        GP_pred = self.latent_model(Z_proj)
        mean = GP_pred.mean
        std = torch.sqrt(GP_pred.variance)
        posterior = torch.distributions.normal.Normal(mean, std)
        q = (mean - torch.max(self.y)) / std
        ei = (mean -  torch.max(self.y)) * posterior.cdf(q) + std * 10**posterior.log_prob(q)
        return -ei.data.cpu().numpy()
 

    def select_query_point(self, iteration=100):
        """

        :param
            iteration (int): if iteration == 1, initialize the latent model, else take the parameter of original model
            batch_size (int): number of query points to return
        :return:
            (batch_size x d_orig) numpy array
        """
        # TODO: Make the random initialization its own function so it can be done separately from the acquisition argmin
        # Initialize with random points
        if len(self.X) < self.initial_random_samples:
            if self.initial_points_list is None:
            # Select query point randomly from embedding_boundaries
                X_query_embedded = \
                    self.rng.uniform(size=self.embedding_boundaries.shape[0]) \
                    * (self.embedding_boundaries[:, 1] - self.embedding_boundaries[:, 0]) \
                    + self.embedding_boundaries[:, 0]
                X_query_embedded = torch.from_numpy(X_query_embedded).unsqueeze(0)
                print("X_query_embedded.shape: {}".format(X_query_embedded.shape))
            else:
                #self.X = torch.cat((self.X, torch.Tensor(self.initial_points_list[len(self.X)]))).double()
                X_query = torch.tensor(self.initial_points_list[len(self.X)], dtype=self.dtype, device=self.device)
                X_query_embedded = self.projection_to_sphere(X_query) @ self.A.T
                return X_query, X_query_embedded

        # Query by maximizing the acquisition function
        else:
            self.latent_model = get_fitted_model_torch(train_x=self.X_embedded, train_obj=self.y,
                                                    covar_module = self.latent_k_fct, likelihood = self.latent_lik_fct, update_param = True)
            
            # Z_cand = lhs(self.d_embedding, 2000) * 2 * np.sqrt(self.d_embedding) - np.sqrt(self.d_embedding)
            # ei_cand = self.proposed_EI(Z_cand)
            # Z_optimize = Z_cand[ei_cand.argmin()]
            # X_query_embedded = torch.tensor(Z_optimize).reshape(-1,self.d_embedding).to(dtype=self.dtype, device=self.device)

            bounds = [(-np.sqrt(self.box_size * self.d_embedding),np.sqrt(self.box_size * self.d_embedding))]*self.d_embedding
            z = np.random.uniform(-np.sqrt(self.box_size * self.d_embedding),np.sqrt(self.box_size * self.d_embedding),(1,self.d_embedding))
            res = scipy.optimize.minimize(self.proposed_EI, z, method = "L-BFGS-B", bounds = bounds, options = {"maxiter":iteration})
            X_query_embedded = torch.tensor(res.x).reshape(-1,self.d_embedding).to(dtype=self.dtype, device=self.device)
            
            X_query = self.projection_to_sphere(X_query_embedded @ self.A)
    
            return torch.clamp(X_query,-self.box_size, self.box_size), X_query_embedded

    def update(self, X_query, y_query, update_param):
        """ Update internal model for observed (X, y) from true function.
        Args:
            X_query ((1,d_orig) np.array):
                Point in original input space to query
            y_query (float):
                Value of black-box function evaluated at X_query
            update_param  (bool):
                Check if update the parameter of model
        """
        # add new rows of data
        self.X = torch.cat((self.X, X_query))
        self.y = torch.cat([self.y, y_query.reshape(-1,1)], axis=0)

        # Update the extreme points
        if y_query >= self.best_value:
            self.best_value = y_query
            self.best_x = X_query

        # torch.tensor(self.X, dtype=torch.float64)
        self.model = get_fitted_model(self.X.clone().detach(), self.y.double(), covar_module = self.k_fct, likelihood = self.lik_fct, update_param = update_param)
        self.projection_matrix = Variable(self.k_fct.base_kernel.projection_matrix.data.clone(), requires_grad=False)
        self.centroid = Variable(self.k_fct.base_kernel.centroid.data.clone(), requires_grad=False)
        self.radius = Variable(self.k_fct.base_kernel.radius.data.clone(), requires_grad= False)
        self.X_embedded = self.projection_to_sphere(self.X) @ self.A.T


    