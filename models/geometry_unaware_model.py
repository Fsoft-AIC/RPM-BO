import torch
import gpytorch
import numpy as np
from sklearn.utils import check_random_state
from scipy.stats import ortho_group
from torch.autograd import Variable
from utils import InvGammaPrior, get_fitted_model_torch
from kernels.orthogonal_projection_kernel_nn import OrthogonalProjectionNNGaussianKernel
import scipy
from botorch.models import SingleTaskGP
from gpytorch import settings as gpt_settings




class GuA_NN_model():
    """
    Maximize a black-box objective function.
    """
    def __init__(self, d_orig, d_embedding,  hidden_units, device,
                 initial_random_samples=10,
                 dtype = torch.double,
                 box_size = 1,
                 gamma = 1,
                 p = 5,
                 q = 100,
                 random_embedding_seed=0,
                 initial_points_list = None):
        """
        Parameters
        ----------
        d_orig (int): Number of dimension for original space. 
        d_embedding (int): Number of dimensions for the lower dimensional subspace
        hidden_units (int): Number of dimensions for the hidden_units for Neural Network projection
        box_size (float): The boundary of the search space
        gamma (float): The weighting factor to balance the supervised loss and the unsupervised cosistency loss
        p (int): Number of random lambda in Equation (7)
        q (int): Number of random point in Equation (7)
        initial_points_list (np.array((num_init, d_orig))): List of initial points
        initial_random_samples (int): If initial_points_list is None, random initial_random_samples initial points
        """
        if initial_points_list is None:
            self.initial_random_samples = initial_random_samples
        else:
            self.initial_random_samples = int(initial_points_list.shape[0])
        self.rng = check_random_state(random_embedding_seed)
        self.initial_points_list = initial_points_list
        self.box_size = box_size
        self.gamma = gamma
        self.dtype = dtype
        self.device=device
        self.d_embedding = d_embedding   # Dimension of the embedded space
        self.d_orig = d_orig   # Dimensions of the original space
        self.hidden_units = hidden_units  # size of hidden units for neural network
        self.p = p   # number of point on line segment
        self.q = q   # number of unlabeled data
        self.x_unlabeled = torch.rand(self.q, self.d_orig).to(dtype=self.dtype, device=self.device) * 2 - self.box_size

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
        self.k_fct = gpytorch.kernels.ScaleKernel(OrthogonalProjectionNNGaussianKernel(dim=self.d_orig, hidden_units=self.hidden_units,
                                        beta_min=0.21, beta_prior=gpytorch.priors.GammaPrior(2.0, 0.15)))
        self.k_fct.to(dtype=self.dtype, device=self.device)

         # Define the weight matrix and bias 
        self.W1 = Variable(self.k_fct.base_kernel.W1.data.clone(), requires_grad=False)
        self.b1 = Variable(self.k_fct.base_kernel.b1.data.clone(), requires_grad=False)
        self.W2 = Variable(self.k_fct.base_kernel.W2.data.clone(), requires_grad=False)
        self.b2 = Variable(self.k_fct.base_kernel.b2.data.clone(), requires_grad=False)

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

    def projection_to_manifold(self, x):
        X = x.to(dtype=self.dtype,device=self.device)
        X_query = torch.relu(X @ self.W1.T + self.b1)
        X_query = X_query @ self.W2.T + self.b2
        X_return = X_query / torch.max(torch.abs(X_query), dim = 1)[0].reshape(-1,1)
        return X_return.reshape(-1,self.d_orig)

    def proposed_EI(self, z):
        Z_torch = torch.tensor(z, requires_grad=True).to(dtype=self.dtype,device=self.device)
        X = Z_torch @ self.A
        Z_proj = self.projection_to_manifold(X) @ self.A.T
        GP_pred = self.latent_model(Z_proj)
        mean = GP_pred.mean
        std = torch.sqrt(GP_pred.variance)
        posterior = torch.distributions.normal.Normal(mean, std)
        q = (mean - torch.max(self.y)) / std
        ei = (mean -  torch.max(self.y)) * posterior.cdf(q) + std * 10**posterior.log_prob(q)
        return -ei.data.cpu().numpy()
 
    def select_query_point(self, iteration=100):
        """
        iteration (int): 
            Number of iteration for maximizer maximize acquistion function
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
                X_query_embedded = self.projection_to_manifold(X_query) @ self.A.T
                return X_query, X_query_embedded

        # Query by maximizing the acquisition function
        else:
            self.latent_model = get_fitted_model_torch(train_x=self.X_embedded, train_obj=self.y,
                                                    covar_module = self.latent_k_fct, likelihood = self.latent_lik_fct, update_param = True)
            
            bounds = [(-np.sqrt(self.box_size * self.d_embedding),np.sqrt(self.box_size * self.d_embedding))]*self.d_embedding
            z = np.random.uniform(-np.sqrt(self.box_size * self.d_embedding),np.sqrt(self.box_size * self.d_embedding),(1,self.d_embedding))
            res = scipy.optimize.minimize(self.proposed_EI, z, method = "L-BFGS-B", bounds = bounds, options = {"maxiter":iteration})
            X_query_embedded = torch.tensor(res.x).reshape(-1,self.d_embedding).to(dtype=self.dtype, device=self.device)
            X_query = self.projection_to_manifold(X_query_embedded @ self.A)
            
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
        self.model = self.get_fitted_model_semi(self.X.clone().detach(), self.y, covar_module = self.k_fct, likelihood = self.lik_fct, update_param = update_param)
        self.W1 = Variable(self.k_fct.base_kernel.W1.data.clone(), requires_grad=False)
        self.b1 = Variable(self.k_fct.base_kernel.b1.data.clone(), requires_grad=False)
        self.W2 = Variable(self.k_fct.base_kernel.W2.data.clone(), requires_grad=False)
        self.b2 = Variable(self.k_fct.base_kernel.b2.data.clone(), requires_grad=False)
        self.X_embedded = self.projection_to_manifold(self.X) @ self.A.T

    def get_fitted_model_semi(self, train_x, train_y, covar_module, likelihood, update_param, state_dict=None):
    # initialize and fit model
        model = SingleTaskGP(train_X=train_x, train_Y=train_y, covar_module=covar_module, likelihood=likelihood)
        
        if state_dict is not None:
            model.load_state_dict(state_dict)
        # Define the marginal log-likelihood
        if update_param:
            training_iter = 200
            model.train()
            model.likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            mll.to(train_x)
            train_inputs, train_targets = mll.model.train_inputs, mll.model.train_targets
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                with gpt_settings.fast_computations(log_prob=True):
                    output = mll.model(*train_inputs)
                    loss = -mll(output, train_targets)
                    self.W1 = mll.model.covar_module.base_kernel.W1
                    self.b1 = mll.model.covar_module.base_kernel.b1
                    self.W2 = mll.model.covar_module.base_kernel.W2
                    self.b2 = mll.model.covar_module.base_kernel.b2
                    X_induce_ = self.projection_to_manifold(self.x_unlabeled)
                    for j in range(self.p-1):
                        M = self.projection_to_manifold(((j+1)/self.p * self.x_unlabeled + (1 - (j+1)/self.p) * X_induce_)) - X_induce_
                        loss += self.gamma * torch.linalg.norm(M, ord = 2, axis = 1).sum() / (self.q * self.p)
                    loss.backward()
                    optimizer.step()
            return mll.model
        else:
            return model

        