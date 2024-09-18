import numpy as np

class DatasetGenerator:
    def __init__(self, num_clients, num_iterations, feature_dim, rff_dim):
        self.num_clients = num_clients
        self.num_iterations = num_iterations
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim

    def generate_dataset(self):
        W = np.random.randn(self.num_clients, self.feature_dim, self.rff_dim)
        b = np.random.uniform(0, 2 * np.pi, (self.num_clients, 1, self.rff_dim))

        x = np.zeros((self.num_clients, self.num_iterations, self.feature_dim))
        z = np.zeros((self.num_clients, self.num_iterations, self.rff_dim))
        y = np.zeros((self.num_clients, self.num_iterations, 1))

        v_k = np.random.laplace(0, 1, (self.num_iterations, self.feature_dim))
        nuk = np.random.normal(0, 1, (self.num_iterations, 1))

        gamma_k = np.random.uniform(0.4, 0.7)
        delta_k = np.random.uniform(0.2, 0.6)
        lambda_k = 0.5

        x[:,0] = v_k[0]
        for k in range(1, self.num_clients):
            for n in range(2, self.num_iterations):
                x[k,n,:] = gamma_k * x[k,n-2,:] + delta_k * np.tanh(x[k,n-1,:]) + lambda_k * v_k[n]

        for k in range(1, self.num_clients):
            z[k,:,:] = np.sqrt(2 / self.rff_dim) * np.cos(np.dot(x[k,:,:], W[k,:,:]) + b[k,:,:])
            for n in range(1, self.num_iterations):
                y[k,n,:] = np.log(1 + np.abs(x[k, n, 0]) * np.cos(np.pi * x[k, n, 1])) + \
                           (1 + np.exp(-x[k, n, 3]**2)) * x[k, n, 2] + nuk[n]

        return [(z[k, -1, :].reshape(-1, 1), y[k, -1, :].reshape(-1, 1)) for k in range(self.num_clients)]
