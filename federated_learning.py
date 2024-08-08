import numpy as np

class FederatedLearning:
    def __init__(self, num_clients=100, feature_dim=4, rff_dim=200, 
                 num_participating_clients=40, learning_rate=0.75, 
                 num_iterations=1000, num_shared_params=40, coordinated=False):
        self.num_clients = num_clients
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
        self.num_participating_clients = num_participating_clients
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_shared_params = num_shared_params
        self.share_fraction = num_shared_params / rff_dim
        self.coordinated = coordinated
        self.global_weights = np.random.randn(rff_dim)
        self.local_weights = [self.global_weights.copy() for _ in range(num_clients)]
        self.selection_matrices = self.initialize_selection_matrices()
        self.x, self.y, self.z, self.W, self.b = self.initialize_data()

    def initialize_data(self):
        x = np.zeros((self.num_clients, self.num_iterations, self.feature_dim))
        y = np.zeros((self.num_clients, self.num_iterations, 1))
        z = np.zeros((self.num_clients, self.num_iterations, self.rff_dim))
        W = np.random.randn(self.num_clients, self.feature_dim, self.rff_dim)
        b = np.random.uniform(0, 2 * np.pi, (self.num_clients, 1, self.rff_dim))
        return x, y, z, W, b

    def initialize_selection_matrices(self):
        selection_matrices = []
        for _ in range(self.num_clients):
            selection_matrix = np.zeros(self.rff_dim)
            selected_indices = np.random.choice(self.rff_dim, self.num_shared_params, replace=False)
            selection_matrix[selected_indices] = 1
            selection_matrix = np.diag(selection_matrix)
            selection_matrices.append(selection_matrix)
        return selection_matrices

    def generate_synthetic_data(self):
        for k in range(self.num_clients):
            theta_k = np.random.uniform(0.2, 0.9)
            mu_k = np.random.uniform(-0.2, 0.2)
            sigma2_uk = np.random.uniform(0.2, 1.2)
            sigma2_nuk = np.random.uniform(0.005, 0.03)
            uk = np.random.normal(mu_k, np.sqrt(sigma2_uk), (self.num_iterations, self.feature_dim))
            nuk = np.random.normal(0, np.sqrt(sigma2_nuk), (self.num_iterations, 1))
            
            self.x[k,0] = uk[0]
            for n in range(1, self.num_iterations):
                self.x[k,n,:] = theta_k * self.x[k,n-1,:] + np.sqrt(1 - theta_k**2) * uk[n]
                self.y[k,n,:] = (np.sqrt(self.x[k, n, 0]**2 + np.sin(np.pi * self.x[k, n, 3])**2) + 
                                 (0.8 - 0.5*np.exp(-self.x[k, n, 1]**2)*self.x[k, n, 2])) + nuk[n]

            self.z[k,:,:] = np.sqrt(2 / self.rff_dim) * np.cos(np.dot(self.x[k,:,:], self.W[k,:,:]) + self.b[k,:,:])

    def update_local_weights(self, selected_indices, n):
        for k in range(self.num_clients):
            if k in selected_indices:
                local_weights_prime = np.dot(self.selection_matrices[k], self.local_weights[k]) + \
                                      np.dot((np.eye(self.rff_dim)-self.selection_matrices[k]), self.local_weights[k])
                epsilon = self.y[k,n,:] - np.dot(local_weights_prime, self.z[k,n,:])
                self.local_weights[k] = self.local_weights[k] + self.learning_rate * self.z[k,n,:] * epsilon
            else:
                epsilon = self.y[k,n,:] - np.dot(self.local_weights[k], self.z[k,n,:])
                self.local_weights[k] = self.local_weights[k] + self.learning_rate * self.z[k,n,:] * epsilon

    def update_global_weights(self, selected_indices):
        for k in selected_indices:
            self.global_weights = np.dot(self.selection_matrices[k], self.local_weights[k]) + \
                                  np.dot((np.eye(self.rff_dim)-self.selection_matrices[k]), self.global_weights)

    def roll_selection_matrices(self):
        for k in range(self.num_clients):
            self.selection_matrices[k] = np.roll(self.selection_matrices[k], 1)

    def compute_mse(self):
        mse_values_per_iteration = np.zeros(self.num_iterations)
        mse_values_per_iteration_per_client = np.zeros((self.num_clients, self.num_iterations))
        
        for n in range(self.num_iterations):
            selected_indices = np.random.choice(self.num_clients, self.num_participating_clients, replace=False)
            self.update_local_weights(selected_indices, n)
            
            for k in range(self.num_clients):
                epsilon = self.y[k,n,:] - np.dot(self.local_weights[k], self.z[k,n,:])
                mse_values_per_iteration_per_client[k,n] = epsilon**2
                mse_values_per_iteration[n] += mse_values_per_iteration_per_client[k,n]
            mse_values_per_iteration[n] /= self.num_clients     

            self.roll_selection_matrices()
            self.update_global_weights(selected_indices)
            self.global_weights /= self.num_participating_clients
        
        return mse_values_per_iteration

    def run_experiment(self, num_trials=500):
        mse_values_all_trials = np.zeros(self.num_iterations)
        for _ in range(num_trials):
            self.generate_synthetic_data()
            mse_values_per_iteration = self.compute_mse()
            mse_values_all_trials += mse_values_per_iteration

        mse_values_all_trials /= num_trials
        mse_values_all_trials /= max(mse_values_all_trials)
        return 10 * np.log10(mse_values_all_trials)
