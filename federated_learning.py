import numpy as np

class FederatedLearning:
    def __init__(self, num_clients, rff_dim, learning_rate, num_epochs, num_shared_params):
        self.num_clients = num_clients
        self.rff_dim = rff_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_shared_params = num_shared_params
        self.global_fc = np.random.randn(rff_dim, 1)
        self.local_fc = [self.global_fc for _ in range(num_clients)]
        self.selection_matrices = []

    def client_update(self, client_data, selected_clients):
        selection_matrix = np.zeros(self.rff_dim)
        selected_indices = np.random.choice(self.rff_dim, self.num_shared_params, replace=False)
        selection_matrix[selected_indices] = 1
        selection_matrix = np.diag(selection_matrix)
        self.selection_matrices = [selection_matrix for _ in range(self.num_clients)]

        client_models = []
        for client_index in selected_clients:
            inputs, labels = client_data[client_index]
            inputs = inputs.reshape(1, -1)
            fc = np.dot(self.selection_matrices[client_index], self.global_fc.copy()) + \
                 np.dot((np.eye(self.rff_dim) - self.selection_matrices[client_index]), self.local_fc[client_index])
            
            for _ in range(self.num_epochs):
                outputs = np.dot(inputs, fc)
                grad = 2 * np.dot(inputs.T, outputs - labels)
                fc -= self.learning_rate * grad
            
            self.local_fc[client_index] = fc
            client_models.append(fc)

        return client_models

    def federated_averaging(self, client_models):
        aggregated_fc = np.zeros_like(self.global_fc)
        for client_fc in client_models:
            aggregated_fc += client_fc
        self.global_fc = aggregated_fc / len(client_models)

    def evaluate_global_model(self, client_data):
        global_mse = 0
        for inputs, labels in client_data:
            inputs = inputs.reshape(1, -1)
            outputs = np.dot(inputs, self.global_fc)
            global_mse += np.mean((outputs - labels) ** 2)
        return global_mse / len(client_data)

    def print_final_global_model(self):
        print("Final global model weights:")
        print(self.global_fc)
