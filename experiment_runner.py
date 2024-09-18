import numpy as np
from dataset_generator import DatasetGenerator
from federated_learning import FederatedLearning

class ExperimentRunner:
    def __init__(self, num_clients, num_iterations, feature_dim, rff_dim, learning_rate, num_epochs, num_shared_params, num_experiments, num_participating_clients):
        self.num_clients = num_clients
        self.num_iterations = num_iterations
        self.feature_dim = feature_dim
        self.rff_dim = rff_dim
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.num_shared_params = num_shared_params
        self.num_experiments = num_experiments
        self.num_participating_clients = num_participating_clients

    def run(self):
        all_global_mse_histories = []
        for experiment in range(self.num_experiments):
            print(f"Running Experiment {experiment + 1}/{self.num_experiments}")

            dataset_gen = DatasetGenerator(self.num_clients, self.num_iterations, self.feature_dim, self.rff_dim)
            client_data = dataset_gen.generate_dataset()

            federated_model = FederatedLearning(self.num_clients, self.rff_dim, self.learning_rate, self.num_epochs, self.num_shared_params)

            global_mse_history = []
            for round in range(self.num_iterations):
                selected_clients = np.random.choice(range(self.num_clients), size=self.num_participating_clients, replace=False)
                client_models = federated_model.client_update(client_data, selected_clients)
                federated_model.federated_averaging(client_models)

                global_mse = federated_model.evaluate_global_model(client_data)
                global_mse_history.append(global_mse)

            all_global_mse_histories.append(global_mse_history)

        return np.mean(all_global_mse_histories, axis=0)
