from federated_learning import FederatedLearning
from plotting import plot_mse

if __name__ == "__main__":
    fl = FederatedLearning()
    mse_values = fl.run_experiment()
    plot_mse(mse_values)
