from experiment_runner import ExperimentRunner

def main():
    num_clients = 100
    num_iterations = 1000
    feature_dim = 5
    rff_dim = 200
    learning_rate = 0.55
    num_epochs = 10
    num_shared_params = 175
    num_experiments = 1
    num_participating_clients = 20

    experiment_runner = ExperimentRunner(
        num_clients=num_clients,
        num_iterations=num_iterations,
        feature_dim=feature_dim,
        rff_dim=rff_dim,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        num_shared_params=num_shared_params,
        num_experiments=num_experiments,
        num_participating_clients=num_participating_clients
    )

    average_global_mse_history = experiment_runner.run()
    
    # Save the result
    import numpy as np
    np.save('pso_175.npy', average_global_mse_history)

if __name__ == "__main__":
    main()
