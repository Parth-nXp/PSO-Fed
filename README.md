# PSO-Fed: Partial Sharing Online Federated Learning

This repository implements a communication-efficient online federated learning framework for nonlinear regression using partial sharing. Clients update local models with streaming data and share only portions of those updates with the server, reducing communication overhead while maintaining performance.

This implementation is inspired by the work of Vinay Chakravarthi Gogineni et al., titled "Communication-Efficient Online Federated Learning Framework for Nonlinear Regression," presented at International Conference on Acoustics, Speech, and Signal Processing 2022. [Read the paper](https://doi.org/10.1109/ICASSP43922.2022.9746228).

## Project Structure

The project is divided into four main scripts:

### 1. `dataset_generator.py`
   - **Purpose**: Contains the `DatasetGenerator` class, responsible for generating synthetic data for federated learning. It uses random Fourier features (RFF) to map each client’s input-output data and generates corresponding labels.
   - **Key Functionality**:
       - `generate_dataset()`: Produces input-output pairs for each client using random Fourier features and an autoregressive model to generate inputs.

### 2. `federated_learning.py`
   - **Purpose**: Contains the `FederatedLearning` class, which performs local updates for each client and applies federated averaging to update the global model.
   - **Key Functionality**:
      - `client_update()`: Updates the local models by computing gradients on each client's data and adjusting the model weights accordingly.
      - `federated_averaging()`: Aggregates the local models from selected clients to compute the global model using federated averaging.
      - `evaluate_global_model()`: Evaluates the performance of the global model across clients by calculating mean squared error (MSE).
      - `print_final_global_model()`: Displays the final global model weights at the end of training.

### 3. `experiment_runner.py`
   - **Purpose**: Contains the `ExperimentRunner` class, which manages the execution of federated learning experiments by integrating dataset generation and local model updates.
   - **Key Functionality**:
      - `run()`: Executes multiple experiments, generates training and testing datasets, conducts federated learning rounds, and tracks global loss histories across experiments.

### 4. `main.py`
   - **Purpose**: The entry point of the project, responsible for initializing and running the federated learning experiments using the `ExperimentRunner` class.
   - **Key Functionality**:
      - `main()`: Sets up the parameters for the experiment, calls the `ExperimentRunner`, and runs the entire training and evaluation process.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Partial-Sharing-Online-Federated-Learning.git
    cd PSO-Fed
    ```

2. Create a virtual environment and activate it:
    ```bash
    python -m venv pso-fed-env
    source pso-fed-env/bin/activate  # On Windows use `pso-fed-env\Scripts\activate`
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main script to start the experiment:
```bash
python main.py
```

## Troubleshooting

If you encounter any issues or errors while running the project, please check the following:

- Ensure all dependencies are installed correctly by running `pip install -r requirements.txt`.
  
- Make sure you are using a compatible version of Python (e.g., Python 3.6 or higher).
 
- If you encounter issues related to missing files or incorrect paths, verify that you are in the correct directory (`Partial-Sharing-Online-Federated-Learning`).

If problems persist, feel free to open an issue on GitHub.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository.

2. Create a new branch (`git checkout -b feature-branch`).

3. Make your changes and commit them (`git commit -m 'Add some feature'`).

4. Push to the branch (`git push origin feature-branch`).

5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
