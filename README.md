# PSO-Fed: Partial Sharing Online Federated Learning

This repository implements a communication-efficient online federated learning framework for nonlinear regression using partial sharing. Clients update local models with streaming data and share only portions of those updates with the server, reducing communication overhead while maintaining performance.

This implementation is inspired by the work of Vinay Chakravarthi Gogineni et al., titled "Communication-Efficient Online Federated Learning Framework for Nonlinear Regression," presented at ICASSP 2022. [Read the paper](https://doi.org/10.1109/ICASSP43922.2022.9746228).

## Project Structure

The project is divided into several scripts:

1. **federated_learning.py**
   - Contains the `FederatedLearning` class, which implements the PSO-Fed algorithm.

2. **data_generation.py**
   - Contains functions for generating synthetic data used in the experiments.

3. **plotting.py**
   - Contains functions for plotting the results.

4. **main.py**
   - The main script that runs the federated learning experiment and plots the results.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/PSO-Fed.git
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

- Ensure all dependencies are installed correctly by running pip install -r requirements.txt.
  
- Make sure you are using a compatible version of Python (e.g., Python 3.6 or higher).
 
- If you encounter issues related to missing files or incorrect paths, verify that you are in the correct directory (PSO-Fed).

If problems persist, feel free to open an issue on GitHub.

## Contributing

Contributions are welcome! If you have suggestions for improvements or bug fixes, please follow these steps:

1. Fork the repository.

2. Create a new branch (git checkout -b feature-branch).

3. Make your changes and commit them (git commit -m 'Add some feature').

4. Push to the branch (git push origin feature-branch).

5. Open a pull request.

Please ensure your code follows the existing style and includes appropriate tests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.txt) file for details.
