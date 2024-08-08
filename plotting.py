import matplotlib.pyplot as plt

def plot_mse(mse_values):
    plt.plot(mse_values)
    plt.xlabel("Iterations")
    plt.ylabel("MSE (dB)")
    plt.title("Mean Squared Error Over Iterations")
    plt.show()
