import numpy as np

def generate_synthetic_data(num_clients, num_iterations, feature_dim, rff_dim):
    x = np.zeros((num_clients, num_iterations, feature_dim))
    y = np.zeros((num_clients, num_iterations, 1))
    z = np.zeros((num_clients, num_iterations, rff_dim))
    W = np.random.randn(num_clients, feature_dim, rff_dim)
    b = np.random.uniform(0, 2 * np.pi, (num_clients, 1, rff_dim))
    
    for k in range(num_clients):
        theta_k = np.random.uniform(0.2, 0.9)
        mu_k = np.random.uniform(-0.2, 0.2)
        sigma2_uk = np.random.uniform(0.2, 1.2)
        sigma2_nuk = np.random.uniform(0.005, 0.03)
        uk = np.random.normal(mu_k, np.sqrt(sigma2_uk), (num_iterations, feature_dim))
        nuk = np.random.normal(0, np.sqrt(sigma2_nuk), (num_iterations, 1))
        
        x[k,0] = uk[0]
        for n in range(1, num_iterations):
            x[k,n,:] = theta_k * x[k,n-1,:] + np.sqrt(1 - theta_k**2) * uk[n]
            y[k,n,:] = (np.sqrt(x[k, n, 0]**2 + np.sin(np.pi * x [k, n, 3])**2) + 
                        (0.8 - 0.5*np.exp(-x[k, n, 1]**2)*x[k, n, 2])) + nuk[n]

        z[k,:,:] = np.sqrt(2 / rff_dim) * np.cos(np.dot(x[k,:,:], W[k,:,:]) + b[k,:,:])
    
    return x, y, z, W, b
