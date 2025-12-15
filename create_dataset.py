## Creating the dataset of normalized potentials (that still satisfy the same boundary conditions) that the model will learn from.
import numpy as np
from scipy.linalg import eigh_tridiagonal as eigh_tri
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh

L = 1
N = 1000*L
M = N-1
x= np.linspace(0,L,N+1)
x2 = np.linspace(0.001,0.999,M)
dx = L/N
hbar = 1
m = 1
def generate_potential(M):
    V=np.zeros(M)
    if np.random.rand() < 0.73 :
        a = np.random.uniform(1,10)
        b = np.random.uniform(0.2, 0.8)
        sigma = np.random.uniform(0.03, 0.3)
        V += a*np.exp(-(x2-b)**2/(2*sigma**2))

    if np.random.rand() < 0.45:
        n = np.random.randint(1,5)
        V += np.random.uniform(1,5) * np.sin(n*np.pi*np.linspace(0,1,N-1))

    V += np.random.rand(M)
    return V
## Combining all the constants into a singular constant
a = hbar**2/(2*m*dx**2)

## Solving the Hamiltonian Matrix


diagonal = np.full(N-1, 2*a)
off_diagonal = np.full(N-2, -1*a)
def Hamilton_solver(V) :
    H_sparse = diags([off_diagonal,diagonal + V,off_diagonal], offsets=[-1,0,1])
    k=5
    eigenvalues, eigenvectors = eigsh(H_sparse,k=k,which='SM')
    return eigenvalues, eigenvectors
## Normalizing and standardizing the eigenvectors
k=5

def eigenvector_normalizing(eigenvectors) :
    fixed_eigenvectors = np.zeros((N+1,k))
    for i in range (k) :
        v = np.zeros(N+1)
        v[1:-1] = eigenvectors[:,i]
        v /= np.sqrt(np.sum(np.abs(v)**2)*dx)
        for j in range (M) :
            val = v[j]
            if np.abs(val) >1e-10 :
                if val < 0 :
                    v *= -1
                    break
                if val > 0:
                    break
        fixed_eigenvectors[:,i] = v
    return fixed_eigenvectors
def dataset_creation(num_tests=10, M = M, k = k, save_path = 'dataset3.npz') :
    X = np.zeros((num_tests, M), dtype=np.float32)
    Y_E = np.zeros((num_tests, k), dtype=np.float32)
    Y_PSI = np.zeros((num_tests, k, N+1), dtype=np.float32)
    dx = 1.0/N
    for i in range(num_tests) :
        V = generate_potential(M)
        eigval, eigvec = Hamilton_solver(V)
        eigvec = eigenvector_normalizing(eigvec)
        X[i] = V
        Y_E[i] = eigval[:k]
        Y_PSI[i] = eigvec.T
        if (i+1) % 50 == 0 :
            print ('50 Tests have been created')
    np.savez(save_path, X=X, Y_E=Y_E, Y_PSI=Y_PSI)
    print("Dataset saved.")
dataset_creation()
