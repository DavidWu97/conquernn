'''
The collection of example simulated functions used in the paper.
'''
import numpy as np
from scipy.stats import t as t_dist, laplace

# # Sample some iid U(0,1) covariates
# X = np.random.random(size=(nsamples, dim))


class Scenario1():
    def __init__(self):
        super().__init__()
        self.beta = np.array([1, 0])
        self.n_in = 2

    def noiseless(self, X):
        return np.cos(2*np.pi*X[:,0]**2) + np.sin(np.sqrt(X[:,0]**2+2*X[:,1])+2)

    def quantile(self, X, q):
        return self.noiseless(X) + self.g1(X) * t_dist.ppf(q, 2)

    def sample(self, X):
        return self.noiseless(X) + self.g1(X) * np.random.standard_t(2, size=X.shape[0])
    
    def g1(self,X):
        return 0.5 * np.linalg.norm(X - self.beta, axis=1)


# class Scenario2():
#     def __init__(self):
#         super().__init__()
#         self.n_in = 2

#     def noiseless(self, X):
#         return self.g2(self.g1(X))

#     def quantile(self, X, q):
#         return self.noiseless(X) + laplace.ppf(q, scale=2)

#     def sample(self, X):
#         return self.noiseless(X) + laplace.rvs(scale=2, size=X.shape[0])
    
#     def g1(self,X):
#         condition = X[:,0] < 0.5
#         result = np.where(condition[:, np.newaxis],
#                           np.column_stack([X[:,0] + np.sqrt(X[:,1]), np.sqrt(2*X[:,0] + X[:,1])]),
#                           np.column_stack([2*X[:,0]**2, np.abs(X[:,0] - X[:,1])]))
#         return result
    
#     def g2(self,X):
#         return 2 * np.sqrt(X[:,0]+2*X[:,1]+1)
    
class Scenario2():
    def __init__(self):
        super().__init__()
        self.beta = np.array([0.5, 0, 0.5, 0, 0.5])
        self.n_in = 5

    def noiseless(self, X):
        return np.sqrt(X[:,0]+2*X[:,1]+X[:,2]+2*X[:,3]+X[:,4])

    def quantile(self, X, q):
        return self.noiseless(X) + np.sqrt(X.dot(self.beta)) * t_dist.ppf(q, 3)

    def sample(self, X):
        return self.noiseless(X) + np.sqrt(X.dot(self.beta)) * np.random.standard_t(3, size=X.shape[0])
    

class Scenario3():
    def __init__(self):
        super().__init__()
        self.n_in = 5

    def noiseless(self, X):
        return self.g2(self.g1(X))

    def quantile(self, X, q):
        return self.noiseless(X) + laplace.ppf(q, scale=2)

    def sample(self, X):
        return self.noiseless(X) + laplace.rvs(scale=2, size=X.shape[0])
    
    def g1(self,X):
        result = np.array([X[:,0]+3*X[:,1],np.cos(2*np.pi*(X[:,2]+X[:,3])),X[:,1]+np.sqrt(X[:,2])+2*X[:,4]]).T
        return result
    
    def g2(self,X):
        condition = X[:,1] < 0
        return np.where(condition,X[:,0]+np.sqrt(X[:,1]**2+X[:,2]),np.sqrt(X[:,0]+X[:,1])+0.5*X[:,2])










