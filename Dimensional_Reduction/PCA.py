import numpy as np
import matplotlib.pyplot as plt

#generate some example 2D data 
np.random.seed(42)
mean=[0,0]
cov=[[3,1],[1,2]]#covarience matrix
X=np.random.multivariate_normal(mean,cov,200)

#standardize the data
X_meaned=X-np.mean(X,axis=0)

#convarience matrix
cov_matrix=np.cov(X_meaned,rowvar=False)

#eigen decomposition
eigenvalues,eigenvectors=np.linalg.eigh(cov_matrix)

#sort eigenvalue and eigenvectors
sorted_idx=np.argsort(eigenvalues)[::-1]
eigenvalues=eigenvalues[sorted_idx]
eigenvectors=eigenvectors[:,sorted_idx]

#project data onto the top k eigenvectors
k=1
eigenvector_subset=eigenvectors[:, :k]
X_reduced=np.dot(X_meaned,eigenvector_subset)

print("Reduced Data Shape:",X_reduced.shape)

