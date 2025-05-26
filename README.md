# PCA-Clustering
This Python script implements dimensionality reduction and unsupervised clustering from scratch using NumPy, SciPy, and scikit-learn for synthetic data generation. 

# Features
- Implements PCA via both Eigen Decomposition and Singular Value Decomposition (SVD).

- Visualizes explained variance and scree plots to assist in dimensionality selection.

- Allows projection of high-dimensional data onto a reduced subspace.

- Custom implementation of K-Means Clustering with support for:

  - Euclidean

  - Manhattan (City Block)

  - Mahalanobis distances.

- Silhouette score calculation for cluster evaluation.

- Adjusted Rand Index (ARI) for clustering quality based on ground-truth labels.

- Visualization of clustering results in 2D.

- Full EM algorithm implementation for Mixture of Gaussians:

  - E-Step: Computes posterior probabilities.

  - M-Step: Updates means, covariances, and mixing coefficients.

  - Supports random or K-Means-based initialization.

- Log-likelihood convergence monitoring.
