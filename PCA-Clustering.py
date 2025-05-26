from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal

##################################### Conventional PCA ######################################################################
def conventionalPCA(x,dims,method):
    """
    Perform Principal Component Analysis (PCA) using either Eigen Decomposition or Singular Value Decomposition (SVD).

    Parameters:
    x (numpy.ndarray): Unscaled input data matrix of shape (n_samples, n_features).
    dims (int): Number of principal components to retain.
    method (str): Dimensionality reduction method, either "eig" for eigen decomposition or "svd" for singular value decomposition.

    Returns:
    loadings (numpy.ndarray): Matrix of feature loadings for each principal component.
    X_projected (numpy.ndarray): Data projected onto the top principal components.
    totalvarianceexplaned (float): Cumulative variance explained by the selected components.
    """
    # Standardize the data: zero mean and unit variance
    X_scaled = (x - np.mean(x , axis = 0))/np.std(x,axis=0)
    # Compute the covariance matrix of the scaled data
    cov_mat = np.cov(X_scaled, ddof = 1, rowvar=False )
    # Compute eigenvalues and eigenvectors of the covariance matrix
    eigen_values , eigen_vectors = np.linalg.eig(cov_mat)
    # Sort the eigenvalues and corresponding eigenvectors in descending order
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvalue = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]
    # Compute the proportion of explained variance for each principal component
    explained_variance = sorted_eigenvalue / np.sum(sorted_eigenvalue)
    #scree plots
    plt.plot(np.arange(1, len(explained_variance)+1), np.cumsum(explained_variance), marker='o')
    plt.xlabel('Principal Component')
    plt.ylabel('Cummulative Variance Explained')
    plt.title('Scree Plot')
    plt.show()
    plt.plot(np.arange(1, len(explained_variance)+1), explained_variance, marker='o')
    plt.ylabel('Variance Explained')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()
    # Calculate cumulative variance explained by the selected components
    totalvarianceexplaned = np.cumsum(explained_variance[0:dims])[-1]
    if method == "eig" :
        # Select the top eigenvectors (principal axes)
        Qtop = sorted_eigenvectors[:,0:dims] 
        # Calculate feature loadings (scaled eigenvectors)
        loadings = np.zeros((dims, X.shape[1]))  # Initialize loadings matrix
        for i in range(dims):
            loadings[i] = np.dot(Qtop[:, i], np.sqrt(sorted_eigenvalue[i]))
        # Project the data onto the top principal components
        X_projected = np.dot(Qtop.transpose() , X_scaled.transpose() ).transpose()
    elif method == "svd":
        # Perform Singular Value Decomposition
        U, s, Vt = np.linalg.svd(X_scaled)
        # Vt contains the right singular vectors (principal axes)
        Vtop = Vt[0:dims,:]
        # Calculate feature loadings using the corresponding eigenvalues
        loadings = np.zeros((dims, X.shape[1]))  # Initialize loadings matrix
        for i in range(dims):
            loadings[i] = np.dot(Vtop[i,:].transpose(), np.sqrt(sorted_eigenvalue[i]))
        # Project the data onto the top principal components
        X_projected = np.dot(Vtop,X_scaled.transpose()).transpose()
    else:
        raise ValueError('Input Error. Method should be `eig` or `svd`')
    return  loadings, X_projected,totalvarianceexplaned

X, y = make_blobs(n_samples=1000,n_features=20,centers=3,cluster_std=1.0,random_state=87)
print(conventionalPCA(X,2,"svd"))

###################################### Clustering ###################################################################

# ---------------------- Initialization ----------------------
def initialize_random_centroids(X, K):
    """
    Randomly initialize K centroids from the dataset X.
    """
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    return centroids

# ---------------------- Distance Functions ----------------------
def euclidean_distance(a, b):
    """
    Compute the Euclidean distance between two vectors.
    """
    return np.sqrt(np.sum((a - b) ** 2))

def cityblock_distance(a, b):
    """
    Compute the Manhattan (L1) distance between two vectors.
    """
    return np.sum(np.abs(a - b))

# ---------------------- Label Assignment ----------------------
def assign_labels_euclidean(X, centroids):
    """
    Assign each point in X to the nearest centroid using Euclidean distance.
    """
    labels = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        distances = np.array([euclidean_distance(X[i], centroid) for centroid in centroids])
        labels[i] = np.argmin(distances)
    return labels

def assign_labels_cityblock(X, centroids):
    """
    Assign each point in X to the nearest centroid using Manhattan distance.
    """
    labels = np.zeros(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distances = np.array([cityblock_distance(X[i], centroid) for centroid in centroids])
        labels[i] = np.argmin(distances)
    return labels

def assign_labels_mahalanobis(X, centroids, inv_cov_matrix):
    """
    Assign each point in X to the nearest centroid using Mahalanobis distance.
    """
    distances = np.array([[mahalanobis(x, centroid, inv_cov_matrix) for centroid in centroids] for x in X])
    labels = np.argmin(distances, axis=1)
    return labels

# ---------------------- Covariance Matrix ----------------------
def compute_covariance_matrix(X):
    """
    Compute the covariance matrix of the dataset X.
    """
    return np.cov(X, rowvar=False)


# ---------------------- Centroid Update ----------------------
def update_centroids(X, labels, K):
    """
    Recalculate the centroids as the mean of points assigned to each cluster.
    """
    new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])
    return new_centroids

# ---------------------- Silhouette Score ----------------------
def compute_silhouette_scores(X, labels,distance):
    """
    Compute the mean silhouette score for the clustering using either Euclidean or Manhattan distances.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    if n_clusters == 1:
        return np.zeros(n_samples)  # Silhouette scores are equal to 0 if there is only one cluster
    silhouette_scores = np.zeros(n_samples)
    if distance == "Euclidean":
        for i in range(n_samples):
            same_cluster = labels == labels[i]
        
            # Mean distance of points in the same cluster
            a = np.mean([euclidean_distance(X[i], X[j]) for j in range(n_samples) if same_cluster[j] and i != j])
        
            #Mean distance of points in the nearest cluster
            b = np.min([np.mean([euclidean_distance(X[i], X[j]) for j in range(n_samples) if labels[j] == label]) for label in unique_labels if label != labels[i]])
    
            #silhouette scores
            silhouette_scores[i] = (b - a) / max(a, b)
    if distance == "Manhattan":
        for i in range(n_samples):
            same_cluster = labels == labels[i]
            a = np.mean([cityblock_distance(X[i], X[j]) for j in range(len(X)) if same_cluster[j]])
            b = np.min([np.mean([cityblock_distance(X[i], X[j]) for j in range(len(X)) if labels[j] == label]) for label in unique_labels if label != labels[i]])
            silhouette_scores[i] = (b - a) / max(a, b)
    return np.mean(silhouette_scores)

def compute_silhouette_scores_mahalanobis(X, labels, inv_cov_matrix):
    """
    Compute the mean silhouette score for the clustering using Mahalanobis distance.
    """
    n_samples = len(X)
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return np.zeros(n_samples)  
    
    silhouette_scores = np.zeros(n_samples)
    
    for i in range(n_samples):
        same_cluster = labels == labels[i]
        a = np.mean([mahalanobis(X[i], X[j], inv_cov_matrix) for j in range(len(X)) if same_cluster[j]])
        b = np.min([np.mean([mahalanobis(X[i], X[j], inv_cov_matrix) for j in range(len(X)) if labels[j] == label]) for label in unique_labels if label != labels[i]])
        
        silhouette_scores[i] = (b - a) / max(a, b)
    return np.mean(silhouette_scores)

# ---------------------- KMeans Clustering ----------------------
def customKmeans(X, K,distance,max_iterations=1000 ):
    """
    Custom KMeans clustering algorithm supporting Euclidean, Manhattan, and Mahalanobis distances.

    Parameters:
    - X (numpy.ndarray): Input dataset. It should be standardized beforehand.
    - K (int): Number of clusters.
    - distance (str): Distance metric - 'Euclidean', 'Manhattan', or 'Mahalanobis'.
    - max_iterations (int): Maximum number of iterations to run.

    Returns:
    - centroids (numpy.ndarray): Final centroids.
    - labels (numpy.ndarray): Cluster assignments for each point.
    - score (float): Mean silhouette score for the clustering.
    """
    # Ensure the input data is standardized before calling this function
    centroids = initialize_random_centroids(X, K)
    if distance == 'Euclidean':
        print ('Euclidean distance is used.')
        for i in range(max_iterations):
            labels = assign_labels_euclidean(X, centroids)
            new_centroids = update_centroids(X, labels, K)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        score = compute_silhouette_scores(X, labels,distance)
    elif distance =='Manhattan':
        print ('Manhattan distance is used.')
        for i in range(max_iterations):
            labels = assign_labels_cityblock(X, centroids)
            new_centroids = update_centroids(X, labels, K)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        score = compute_silhouette_scores(X, labels,distance)
    elif distance =='Mahalanobis':
        print ('Mahalanobis distance is used.') 
        cov_matrix = compute_covariance_matrix(X)
        inv_cov_matrix = np.linalg.inv(cov_matrix)
        for i in range(max_iterations):
            labels = assign_labels_mahalanobis(X, centroids,inv_cov_matrix)
            new_centroids = update_centroids(X, labels, K)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        score = compute_silhouette_scores_mahalanobis(X, labels,inv_cov_matrix)
    else:
         raise ValueError('Input Error. Distance should be Euclidean, manhattan or mahalanobis')
    return (centroids, labels,score)

# ---------------------- Log Likelihood ----------------------
def log_likelihood(X, means, covariances, p):
    """
    Compute the total log-likelihood of the data given current GMM parameters.
    """
    log_likelihood = 0
    for k in range(len(means)):
        log_likelihood += p[k] * multivariate_normal.pdf(X, means[k], covariances[k])
    return np.sum(np.log(log_likelihood))

# ---------------------- E-Step ----------------------
def Estep(X,K, means, covariances, p):
    """
    E-step: Compute the responsibilities (posterior probabilities for each cluster).
    """
    n_samples= X.shape[0]
    gamma = np.zeros((n_samples, K))
    for k in range(K):
        gamma[:, k] = p[k] * multivariate_normal.pdf(X, means[k], covariances[k])
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma

# ---------------------- M-Step ----------------------
def Mstep(X,K, gamma):
    """
    M-step: Update mixture weights, means, and covariances based on responsibilities.
    """
    n_samples = X.shape[0]
    n_features = X.shape[1]
    Nk = gamma.sum(axis=0)
    p = Nk / n_samples
    means = np.dot(gamma.T, X) / Nk[:, np.newaxis]
    covariances = np.zeros((K, n_features, n_features))
    for k in range(K):
        diff = X - means[k]
        covariances[k] = np.dot(gamma[:, k] * diff.T, diff) / Nk[k]
    
    return means, covariances, p

# ---------------------- MoG Model ----------------------
def customMoG(X,K, init,max_iterations = 100,):
    """
    Fit a Gaussian Mixture Model using the EM algorithm.

    Parameters:
    - X (numpy.ndarray): Pre-scaled input data of shape (n_samples, n_features).
    - K (int): Number of clusters/components.
    - init (str): Initialization method ('random' or 'kmeans').
    - max_iterations (int): Maximum number of EM iterations.

    Returns:
    - means (numpy.ndarray): Final component means.
    - labels (numpy.ndarray): Cluster assignments (argmax of responsibilities).
    """
    if init == 'random':
        print ('Random initialization of the parameters.')
        n_samples, n_features = X.shape
        means = X[np.random.choice(n_samples, K, replace=False)]
    elif init == 'kmeans':
        print ('Kmeans is running to initialize the parameters.')
        means = customKmeans(X,K,"Euclidean")[0]
    else:
        raise ValueError('Input Error. Init should be random or kmeans.')
    covariances = np.array([np.cov(X, rowvar=False)] * K)
    p = np.ones(K) / K
    loglikelihood = log_likelihood(X, means, covariances, p)
    for i in range(max_iterations):
        gamma = Estep(X,K, means, covariances, p)
        means, covariances, p = Mstep(X,K, gamma)
        updated_log_likelihood = log_likelihood(X, means, covariances, p)
        
        if abs(updated_log_likelihood - loglikelihood) < 1e-4:
            break
        loglikelihood = updated_log_likelihood
    
    return means, np.argmax(gamma,axis=1)

# ---------------------- Plotting ----------------------
def plot_2d(X, labels, centroids, title):
    """
    Plot the 2D data with cluster labels and centroids.

    Parameters:
    - X (numpy.ndarray): 2D input data.
    - labels (numpy.ndarray): Cluster assignments.
    - centroids (numpy.ndarray): Coordinates of cluster centroids.
    - title (str): Plot title.
    """
    """Plots the data points and centroids in 2D."""
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=300, c='red', marker='X')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(scatter)
    plt.show()

X, y = make_blobs(n_samples=1000,n_features=10,centers=4,cluster_std=3.0,random_state=0)
#scaling
X = (X - np.mean(X , axis = 0))/np.std(X,axis=0)
#check mean of silhouette scores calculated via euclidean distances for k in [2,7]
for k in range(2,8):
    score = customKmeans(X,k,"Euclidean")[2]
    print(f"clusters: {k}, Silhouette Score: {score}")
#check mean of silhouette scores calculated via city block distances for k in [2,7]
for k in range(2,8):
    score = customKmeans(X,k,"Manhattan")[2]
    print(f"clusters: {k}, Silhouette Score: {score}")
#check mean of silhouette scores calculated via mahalanobis distances for k in [2,7]
for k in range(2,8):
    score = customKmeans(X,k,"Mahalanobis")[2]
    print(f"clusters: {k}, Silhouette Score: {score}")
#rand index between true labels and predicted ones
for i in ["Euclidean","Manhattan"]:
    labels = customKmeans(X,4,i)[1]
    rand_index = adjusted_rand_score(y, labels)
    print(f"Distance : {i}, Rand Index: {rand_index}")
labels = customKmeans(X,5,"Mahalanobis")[1]
rand_index = adjusted_rand_score(y, labels)
print(f"Distance : Mahalanobis, Rand Index: {rand_index}")
# Plot the K-means clustering result and the actual clusters
for i in ["Euclidean","Manhattan"]:
    centroids = customKmeans(X,4,i)[0]
    labels = customKmeans(X,4,i)[1]
    plot_2d(X, labels, centroids, title="K-Means Clustering")
    plot_2d(X, y, centroids, title="Actual Clustering")
centroids = customKmeans(X,5,"Mahalanobis")[0]
labels = customKmeans(X,5,"Mahalanobis")[1]
plot_2d(X, labels, centroids, title="K-Means Clustering")
#MoG clustering
centroids_random, labels_random = customMoG(X,4,"random")
centroids_kmeans,labels_kmeans = customMoG(X,4,"kmeans")
# Plot the MoG clustering result and the actual clusters
plot_2d(X,labels_random, centroids_random, title="MoG Clustering with random initialization")
plot_2d(X,labels_kmeans,centroids_kmeans, title="MoG Clustering with kmeans initialization")

