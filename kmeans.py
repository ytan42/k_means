from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import numpy as np

np.random.seed(123)

def init_centroid(X : np.ndarray, k : int = 2) -> np.ndarray:
    """Randomly picks k records from X and use them as
    the initial centroids.
    """
    row_idx = np.random.choice(len(X), size=k, replace=False)
    centroids = X[row_idx, :]
    return centroids

def euclidean_dist(X1 : np.ndarray, X2 : np.ndarray) -> np.float64:
    """Calculate euclidean distance between 2 points
    """
    return np.sqrt(sum((np.array(X1) - np.array(X2)) ** 2))

def pairwise_dist(X : np.ndarray,
                 centroids : np.ndarray, 
                 dist_func : callable) -> np.ndarray:
    """Calculate distance between data points (X) to the centroids.
    Returning an array where each row is a data point in X and each column
    is a class defined by k. Cell [0, 1] means the distance between the first
    data point to the second class.
    """
    dists = list()

    for centroid in centroids:
        dist = np.apply_along_axis(dist_func, axis=1, arr=X, X2=centroid)
        dists.append(dist)

    dists_array = np.stack(dists, axis=0).T
    return dists_array

def assign(dists_array : np.ndarray):
    """Assign the class based on the distance calculation"""
    return np.argmin(dists_array, 1).reshape(-1, 1)

def calculate_centroid(X : np.ndarray,
                       assignment : np.ndarray) -> np.ndarray:
    """Calculate centroids based on cluster assignment
    """
    new_centroids = list()
    for cluster in np.unique(assignment):
        new_centroid = np.mean(X, axis=0, where = (assignment == cluster), keepdims=False)
        new_centroids.append(new_centroid)
    return np.stack(new_centroids, axis=0)

def vars_within(dists_array : np.ndarray,
                assignment : np.ndarray) -> np.float64:
    """Calculate the total variances within clusters
    """
    assignment_vector = assignment == np.unique(assignment)
    return np.sum(dists_array * assignment_vector)
    
def fit_KMeans(X : np.ndarray, 
               k : int, 
               max_iter : int = 50,
               random_starts : int = 3) -> np.ndarray:
    """Train the K means model
    """

    models = list()

    for s in range(random_starts):

        model = dict()
        model['random_start'] = s

        # Init centroids and assign class based on distance to centroids
        centroids = init_centroid(X=X, k=k)
        dists = pairwise_dist(X=X, centroids=centroids, dist_func=euclidean_dist)
        assignment = assign(dists_array=dists)

        total_dists = dict()
        total_dists[0] = vars_within(dists_array=dists, assignment=assignment)

        for i in np.arange(start=1, stop=max_iter, step=1):
            new_centroids = calculate_centroid(X=X, assignment=assignment)
            dists = pairwise_dist(X=X, centroids=new_centroids, dist_func=euclidean_dist)
            assignment = assign(dists_array=dists)

            total_dists[i] = vars_within(dists_array=dists, assignment=assignment)
        
        model['centroids'] = new_centroids
        model['vars_within'] = total_dists[len(total_dists) - 1]
        model['k'] = k
        
        models.append(model)

    # Output the model with least variance within
    idx = np.argmin([e['vars_within'] for e in models])
    return models[idx]

# kmeans_model = fit_KMeans(X, k=5)

def pred_kmeans(x : np.ndarray, 
                kmeans_model : any) -> np.ndarray:
    
    centroids = kmeans_model['centroids']
    dists = pairwise_dist(X=x, centroids=centroids, dist_func=euclidean_dist)
    return assign(dists_array=dists)

def main():

    X, y = make_blobs(n_samples=1000, n_features=10, centers=2, random_state=123)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    model = fit_KMeans(X=X_train, k=2)
    pred = pred_kmeans(x=X_test, kmeans_model=model)

    print(np.ravel(pred))
    print(y_test)


if __name__ == "__main__":
    main()