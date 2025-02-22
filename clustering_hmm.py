import numpy as np
from hmmlearn import hmm
from sklearn.cluster import DBSCAN

def detect_fixations_hmm(trajectory, n_states):
    """
    Detect fixations using a Hidden Markov Model (HMM) applied to MALA trajectory points.
    Args:
        trajectory (np.array): Nx2 array of (X, Y) points representing eye movement trajectory.
        n_states (int): Number of hidden states (fixations) to detect.
    Returns:
        np.array: Detected fixations with (X, Y, duration).
    """
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000)
    model.fit(trajectory)
    hidden_states = model.predict(trajectory)
    
    fixations = []
    for i in range(n_states):
        state_points = trajectory[hidden_states == i]
        if len(state_points) > 0:
            centroid = np.mean(state_points, axis=0)
            duration = len(state_points) # Assuming 10 ms per point
            fixations.append([centroid[0], centroid[1], duration])
    
    return np.array(fixations)

def detect_fixations_dbscan(trajectory, eps, min_samples, time_weight):
    """
    Detect fixations using DBSCAN clustering on MALA trajectory points, considering spatial and temporal information.
    
    Args:
        trajectory (np.array): Nx2 array of (X, Y) points representing eye movement trajectory from MALA.
        eps (float): Maximum distance between samples to be considered part of the same cluster.
        min_samples (int): Minimum number of points required to form a cluster.
        time_weight (float): Weight factor for the temporal dimension to influence clustering.
    
    Returns:
        np.array: Detected fixations with (X, Y, duration).
    """
    if trajectory.shape[1] != 2:
        raise ValueError("Trajectory must have two columns: X, Y")
    
    # Creiamo la colonna temporale assumendo 10 ms per punto
    num_points = trajectory.shape[0]
    time_column = np.linspace(0, num_points * 10, num_points)  # Ogni punto dista 10ms dal successivo
    
    # Normalizziamo il tempo e applichiamo il peso
    time_scaled = (time_column / np.max(time_column)) * time_weight
    data = np.column_stack((trajectory[:, 0], trajectory[:, 1], time_scaled))
    
    # Applichiamo DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(data)
    labels = clustering.labels_
    
    fixations = []
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)  # Rimuoviamo i punti considerati "rumore"

    for label in unique_labels:
        cluster_points = data[labels == label]
        if len(cluster_points) > 0:

            centroid = np.mean(cluster_points[:, :2], axis=0)

            duration = len(cluster_points) # Durata in ms (10 ms per punto)
            fixations.append([centroid[0], centroid[1], duration])


    fixations = np.array(fixations)


    # Ensure minimum fixations constraint
    #if len(fixations) < min_fixations:
        # Adjust clustering parameters to allow more fixations
        #new_eps = eps * 1.2  # Increase eps to merge more points
        #new_min_samples = max(1, min_samples - 1)  # Decrease min_samples to allow smaller clusters
        
        #return detect_fixations_dbscan(trajectory, new_eps, new_min_samples, time_weight, min_fixations)
    return fixations