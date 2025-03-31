# Assignment Solution

## Instructions

1.  **K-Means Clustering on Iris Dataset:**

    - Use the Iris dataset to perform K-Means clustering.
    - Determine the optimal number of clusters using the elbow method.
    - **Solution:**

    ```python
    from sklearn.datasets import load_iris
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt

    iris = load_iris()
    X = iris.data

    # Determine optimal number of clusters (Elbow Method)
    inertia = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)

    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.show()

    # Apply K-Means with the optimal number of clusters
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)  # Iris dataset has 3 clusters
    kmeans.fit(X)
    labels = kmeans.labels_

    print(labels)
    ```

2.  **DBSCAN on Iris Dataset:**

    - Apply DBSCAN clustering on the Iris dataset.
    - Experiment with different values for `eps` and `min_samples` to find meaningful clusters.
    - **Solution:**

    ```python
    from sklearn.datasets import load_iris
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    iris = load_iris()
    X = iris.data

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)  # Experiment with eps and min_samples
    clusters = dbscan.fit_predict(X_scaled)

    print(clusters)
    ```

## Submission

- Submit the URL of the GitHub Repository that contains your work to NTU black board.
- Should you reference the work of your classmate(s) or online resources, give them credit by adding either the name of your classmate or URL.
