from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from src.models.metrics import calculate_silhouette_score
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import os
from paths import Path
from joblib import dump

def fit_kmeans(dataset, scaler, n_clusters, random_state=42,streamlit=False,manual=False):
    """
    The fit_kmeans function applies the KMeans clustering algorithm to a given dataset using a specified scaler and
    number of clusters. It calculates model performance metrics, saves the trained model, and optionally visualizes the results.

    Parameters:

    dataset: A pandas DataFrame or numpy array containing the data to be clustered.

    scaler: A scikit-learn transformer for preprocessing the data, usually used for feature scaling.

    n_clusters: An integer specifying the number of clusters for the KMeans algorithm.

    random_state: An integer specifying the random seed for reproducibility. Defaults to 42.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.

    manual: A boolean flag indicating whether to display manual plots. If True, it generates a scatter plot using Plotly
    for manual inspection of clusters. Defaults to False.

    Returns:

    The calculated Silhouette Score rounded to 3 decimal places.
    The inertia score of the KMeans model rounded to 2 decimal places.
    An array of cluster labels assigned by the KMeans model.
    """
    pipeline = Pipeline([
        ('scaler', scaler),
        ('kmeans', KMeans(n_clusters=n_clusters, random_state=random_state))
    ])
    labels = pipeline.fit_predict(dataset)

    silhouette_avg = round(calculate_silhouette_score(dataset, labels), 3)
    inertia_score = round(pipeline.named_steps['kmeans'].inertia_, 2)
    directory = os.path.join(Path.models_path)
    if not os.path.exists(os.path.join(directory, str(f'{type(pipeline.named_steps["kmeans"]).__name__}'))):
        os.makedirs(os.path.join(directory, str(f'{type(pipeline.named_steps["kmeans"]).__name__}')), exist_ok=True)
    model_name = os.path.join(f'{directory}/{str(type(pipeline.named_steps["kmeans"]).__name__)}/{str(type(pipeline.named_steps["kmeans"]).__name__)}_{n_clusters}.gz')
    dump(pipeline, model_name, compress=('gzip', 3))
    if manual:
        fig, ax = plt.subplots(1, figsize=(10, 5))
        fig = px.scatter(data_frame=dataset, x='Monetary', y='Frequency', color=labels,symbol=labels)
        ax.set_title(f'Manual Scatter Plot Cluster : {n_clusters}')
        if not streamlit:
            plt.show(fig)
        else:
            st.plotly_chart(fig, use_container_width=True)
    return silhouette_avg, inertia_score,labels

def optimal_kmeans(dataset, scaler,start=2, end=11, random_state=42, streamlit=False):
    """
    The optimal_kmeans function aims to find the optimal number of clusters for KMeans clustering within a specified range.
    It iterates over different cluster numbers, calculates relevant metrics, and optionally displays a visualization of the evaluation results.

    Parameters:

    dataset: A pandas DataFrame or numpy array containing the data to be clustered.

    scaler: A scikit-learn transformer for preprocessing the data, usually used for feature scaling.

    start: An integer specifying the starting number of clusters to evaluate. Defaults to 2.

    end: An integer specifying the ending number of clusters (exclusive) to evaluate.

    random_state: An integer specifying the random seed for reproducibility. Defaults to 42.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.

    """
    n_clusters = []
    silhouette_scores = []
    inertia_scores = []
    for n_cluster in range(start, end):
        silhouette_avg, inertia_score,labels = fit_kmeans(dataset, scaler, n_cluster, random_state)
        silhouette_scores.append(silhouette_avg)
        n_clusters.append(n_cluster)
        inertia_scores.append(inertia_score)
        st.write("Cluster No : ", n_cluster,
                 "Silhouette Score : ", silhouette_avg,
                 "Inertia : ", inertia_score,
                 "Silhouette Score Delta : ", (silhouette_scores[n_cluster - start] - silhouette_scores[n_cluster - start - 1]).round(3),
                 "Inertia Delta : ", (inertia_scores[n_cluster - start] - inertia_scores[n_cluster - start - 1])
                 )
        if n_cluster == end - 1:
            inertia_trace = go.Scatter(x=n_clusters, y=inertia_scores, mode='lines+markers', name='Inertia')
            silhouette_trace = go.Scatter(x=n_clusters, y=silhouette_scores, mode='lines+markers',
                                          name='Silhouette Score')
            fig = go.Figure(data=[inertia_trace, silhouette_trace])
            fig.update_layout(title='K-Means Evaluation', xaxis_title='Number of Clusters', yaxis_title='Score')

            if not streamlit:
                fig.show()
            else:
                st.plotly_chart(fig, use_container_width=True)

def k_elbow_visualizer(data,scaler,start=2,end=11,streamlit=False):
    """
    The k_elbow_visualizer function is used to visualize the "elbow" point in order to determine the optimal number of
     clusters for KMeans clustering. The elbow point is a point on the curve where the rate of decrease in the
     within-cluster sum of squares (inertia) slows down, indicating a good number of clusters.

    Parameters:
    data: A pandas DataFrame or numpy array containing the data to be clustered.

    scaler: A scikit-learn transformer for preprocessing the data, usually used for feature scaling.

    start: An integer specifying the starting number of clusters to evaluate. Defaults to 2.

    end: An integer specifying the ending number of clusters (exclusive) to evaluate. Defaults to 11.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.suptitle('K Elbow Visualizer', size=18)
    rfm_data_scale = scaler.fit_transform(data)
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(start, end), timings=True)
    visualizer.fit(rfm_data_scale)
    if not streamlit:
        visualizer.show()
    else:
        st.pyplot(fig)

def silhouette_visualizer(data,start=2,end=11,random_state=42,streamlit=False):
    """
    The silhouette_visualizer function is used to visualize the Silhouette Analysis for different numbers of clusters
    in order to determine the quality of clusters in a KMeans clustering model.

    Parameters:

    data: A pandas DataFrame or numpy array containing the data to be clustered.

    start: An integer specifying the starting number of clusters to evaluate. Defaults to 2.

    end: An integer specifying the ending number of clusters (exclusive) to evaluate. Defaults to 11.

    random_state: An integer specifying the random seed for reproducibility. Defaults to 42.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False
    """
    fig, ax = plt.subplots(3, 3, figsize=(13, 8))
    fig.suptitle(f'Silhouette Analysis for {start}-{end-1} Clusters', size=18)
    plt.tight_layout()
    for i in range(start, end):
        km = KMeans(n_clusters=i, random_state=random_state)
        q, mod = divmod(i - 2, 3)
        visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q][mod])
        visualizer.fit(data)
    if not streamlit:
        plt.show()
    else:
        st.pyplot(plt)
