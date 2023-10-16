import streamlit as st
import datetime
import warnings
import pandas as pd
from paths import Path
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from src.models.trainer import Trainer
from src.visualization.visualization import *
from sklearn.cluster import DBSCAN,AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from src.features.feature_engineering import rfm_process
from src.models.k_means_train import optimal_kmeans,k_elbow_visualizer,silhouette_visualizer,fit_kmeans
from sklearn.preprocessing import StandardScaler
from src.data.preprocess_data import time_day_range,order_number_group,cluster_id_crosstab
warnings.filterwarnings("ignore")
st.set_page_config(page_title="End_To_End_Clustering",
                   page_icon="chart_with_upwards_trend", layout="wide")
st.markdown("<h1 style='text-align:center;'>Supermarket RFM Analysis</h1>", unsafe_allow_html=True)
st.write(datetime.datetime.now(tz=None))
tabs = ["Data Analysis", "Visualization", "RFM Analysis","Product Analysis", "About"]
page = st.sidebar.radio("Tabs", tabs)
data = pd.read_csv(Path.train_path)
if page == "Data Analysis":
    variables = {
        "descriptions": {
            "order_id": "A unique number to identity the order",
            "user_id": "A unique number to identify the user",
            "order_number": "Number of the order",
            "order_dow": "Day of the Week the order was made",
            "order_hour_of_day": "Time of the order",
            "days_since_prior_order": "History of the order",
            "product_id": "Id of the product",
            "add_to_cart_order": "Number of items added to cart",
            "reordered": "If the reorder took place",
            "department_id": "Unique number allocated to each department",
            "department": "Names of the departments",
            "product_name": "Name of the products",
        }
    }
    profile = ProfileReport(data, title="Cluster Prediction", variables=variables, dataset={
        "description": "The dataset consists of over 1 million purchase records at a renowned Hunter's supermarket",
        "url": "https://www.kaggle.com/datasets/hunter0007/ecommerce-dataset-for-predictive-marketing-2023",
    }, )
    st.title("Data Overview")
    st.write(data)
    st_profile_report(profile)
elif page == "RFM Analysis":
    rfm_data = rfm_process(data)
    option = st.radio(
        'What model would you like to use for training ?',
        ('KMeans', 'GaussianMixture', 'AgglomerativeClustering','DBSCAN'))
    if option == 'KMeans':
        option_k_means = st.selectbox(
            'Which of the K-Means training and visualizations would you like to do ?',
            ('K-Means', 'K-Means Elbow Visualizer','Silhouette Visualizer','Manual K-Means Training'))
        if option_k_means == 'K-Means':
            optimal_kmeans(rfm_data,StandardScaler(),streamlit=True)
        elif option_k_means == 'K-Means Elbow Visualizer':
            k_elbow_visualizer(rfm_data,StandardScaler(),streamlit=True)
        elif option_k_means == 'Silhouette Visualizer':
            silhouette_visualizer(rfm_data,streamlit=True)
        elif option_k_means == 'Manual K-Means Training':
            default_value = 4
            n_cluster = int(st.number_input("Enter the number of clusters you want to train : ",value=default_value))
            st.title(f'Manual Scatter Plot Cluster : {n_cluster}')
            s,i,labels = fit_kmeans(rfm_data,StandardScaler(),n_cluster,streamlit=True,manual=True)
            rfm_data['Cluster_Id'] = labels
            clusterid_rfm(rfm_data,streamlit=True)
    else:
        if option == 'GaussianMixture':
            model = GaussianMixture(n_components=8)
        elif option == 'AgglomerativeClustering':
            model = AgglomerativeClustering()
        elif option == 'DBSCAN':
            model = DBSCAN(eps=20, min_samples=9, metric='euclidean')
        with st.spinner("Training is in progress, please wait..."):
            score,labels = Trainer(rfm_data,model,StandardScaler(),Path.models_path).train()
            st.write("Silhouette Score : ", score)
        with st.spinner("Prediction data is being visualized, please wait..."):
            labels_visualization(rfm_data,'Monetary','Frequency',labels,True)
            rfm_data['Cluster_Id'] = labels
            clusterid_rfm(rfm_data,streamlit=True)
elif page == "Product Analysis":
    rfm_data = rfm_process(data)
    default_value = 4
    n_cluster = int(st.number_input("Enter the number of clusters you want to train : ",value=default_value))
    st.title(f'Manual Scatter Plot Cluster : {n_cluster}')
    s, i, labels = fit_kmeans(rfm_data, StandardScaler(), n_cluster, streamlit=True, manual=True)
    clstr_crosstab = cluster_id_crosstab(data,labels)
    product_visualization(clstr_crosstab,'Cluster Category',streamlit=True)
elif page == "Visualization":
    with st.spinner("Visuals are being generated, please wait..."):
        missing_control_plot(data, streamlit=True)
        data['days_since_prior_order'] = data['days_since_prior_order'].fillna(0).astype(int)
        count_plot(data, variable_type='num', streamlit=True)
        count_plot(data, variable_type='cat', streamlit=True)
        data['time_day_range'] = data['order_hour_of_day'].apply(time_day_range)
        data['order_number_group'] = data['order_number'].apply(order_number_group)
        buying_product_behavior(data,streamlit=True)
        time_order(data,streamlit=True)
        hour_of_day_order_count(data,streamlit=True)
        hour_of_day_number_orders(data,streamlit=True)
        number_of_orders_group(data,streamlit=True)
        number_of_orders_product_top10(data,streamlit=True)
        rfm_data = rfm_process(data,streamlit=True)
        rfm_analysis(rfm_data,streamlit=True)
elif page == "About":
    st.header("Contact Info")
    st.markdown("""**mahmutyvz324@gmail.com**""")
    st.markdown("""**[LinkedIn](https://www.linkedin.com/in/mahmut-yavuz-687742168/)**""")
    st.markdown("""**[Github](https://github.com/mahmutyvz)**""")
    st.markdown("""**[Kaggle](https://www.kaggle.com/mahmutyavuz)**""")
st.set_option('deprecation.showPyplotGlobalUse', False)
