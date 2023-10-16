import pandas as pd
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from src.data.preprocess_data import time_day_range,order_number_group
import plotly.subplots as sp
def missing_control_plot(df,streamlit=False):
    """
    The missing_control_plot function generates a bar plot to visualize the amount of missing (null) data in each column of a DataFrame.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.

    """
    df_null = df.isna().sum()
    missing_df = pd.DataFrame(
                  data=[df_null],
                  columns=df.columns,
                  index=["Null Size"]).T.sort_values("Null Size", ascending=False)
    missing_df = missing_df.loc[(missing_df["Null Size"] > 0)]
    fig = px.bar(missing_df,x=missing_df.index, y="Null Size", hover_name='Null Size',
                 color='Null Size',labels={
                         "index": "Columns",
                         },
                 color_discrete_sequence=['#D81F26'],template='plotly_dark',
                 title="Dataset Null Graph", width=1400, height=700)
    fig.update_layout(barmode='group')
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)

def count_plot(df,variable_type,streamlit=False):
    """
    The count_plot function generates bar plots to visualize the distribution of categorical or numerical variables in a DataFrame.

    Parameters:
    df: A pandas DataFrame containing the data to be visualized.

    variable_type: A string indicating the type of variables to be visualized. Use 'num' for numerical variables and 'cat' for categorical variables.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    if variable_type == 'num':
        liste = df.loc[:, df.dtypes != "object"].columns
        template = 'plotly_dark'
    if variable_type == 'cat':
        liste = df.loc[:, df.dtypes == "object"].columns
        template = 'ggplot2'
    for i in liste:
        x = df[i].unique()
        if len(x) > 35:
            print(f"{i} kolonunun unique sayısı 25' den fazla")
            continue
        y = [df[i][df[i] == j].count() for j in x]
        bar = px.bar(df, x=x, y=y, text=y, labels={
            "x": i,
            "y": "count",
        },
                     color_discrete_sequence=['#D81F26'], color=y,
                     template=template, title="Unique Count",
                     width=1000, height=500)

        if not streamlit:
            bar.show()
        else:
            st.plotly_chart(bar, use_container_width=True)

def buying_product_behavior(df,streamlit=False):
    """
    The buying_product_behavior function creates a treemap graph to visualize the purchasing behavior based on the number of products added to the cart in each order.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    grouped = df.groupby("order_id")["add_to_cart_order"].max().reset_index()
    grouped = grouped["add_to_cart_order"].value_counts().reset_index()
    grouped.columns = ["Number of products added to cart", "Number of unique orders"]

    fig = px.treemap(grouped, path=["Number of products added to cart"], values="Number of unique orders",
                     labels={"Number of products added to cart": "Number of products added to cart",
                             "Number of unique orders": "Number of unique orders"},
                     title="Purchasing Behavior by Number of Products Added to Cart", template='plotly_dark')

    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)
def time_order(df,streamlit=False):
    """
    The time_order function creates a funnel graph to visualize the number of unique orders based on the time of the day that orders were placed.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    grouped = df.groupby('order_hour_of_day', as_index=True).agg({'user_id': 'count'}).sort_values(by='user_id',
                                                                                                     ascending=False)
    grouped.reset_index(inplace=True)
    fig = px.funnel(grouped, x='order_hour_of_day', y='user_id',
                    labels={'order_hour_of_day': 'Time of the day', 'user_id': 'Number of unique orders'},
                    title='Number of Unique Orders by Time of the Day', color='order_hour_of_day',
                    template='plotly_dark')

    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)
def hour_of_day_order_count(df,streamlit=False):
    """
    The hour_of_day_order_count function creates a set of bar plots to visualize the order counts for each day of the week and different hours of the day.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """

    day_hour_df = df.groupby(["order_dow", "order_hour_of_day"])["order_number"].count().reset_index()
    day_hour_df_piv = day_hour_df.pivot('order_dow', 'order_hour_of_day', 'order_number') / df.shape[0]

    day_hour_df_piv.reset_index(inplace=True)

    figs = []

    templates = ['plotly', 'plotly_dark', 'plotly_white', 'ggplot2', 'seaborn', 'simple_white', 'gridon']
    for i in range(len(day_hour_df_piv)):
        day_df = day_hour_df_piv.iloc[i]
        labels = day_df.index[1:]
        values = day_df[1:].values
        day_name = f"{day_df['order_dow'].astype('str').replace('0.0', 'Monday').replace('1.0', 'Tuesday').replace('2.0', 'Wednesday').replace('3.0', 'Thursday').replace('4.0', 'Friday').replace('5.0', 'Saturday').replace('6.0', 'Sunday')}"

        fig = px.bar(x=labels, y=values, template=templates[i])

        fig.update_layout(
            showlegend=True,
            title=f"Order Count for {day_name}",
            xaxis=dict(title="Hour of Day"),
            yaxis=dict(title="Order Count")
        )

        figs.append(fig)

    for fig in figs:
        if not streamlit:
            fig.show()
        else:
            st.plotly_chart(fig, use_container_width=True)

def hour_of_day_number_orders(df,streamlit=False):
    """
    The hour_of_day_number_orders function generates a stacked bar plot to visualize the number of orders made by day of the week and time of day.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    df['time_day_range'] = df['order_hour_of_day'].apply(time_day_range)
    orders_by_day_time = df.pivot_table(
        index='order_dow',
        columns='time_day_range',
        values='user_id',
        aggfunc='count'
    )
    fig = go.Figure()

    for col, day in enumerate(orders_by_day_time.index):
        fig.add_trace(go.Bar(
            x=orders_by_day_time.columns,
            y=orders_by_day_time.loc[day],
            name='Day {}'.format(
                str(day).replace('0', 'Monday').replace('1', 'Tuesday').replace('2', 'Wednesday').replace('3',
                                                                                                          'Thursday').replace(
                    '4', 'Friday').replace('5', 'Saturday').replace('6', 'Sunday')),
        ))

    fig.update_layout(
        title='The time when the order was made by day and time of day',
        xaxis=dict(title='Time of Day'),
        yaxis=dict(title='Number of orders'),
        xaxis_tickangle=-45,
        barmode='stack',
        legend=dict(title='Orders Made')
    )

    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)

def number_of_orders_group(df,streamlit=False):
    """
    The number_of_orders_group function generates a polar bar plot (also known as a "radar chart") to visualize the distribution of orders among different order number groups.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    df['order_number_group']=df['order_number'].apply(order_number_group)
    orders_by_group = df.groupby('order_number_group')['user_id'].count().sort_values(ascending=False)

    percentage_by_group = orders_by_group / orders_by_group.sum() * 100

    fig = go.Figure()

    fig.add_trace(go.Barpolar(
        r=percentage_by_group.values,
        theta=percentage_by_group.index,
        width=0.5,
        marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5)),
        opacity=0.8,
        text=percentage_by_group.values,
        hovertemplate='%{text:.2f}%<br>Group %{theta}<extra></extra>'
    ))

    fig.update_layout(
        title='Number of Orders by Group',
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(percentage_by_group.values) + 5]
            )
        ),
        width=1200,
        height=700
    )

    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)
def number_of_orders_product_top10(df,streamlit=False):
    """
    The number_of_orders_product_top10 function generates a bar plot to visualize the top 10 most ordered products.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    top_products = df.groupby('product_name')['user_id'].count().sort_values(ascending=False).head(10)

    fig = px.bar(top_products, x=top_products.index, y=top_products.values, color=top_products.index,
                 color_discrete_sequence=px.colors.qualitative.Pastel, title='Top 10 Products')

    fig.update_layout(
        xaxis=dict(title='Product Name', tickfont=dict(size=13)),
        yaxis=dict(title='Number of Orders', tickfont=dict(size=13)),
        showlegend=False
    )

    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)

def rfm_analysis(df,streamlit=False):
    """
    The rfm_analysis function generates a 3D scatter plot to perform RFM (Recency, Frequency, Monetary) analysis using the first 50 records from the DataFrame.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    df = df.reset_index()
    plot_rfm = df.head(50).set_index('user_id')

    fig = px.scatter_3d(plot_rfm, x='Recency', y='Frequency', z='Monetary',
                        color=plot_rfm.index, width=800, height=500)

    fig.update_layout(
        title='Multiple Series 3D Bar Chart',
        scene=dict(
            xaxis=dict(title='Recency'),
            yaxis=dict(title='Frequency'),
            zaxis=dict(title='Monetary')
        ),
        showlegend=True
    )

    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)

def clusterid_rfm(df,streamlit=False):
    """
    The clusterid_rfm function generates a set of box plots to visualize the distribution of RFM (Recency, Frequency, Monetary) values for different cluster IDs.

    Parameters:
    df: A pandas DataFrame containing the data to be analyzed.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    fig1 = px.box(df, x='Cluster_Id', y='Recency')
    fig2 = px.box(df, x='Cluster_Id', y='Monetary')
    fig3 = px.box(df, x='Cluster_Id', y='Frequency')
    subplot_titles = ['Recency', 'Monetary', 'Frequency']
    fig = sp.make_subplots(rows=1, cols=3, subplot_titles=subplot_titles)
    fig.add_trace(fig1.data[0], row=1, col=1)
    fig.add_trace(fig2.data[0], row=1, col=2)
    fig.add_trace(fig3.data[0], row=1, col=3)
    fig.update_layout(title_text="RFM Analysis")
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)

def labels_visualization(data,x,y,labels,streamlit=False):
    """
    The labels_visualization function generates a scatter plot to visualize data points based on two specified features (x and y) with different labels.

    Parameters:

    data: A pandas DataFrame containing the data to be visualized.

    x: The name of the feature to be plotted on the x-axis.

    y: The name of the feature to be plotted on the y-axis.

    labels: The labels for coloring and symbolizing the data points.

    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    plt.figure(figsize=(15, 7))
    fig = px.scatter(data_frame=data, x=x, y=y, color=labels, symbol=labels,
                     width=800, height=400, opacity=0.7)
    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)

def product_visualization(data, x, streamlit=False):
    """
    The product_visualization function generates a set of box plots to visualize the distribution of different product-related features across clusters.

    Parameters:
    data: A pandas DataFrame containing the data to be analyzed.
    x: The name of the feature (x-axis) used to categorize the data.
    streamlit: A boolean flag indicating whether the function should generate plots for Streamlit display. Defaults to False.
    """
    num_columns = len(data.columns) - 1
    column_list = data.columns.tolist()
    column_list.remove('Cluster Category')
    num_rows = int(num_columns/7)
    fig = sp.make_subplots(rows=num_rows, cols=num_columns // num_rows, subplot_titles=column_list)

    for index, col_name in enumerate(column_list, start=1):
        trace = px.box(data, x=x, y=col_name).data[0]
        row = (index - 1) // (num_columns // num_rows) + 1
        col = (index - 1) % (num_columns // num_rows) + 1
        fig.add_trace(trace, row=row, col=col)

    fig.update_layout(width=2000, height=1500,title_text="Product Analysis")

    if not streamlit:
        fig.show()
    else:
        st.plotly_chart(fig, use_container_width=True)
