import pandas as pd
def rfm_process(data,streamlit=False):
    """
    The rfm_process function calculates the RFM (Recency, Frequency, Monetary) metrics and scores for customer
    segmentation based on purchase data. It can also format the results for Streamlit display.

    Parameters:

    data: A pandas DataFrame containing purchase data, with at least the columns "user_id", "department",
     and "order_id" representing user IDs, purchased departments, and order IDs, respectively.

    streamlit: A boolean flag indicating whether the function should format the results for display in Streamlit.
    Defaults to False.

    Returns:

    A pandas DataFrame containing the RFM metrics (Frequency, Recency, Monetary) for each user.
    If the streamlit flag is set to True, the index of the DataFrame is set to user IDs.
    """
    prices = {
        'beverages': 7,
        'breakfast': 20,
        'snacks': 11,
        'international': 12,
        'meat seafood': 80,
        'frozen': 90,
        'personal care': 40,
        'babies': 150,
        'deli': 40,
        'dry goods pasta': 30,
        'alcohol': 45,
        'pets': 60,
        'bulk': 30,
        'other': 40,
        'bakery': 10,
        'pantry': 20,
        'dairy eggs': 30,
        'produce': 15,
        'canned goods': 30,
        'household': 400,
        'missing': 0
    }
    total_prices = sum(prices.values())
    prices = {k: v / total_prices * 100 for k, v in prices.items()}
    data['price'] = data['department'].map(prices)
    rfm_m = data.groupby('user_id')['price'].sum().reset_index().rename(columns={'price': 'Monetary'})
    rfm_f = data.groupby('user_id')['order_id'].count().reset_index().rename(columns={'order_id': 'Frequency'})
    rfm_r = (data['order_id'].max() - data.groupby('user_id')['order_id'].max()).reset_index().rename(
        columns={'order_id': 'Recency'})
    rfm = rfm_r.merge(rfm_f, on='user_id').merge(rfm_m, on='user_id')
    rfm['Recency_score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['Frequency_score'] = pd.qcut(rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm['Monetary_score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm['Score'] = rfm['Recency_score'].astype(int) + rfm['Frequency_score'].astype(int) + rfm['Monetary_score'].astype(int)
    rfm_data = rfm[['Frequency', 'Recency', 'Monetary']]
    if streamlit:
        rfm_data.index = rfm['user_id']
    return rfm_data
