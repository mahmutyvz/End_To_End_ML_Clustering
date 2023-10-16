import pandas as pd
import numpy as np


def time_day_range(x):
    """
    The time_day_range function is designed to categorize a given input value representing an hour of the day into one
    of four possible time-of-day categories: "morning," "afternoon," "evening," or "night."

    Parameters:
    x: An integer representing the hour of the day (24-hour format) to be categorized.

    Returns:
    A string representing the categorized time of day based on the input hour.

    Time of Day Categories:
    "morning": If the input hour falls within the range of 6 to 11 (inclusive), it is considered morning.

    "afternoon": If the input hour falls within the range of 12 to 17 (inclusive), it is considered afternoon.

    "evening": If the input hour falls within the range of 18 to 22 (inclusive), it is considered evening.

    "night": If the input hour is not covered by any of the above ranges (i.e., not within 6-11, 12-17, or 18-22),
    it is categorized as night.
    """
    return f'{"morning" if x in range(6, 12) else "afternoon" if x in range(12, 18) else "evening" if x in range(18, 23) else "night"}'


def order_number_group(num_orders):
    """
    The order_number_group function categorizes the given input value representing the number of orders into predefined
    groups based on specific ranges.

    Parameters:

    num_orders: An integer representing the number of orders to be categorized.

    Returns:

    A string representing the group that the input number of orders falls into.

    Order Number Groups:

    The function categorizes the input number of orders into the following predefined groups:

    "1-10 orders": If the input number of orders falls within the range of 1 to 10 (inclusive).

    "11-20 orders": If the input number of orders falls within the range of 11 to 20 (inclusive).

    "21-30 orders": If the input number of orders falls within the range of 21 to 30 (inclusive).

    ...

    "91-100 orders": If the input number of orders falls within the range of 91 to 100 (inclusive).

    "More than 100 orders": If the input number of orders is greater than 100, and it doesn't fall within any of the
     predefined ranges.
    """
    ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 60), (61, 70), (71, 80), (81, 90), (91, 100)]
    for r in ranges:
        if num_orders in range(r[0], r[1] + 1):
            return f"{r[0]}-{r[1]} orders"
    return "More than 100 orders"


def cluster_id_crosstab(data, labels):
    """
    The cluster_id_crosstab function generates a cross-tabulation of department-wise purchase counts for each user,
     alongside their assigned cluster category based on the input labels.

    Parameters:

    data: A pandas DataFrame containing purchase data, with at least the columns "user_id" and "department"
    representing user IDs and purchased departments, respectively.

    labels: A list or array containing cluster labels assigned to each user.

    Returns:

    A pandas DataFrame containing the cross-tabulation of department-wise purchase counts for each user,
     with an additional "Cluster Category" column indicating the cluster category assigned to each user.
    """
    clst_prd = pd.crosstab(data['user_id'], data['department'])
    clst_prd['clusters'] = labels
    unique_labels = np.unique(labels)
    clst_prd["Cluster Category"] = "No Data"
    for i in range(len(unique_labels)):
        clst_prd["Cluster Category"].loc[clst_prd["clusters"] == i] = f"Cluster {i}"
    clst_prd.drop('clusters', axis=1, inplace=True)
    return clst_prd
