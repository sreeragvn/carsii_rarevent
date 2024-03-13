import pandas as pd


def remove_consecutive_labels(df, label_ids, theta=1):
    """ 
    if consecitive interactions keep only the last 
    Args: 
        df -> pd.Dataframe: dataframe containg the labels
        labels_ids -> pd.Dataframe : indices of label to filter 
        theta -> int: threshold in minute to define what is consecutive
    Return: 
        label_ids -> pd.Dataframe: filtered labels
    """
    time_diff = df.loc[label_ids, "datetime"].diff()
    time_diff_shift = time_diff.fillna(pd.Timedelta(0)).shift(-1).fillna(pd.Timedelta(minutes=5))
    time_threshold = pd.Timedelta(minutes=theta) # Parameter for threshold

    return label_ids[time_diff_shift > time_threshold]

def to_weekday(df_row):
    return df_row.weekday()