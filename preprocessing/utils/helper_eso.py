"""
Script Name:    helper_eso.py

Description:    Helper functions for EsoTrace.
                
Features:       Label generation, Filtering for non-intentional interactions.

Author:         Jeremy Wanner (EGD1)
"""
import pandas as pd 
import numpy as np
import os

def load_csv_to_dic(path, v_names, ext="_eso.csv") -> dict:
    """ Return: dic with v_name as key and dataframe as value """
    dic = {}
    for vn in v_names:
        dic[vn] = pd.read_csv(os.path.join(path, vn + ext), parse_dates=['datetime'])
    return dic

def make_unique_dataframe(df_dic, v_names) -> pd.DataFrame:
    dfEso = pd.DataFrame()
    for vn in v_names:
        dfEso = pd.concat([dfEso, df_dic[vn]], ignore_index=True)
    return dfEso

def get_domains_from_id(df):
    dfId = df.ID.str.split('/')
    domain = dfId.apply(lambda x: x[1] if len(x) > 1 else 'None')
    function = dfId.apply(lambda x: x[2] if len(x) > 1 else 'None')
    return domain, function

def keep_buttonPress_log(df, ids):
    """ Keep only if closest log is a buttonPress"""
    ids_to_keep = []
    for id in ids:
        time_diff = (df.datetime[id] - df.datetime).abs()
        sorted_indices = time_diff.argsort()
        previous_index = sorted_indices.iloc[[1,2]] # take first two closest
        previous = df.ID[previous_index].values
        if "/main/buttonPress" in previous:
            ids_to_keep.append(id)
    return ids_to_keep 

def get_index_close_BeginTime(df, ids):
    """ Remove labels to close to the beginTime """
    ids_cloese_to_begin = []
    for id in ids:
        #time_diff = (df.loc[id, 'BeginTime'] - df.loc[id, 'datetime']).abs()
        time_diff = (df.datetime[id] - df.BeginTime[id])
        if time_diff < pd.Timedelta(minutes=1):
            ids_cloese_to_begin.append(id)
    return ids_cloese_to_begin

def get_index_of_consecutive_to_delete(df, ids):
    """ If interactions are consecutive delete them and keep only the last most recent. 
        Return: ids to delete
    """
    time_diff = df.loc[ids, "datetime"].diff()
    time_diff_shift = time_diff.fillna(pd.Timedelta(0)).shift(-1).fillna(pd.Timedelta(minutes=5))
    time_threshold = pd.Timedelta(minutes=1) # Parameter for threshold
    return ids[time_diff_shift < time_threshold]

def clean_radio_label(df, label_to_remove):
    ids = df[df.Label == label_to_remove].index
    ids_1 = get_index_close_BeginTime(df, ids) 
    ids_2 = get_index_of_consecutive_to_delete(df, ids)
    ids_to_delete = list(set(ids_1) | set(ids_2))
    return df.drop(ids_to_delete)

def label_dataframe(df, labels):
    """ 
    Create a new dataframe that includes only the relevant HMI singals.
    Add a columns "Label" to encode the interaction name.

    Args: 
        df -> pd.dataframe: original dataframe
        labels -> dict: as key labels, as value tuple (id, functionValue)

    Return: dataframe ordered by datetime
    """
    new_df = pd.DataFrame()
    for label in labels.keys():
    
        ids = get_action_ids(df, label, labels[label][0], labels[label][1])
        labeled_df = df.iloc[ids].copy()
        labeled_df.loc[ids, "Label"] = np.repeat(label, len(ids))
        new_df = pd.concat([new_df, labeled_df], ignore_index=True)
    new_df = new_df.sort_values(by="datetime")
    return new_df

def get_action_ids(df, label, id, functionValue):
    if label in ["clima/AC/on"]:
        ids = df[np.logical_and(df.ID == id, df.FunctionValue == functionValue)].index
        ids = keep_buttonPress_log(df, ids)
    elif type(functionValue) == list:
        ids = np.where(np.logical_and(df.ID == id, df.FunctionValue.str.contains('|'.join(functionValue))))[0]
    elif functionValue == "":
        ids = np.where(df.ID == id)[0]
    else:
        ids = np.where(np.logical_and(df.ID == id, df.FunctionValue == functionValue))[0]
    return ids  

