"""
Script Name:    04_merging.py

Description:    Extend MF4 dataset with EsoTrace Labels.

Features:       Data merging, session filtering, feature engineering
"""
import pandas as pd 
import sys
import os
from tqdm import tqdm
import utils.helper_eso as helper_eso
import utils.helper_labeling as helper_labeling

PATH_TO_LOAD = "./data/03_Labeled"
PATH_TO_SAVE = "./data/04_Merged"

vehicle_names = ["SEB880", "SEB882", "SEB883", "SEB885", "SEB888", "SEB889"]

eso_to_parse = [
    'ID', 'FunctionValue', 'domain', 'BeginTime', 'Label'
]

for vehicle in vehicle_names:
    print("%"*40)
    print(f"[Processing vehicle {vehicle}]\n")

    # Load data for vehicle 
    dfEso = pd.read_csv(os.path.join(PATH_TO_LOAD, vehicle + "_labeled_eso.csv"), parse_dates=['datetime'])
    dfMf4 = pd.read_csv(os.path.join(PATH_TO_LOAD, vehicle + "_labeled_mf4.csv"), parse_dates=['datetime'], low_memory=False)

    count_merged_labels = 0
    num_labels_mf4 = dfMf4.Label.notna().sum()

    # Inerate over Esotrace Labels to find matching MF4 signals 
    for index, row in tqdm(dfEso.iterrows()):

        # Find closest MF4 to EsoTrace Label
        time_diff = (dfMf4.datetime - row.datetime).abs()
        id_mf4 = time_diff.idxmin()     

        # Delta time has to be small to match a mf4 recording 
        if time_diff[id_mf4] > pd.Timedelta(minutes=1):
            continue    
        
        count_merged_labels += 1

        # If label not yet assigned
        if pd.isna(dfMf4.loc[id_mf4, 'Label']):  
            dfMf4.loc[id_mf4, eso_to_parse] = row[eso_to_parse]
        # If label already assigned, append new row
        else:
            new_row = dfMf4.loc[id_mf4].copy()
            new_row.loc[eso_to_parse] = row[eso_to_parse]
            dfMf4 = pd.concat([dfMf4, new_row.to_frame().transpose()], ignore_index=True)

    # Rename domain
    dfMf4['domain'] = dfMf4['Label'].str.split('/').str[0]
    
    # Session Filtering & Feature Engineering: 

    empty_sessions = 0
    zero_speed = 0
    sessions_to_remove = [] 

    session_unique = dfMf4.session.unique()

    for sess in tqdm(session_unique):
        
        df_sess = dfMf4[dfMf4.session == sess]
        ids_sess = df_sess.index

        # Remove session if zero interactions 
        if df_sess.Label.notna().sum() == 0:
            empty_sessions += 1
            sessions_to_remove.append(sess)
            continue

        # Remove session with zero speed
        if df_sess["KBI_speed"].sum() == 0:
            zero_speed += 1
            dfMf4 = dfMf4.drop(ids_sess)
            sessions_to_remove.append(sess)
            continue

        # Time in second from beginning of driving session
        dfMf4.loc[ids_sess, 'BeginTime'] = df_sess.datetime.min()
        dfMf4.loc[ids_sess, 'time_second'] = (df_sess.datetime - df_sess.datetime.min()).dt.total_seconds().astype(float)
        
        # From odometer distance driven
        dfMf4.loc[ids_sess, 'distance_driven'] = (df_sess.odometer - df_sess.odometer.min()).astype(float)

        # Normalize time_seconds
        ts_sess = dfMf4.loc[ids_sess, 'time_second']
        ts_sess_min = ts_sess.min()
        ts_sess_max = ts_sess.max()
        if ts_sess_max - ts_sess_min == 0:
            dfMf4.loc[ids_sess, "ts_normalized"] = 0.0
        else:
            dfMf4.loc[ids_sess, "ts_normalized"] = round((ts_sess - ts_sess_min) / (ts_sess_max - ts_sess_min),3)
    
    dfMf4 = dfMf4[~dfMf4.session.isin(sessions_to_remove)]

    print(f"Num meged labels from Eso: {count_merged_labels}/{len(dfEso)}")
    print(f"Num of empty sessions: {empty_sessions}/{len(session_unique)} ")
    print(f"Num of zero speed sessions: {zero_speed}/{len(session_unique)} ")

    # Filter SelectSource Labels
    num_labels = dfMf4.Label.notna().sum()
    for label in ["media/selectedSource/Radio", "media/selectedSource/Bluetooth", "media/selectedSource/Favorite"]:
        dfMf4 = helper_eso.clean_radio_label(dfMf4, label)
    print(f"Removed SelectSource: {num_labels - dfMf4.Label.notna().sum()}")
    print(f"Num of final labels: {dfMf4.Label.notna().sum()}")

    # Add weekday (0: Monday)
    def to_weekday(df_row):
        return df_row.weekday()
    dfMf4['weekday'] = dfMf4.datetime.apply(to_weekday)

    dfMf4 = dfMf4.reset_index()

    if not os.path.exists(os.path.join(PATH_TO_SAVE)):
        os.makedirs(os.path.join(PATH_TO_SAVE))
    dfMf4.to_csv(os.path.join(PATH_TO_SAVE, vehicle + "_merged.csv"), index=False)