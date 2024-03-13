"""
Script Name:    helper_mf4.py

Description:    Helper functions MF4.
                
Author:         Jeremy Wanner (EGD1)
"""
import pandas as pd
import numpy as np
import utils.helper_labeling as helper_labeling

def convert_byte_to_float(value):
    # if value == "b'Init'":
    #     return float(-1)
    if value in ['nan', "b'nan'", "b'Fehler'", "b'Init'"]:
        return float('nan') 
    if isinstance(value, str) and value.startswith("b'") and value.endswith("'"):
        numeric_part = value[2:-1]
        return float(numeric_part)       
    else:
        return float(value)
    
def process_context_variables(context):
    """ Process each variable with different logics """
    if context.name in ["speed", "latitude", "longitude", "temperature", "soc"]:
        """ Average over window (type -> float) """
        context = context.apply(convert_byte_to_float)
        return round(context.mean(),3)
    elif context.name == "break_pedal1":
        """ Keep the last value """
        context = round(context.tail(1), 3).item() if len(context) > 0 else 0 
        return context
    elif context.name in ["light_sensor1", "light_sensor2", "rain_sensor"]:
        """ Keep last value (convert byte) """
        context = context.apply(convert_byte_to_float)
        return context.tail(1).item() if len(context) > 0 else 0 
    elif context.name in ["drive_mode", "street_category"]:
        """ Keep the most frequent value in the time window"""
        context_frequency = context.value_counts()
        return context_frequency.index[0] if len(context_frequency.index) > 0 else 0
    elif context.name in ["seatbelt_driver", "seatbelt_codriver"]:
        """ If seatbelt is detected assume there is a passenger """
        return 1 if np.any(context == 1.0) else 0
    
def get_external_temperature(df):
        """ Only for car "SEB889" """
        t_out = df.temperature_outside
        t_out = t_out.replace("b'nicht_verfuegbar'",0)
        return t_out.apply(convert_byte_to_float)
    
def label_drive_mode(df):
    for i, sess in enumerate(df.session.unique()):
        
        df_sess = df[df.session == sess]
        
        # Check where the drive mode changed
        #df_sess["CHA_MO_drive_mode"] = df_sess["CHA_MO_drive_mode"].astype(int)
        change_indices = df_sess.index[df_sess["CHA_ESP_drive_mode"].diff() != 0]
        
        if len(change_indices) <= 1:  # Skip if there's no interaction
            continue
        else:
            change_indices = change_indices[1:]
            
            # Keep only the last if many consecutive interactions
            last_change_index = helper_labeling.remove_consecutive_labels(df_sess, change_indices)
            
            last_change_index = [index for index in last_change_index if pd.notnull(df.loc[index, 'CHA_ESP_drive_mode'])] # filter the nan

            df.loc[last_change_index, "Label"] = "car/driveMode/" + df.loc[last_change_index, "CHA_ESP_drive_mode"].astype(str)

    return df
