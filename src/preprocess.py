import pandas as pd
import numpy as np
import gc

def optimize_memory(df):
    """Downcast numeric types to save RAM by ~40%."""
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and not pd.api.types.is_datetime64_any_dtype(df[col]):
            c_min, c_max = df[col].min(), df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                else: df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.float32)
    return df

def run_full_preprocessing(path):
    cols = ['Severity', 'Start_Time', 'Start_Lat', 'Start_Lng', 'Distance(mi)',
            'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)',
            'Wind_Speed(mph)', 'Precipitation(in)', 'Weather_Condition',
            'Sunrise_Sunset', 'Amenity', 'Bump', 'Crossing', 'Junction', 'Traffic_Signal']
    
    df = pd.read_csv(path, usecols=cols)
    df.dropna(subset=['Severity'], inplace=True)
    df.drop_duplicates(subset=['Start_Time', 'Start_Lat', 'Start_Lng'], inplace=True)
    
    # Time Engineering
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Hour'] = df['Start_Time'].dt.hour
    df['Weekday'] = df['Start_Time'].dt.weekday
    df['Month'] = df['Start_Time'].dt.month
    df.drop(columns=['Start_Time'], inplace=True)
    
    # Impute & Simplify
    df['Weather_Condition'] = df['Weather_Condition'].fillna('Unknown')
    top_w = df['Weather_Condition'].value_counts().nlargest(15).index
    df['Weather_Condition'] = df['Weather_Condition'].apply(lambda x: x if x in top_w else 'Other')
    
    df = optimize_memory(df)
    gc.collect()
    return df