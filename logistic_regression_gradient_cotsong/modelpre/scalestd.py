import pandas as pd
import numpy as np
import os

def calculate_and_save_params(df, columns):
    """Chỉ tính mean/std cho các cột feature (X)."""
    if not os.path.exists("meta"): os.makedirs("meta")
    data = df[columns].values
    mean = data.mean(axis=0)
    std = data.std(axis=0)
    std = np.where(std == 0, 1, std) 
    pd.DataFrame({'mean': mean, 'std': std}, index=columns).to_csv("meta/scale_params.csv")

def apply_standard_scale(df, columns):
    """Chỉ áp dụng scale cho X."""
    params = pd.read_csv("meta/scale_params.csv", index_col=0)
    mean = params.loc[columns, 'mean'].values
    std = params.loc[columns, 'std'].values
    return pd.DataFrame((df[columns].values - mean) / std, columns=columns)

