from keras import models
from keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def import_data():
    df_bp = []
    df_bp = pd.read_csv('x_data/X_bp.csv')
    df_nup = []
    df_nup = pd.read_csv('x_data/X_nup.csv')
    df = df_bp.merge(df_nup, left_on='Unnamed: 0', right_on='Unnamed: 0')
    return df


data = import_data()
data.drop(['Unnamed: 0'], inplace=True, axis=1)
