# scale.py
# scale some parameter in the raw dataset to [0, 1]

import pandas as pd
from sklearn import preprocessing

parameter_columns = ['tempo', 'loudness', 'duration_ms']

df = pd.read_csv('../Data/data_by_artist.csv')
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(df[parameter_columns])
df[['scaled_' + parameter for parameter in parameter_columns]] = scaled
df.to_csv('../Data/data_by_artist_scaled.csv')

df = pd.read_csv('../Data/full_music_data.csv')
min_max_scaler = preprocessing.MinMaxScaler()
scaled = min_max_scaler.fit_transform(df[parameter_columns])
df[['scaled_' + parameter for parameter in parameter_columns]] = scaled
df.to_csv('../Data/full_music_data_scaled.csv')
