# influencer_follower_similarity.py
# calculate the similarity between every pair of follower and influencer

import pandas as pd
import numpy as np
from scipy.spatial import distance


influence_df = pd.read_csv('../Data/influence_data.csv')
artist_df = pd.read_csv('../Data/data_by_artist_scaled.csv')

parameter_columns = ['danceability', 'energy', 'valence', 'scaled_tempo',
                     'scaled_loudness', 'acousticness', 'instrumentalness',
                     'liveness', 'speechiness', 'scaled_duration_ms']


def calc_similarity(artist1, artist2):
    row1 = artist_df[artist_df['artist_name'] == artist1]
    row2 = artist_df[artist_df['artist_name'] == artist2]
    if row1.empty or row2.empty:
        return np.nan
    return np.square(1 - distance.cosine(
        row1[parameter_columns].iloc[0, :],
        row2[parameter_columns].iloc[0, :]))


influence_df['similarity'] = influence_df.apply(
    lambda row: calc_similarity(row['influencer_name'],
                                row['follower_name']), axis=1)

influence_df.to_csv('../Data/influence_data_with_similarity.csv')
