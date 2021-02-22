# distribution.py
# generate distance distribution plot under different similarity formula

import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

parameter_columns = ['danceability', 'energy', 'valence', 'scaled_tempo',
                     'scaled_loudness', 'acousticness', 'instrumentalness',
                     'liveness', 'speechiness', 'scaled_duration_ms']

artist_df = pd.read_csv('../Data/data_by_artist_scaled.csv')
scaled_df = artist_df[parameter_columns]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
dist_arr = np.power((1 - distance.pdist(scaled_df, metric='cosine')), 1)
axes[0].margins(x=0)
axes[0].hist(dist_arr, range=[0, 1], edgecolor='black', bins=20)
axes[0].set_xticks(np.linspace(0, 1, 11))
axes[0].set_xlabel('1 - sim(x, y)')
axes[0].set_ylabel('Frequency')

dist_arr = np.power((1 - distance.pdist(scaled_df, metric='cosine')), 2)
axes[1].margins(x=0)
axes[1].hist(dist_arr, range=[0, 1], edgecolor='black', bins=20)
axes[1].set_xticks(np.linspace(0, 1, 11))
axes[1].set_xlabel('(1 - sim(x, y))^2')
axes[1].set_ylabel('Frequency')

fig.savefig('../Graph/distance_distribution.png',
            dpi=500, bbox_inches='tight')
plt.show()
