# influence_similarity_comparison.py
# compares the average similarity between influencers and followers and the
# overall average similarity

import pandas as pd
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

influence_df = pd.read_csv('../Data/influence_data_with_similarity.csv')
artist_df = pd.read_csv('../Data/data_by_artist_scaled.csv')

parameter_columns = ['danceability', 'energy', 'valence', 'scaled_tempo',
                     'scaled_loudness', 'acousticness', 'instrumentalness',
                     'liveness', 'speechiness', 'scaled_duration_ms']

artist_df = pd.read_csv('../Data/data_by_artist_scaled.csv')
scaled_df = artist_df[parameter_columns]
overall_similarity_arr = np.power(
    (1 - distance.pdist(scaled_df, metric='cosine')), 1)
influence_similarity_arr = influence_df[['similarity']].to_numpy()


fig, ax = plt.subplots(figsize=(10, 4))

# Example data
y_pos = np.arange(0, 0.8, 0.4)
ax.barh(y_pos,
        [np.mean(overall_similarity_arr),
         np.nanmean(influence_similarity_arr)], height=0.3, align='center')
ax.set_xlim([0.8, 0.92])
ax.set_xticks(np.linspace(0.8, 0.92, 7))
ax.set_yticks(y_pos)
ax.set_yticklabels(['Overall Average Similarity',
                    'Average Similarity Between Influencer and Follower'])
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Similarity')

fig.savefig('../Graph/influencer_follower_similarity_compare.png',
            dpi=500, bbox_inches='tight')

plt.show()
