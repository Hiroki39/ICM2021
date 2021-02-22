# similarity_between_genre.py
# calculate the average similarity between/within all genres

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

music_df = pd.read_csv('../Data/full_music_data_scaled.csv')
artist_df = pd.read_csv('../Data/data_by_artist_scaled.csv')
parameter_columns = ['danceability', 'energy', 'valence', 'scaled_tempo',
                     'scaled_loudness', 'acousticness', 'instrumentalness',
                     'liveness', 'speechiness', 'scaled_duration_ms']


def calc_similarity(df, genre1, genre2):
    scaled_df = df[parameter_columns]
    scaled_df['genre'] = df['genre']
    genre1_df = scaled_df[scaled_df['genre'] == genre1]
    genre2_df = scaled_df[scaled_df['genre'] == genre2]
    if genre1 != genre2:
        dist_arr = distance.cdist(genre1_df[parameter_columns],
                                  genre2_df[parameter_columns], metric='cosine')
    else:
        dist_arr = distance.pdist(genre1_df[parameter_columns], metric='cosine')
    return np.mean(np.square(1 - dist_arr))


genres = artist_df['genre'].where(
    artist_df['genre'] != 'Unknown').dropna().unique()

genre_similarity = np.empty((len(genres), len(genres)))

for i in range(len(genres)):
    for j in range(i, len(genres)):
        result = calc_similarity(artist_df, genres[i], genres[j])
        genre_similarity[i, j] = result
        genre_similarity[j, i] = result

fig, ax = plt.subplots(figsize=(7, 7))

# Plot the data:
img = ax.imshow(genre_similarity)

ax.xaxis.tick_top()
ax.set_xticks(np.linspace(0, 18, 19))
ax.set_xticklabels(genres, rotation=45, ha="left")
ax.set_yticks(np.linspace(0, 18, 19))
ax.set_yticklabels(genres)
ax.set_title('Average Similarity Between/Within Genres', fontsize=15)
fig.colorbar(img)
plt.savefig('../Graph/genre_similarity.png', dpi=500, bbox_inches='tight')

similarity = pd.DataFrame(data=genre_similarity, columns=genres)
similarity.to_csv(
    '../Data/genre_similarity.csv', index=False)
