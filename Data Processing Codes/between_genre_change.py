# between_genre_change.py
# generate the stackplot showing the rise and fall between genre over decades

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ast

genres = ['Pop/Rock', 'R&B;', 'Country', 'Vocal', 'Jazz', 'Latin', 'Classical',
          'Folk', 'Reggae', 'Blues', 'Stage & Screen', 'International',
          'Electronic', 'Easy Listening', 'Religious', 'Comedy/Spoken',
          'New Age', 'Avant-Garde', "Children's", 'Unknown']


pagerank_df = pd.read_csv('../Data/average_pagerank.csv')
similarity_df = pd.read_csv('../Data/average_similarity.csv')
music_df = pd.read_csv('../Data/full_music_data_scaled.csv')

music_df['artist_names'] = music_df['artist_names'].apply(
    ast.literal_eval).map(lambda x: x[0])

music_df2 = music_df.merge(
    pagerank_df, left_on='artist_names', right_on='Artist')
music_df3 = music_df.merge(
    similarity_df, left_on='artist_names', right_on='influencer_name')


influence = np.empty((len(genres), 10))

for i in range(len(genres)):
    genre_music = music_df2[music_df2['genre'] == genres[i]]
    for j in range(10):
        target_music = genre_music[(genre_music['year'] > 1920 + j * 10) &
                                   (genre_music['year'] <= 1930 + j * 10)]
        influence[i, j] = np.sum(target_music['norm_updated_2'])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].margins(x=0)
axes[0].stackplot(np.linspace(1920, 2010, 10), influence, labels=genres)
axes[0].set_xticks(np.linspace(1920, 2010, 10))
axes[0].set_xticklabels([f"{i}s" for i in range(1920, 2020, 10)])
axes[0].set_xlabel('Year', fontsize=15)
axes[0].set_ylabel('Total Influence', fontsize=15)

axes[1].margins(x=0, y=0)
axes[1].stackplot(np.linspace(1920, 2010, 10), influence /
                  influence.sum(0), labels=genres)
axes[1].set_xticks(np.linspace(1920, 2010, 10),
                   [f"{i}s" for i in range(1920, 2020, 10)])
axes[1].set_xticklabels([f"{i}s" for i in range(1920, 2020, 10)])
axes[1].set_xlabel('Year', fontsize=15)
axes[1].set_ylabel('Relative Influence', fontsize=15)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.93, 0.5))
fig.suptitle(
    'Change In Influence of Genres Over Time,\nInfluence Measured By Total '
    'Number of Songs Weighted by Artist Betweenness and Smoothed Influence '
    'From Pagerank', fontsize=15)
fig.savefig('../Graph/between_genre_pagerank.png',
            dpi=500, bbox_inches='tight')
plt.show()


for i in range(len(genres)):
    genre_music = music_df3[music_df3['genre'] == genres[i]]
    for j in range(10):
        target_music = genre_music[(genre_music['year'] > 1920 + j * 10) &
                                   (genre_music['year'] <= 1930 + j * 10)]
        influence[i, j] = np.sum(target_music['similarity'])

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].margins(x=0)
axes[0].stackplot(np.linspace(1920, 2010, 10), influence, labels=genres)
axes[0].set_xticks(np.linspace(1920, 2010, 10))
axes[0].set_xticklabels([f"{i}s" for i in range(1920, 2020, 10)])
axes[0].set_xlabel('Year', fontsize=15)
axes[0].set_ylabel('Total Influence', fontsize=15)

axes[1].margins(x=0, y=0)
axes[1].stackplot(np.linspace(1920, 2010, 10), influence /
                  influence.sum(0), labels=genres)
axes[1].set_xticks(np.linspace(1920, 2010, 10),
                   [f"{i}s" for i in range(1920, 2020, 10)])
axes[1].set_xticklabels([f"{i}s" for i in range(1920, 2020, 10)])
axes[1].set_xlabel('Year', fontsize=15)
axes[1].set_ylabel('Relative Influence', fontsize=15)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center left', bbox_to_anchor=(0.93, 0.5))
fig.suptitle(
    'Change In Influence of Genres Over Time,\nInfluence Measured By Total '
    'Number of Songs Weighted by Similarity Score', fontsize=15)
fig.savefig('../Graph/between_genre_similarity.png',
            dpi=500, bbox_inches='tight')
plt.show()
