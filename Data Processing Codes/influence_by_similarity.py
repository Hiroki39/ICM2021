# influence_by_similarity.py
# generate the similarity per song data for artists in order to distribute
# similarity to songs

import pandas as pd

influence_df = pd.read_csv(
    '../Data/influence_data_with_similarity.csv')
artist_df = pd.read_csv('../Data/data_by_artist_scaled.csv')
influence_dict = {}

for index, row in influence_df.iterrows():
    influencer = row[['influencer_name']].values[0]
    if influencer in influence_dict:
        influence_dict[influencer] += row[['similarity']].values[0]
    else:
        influence_dict[influencer] = row[['similarity']].values[0]

influence_index1 = influence_df.groupby(
    ['influencer_name']).similarity.sum().reset_index()
influence_index2 = influence_index1.merge(
    artist_df[['artist_name', 'count']],
    left_on='influencer_name', right_on='artist_name')
influence_index2['similarity'] = influence_index2['similarity'] / \
    influence_index2['count']
influence_index2 = influence_index2.drop(columns=['artist_name', 'count'])

influence_index1 = influence_index1.sort_values(
    by='similarity', ascending=False)

influence_index2 = influence_index2.sort_values(
    by='similarity', ascending=False)

influence_index1.to_csv('../Data/total_similarity.csv')
influence_index2.to_csv('../Data/average_similarity.csv')

updated2_df = pd.read_csv('../Data/updated_2.csv')

influence_index3 = updated2_df.merge(
    artist_df[['artist_name', 'count']],
    left_on='Artist', right_on='artist_name')
influence_index3['norm_updated_2'] = influence_index3['norm_updated_2'] / \
    influence_index3['count']

influence_index3 = influence_index3[['Artist', 'norm_updated_2']]
influence_index3.to_csv('../Data/average_pagerank.csv')
