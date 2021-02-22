# similatity_influence_relation.py
# explore the relationship between the similarity between/within genres and
# number of interactions between/within genres

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

similarity_df = pd.read_csv('../Data/genre_similarity.csv')
influence_df = pd.read_csv('../Data/influence_data.csv')

labels = []
influence_interactions = []
similarities = []

for i in range(similarity_df.shape[0]):
    for j in range(i, similarity_df.shape[0]):
        curr_influence_interaction = len(
            influence_df[(influence_df['influencer_main_genre'] ==
                          similarity_df.columns[i]) &
                         (influence_df['follower_main_genre'] ==
                          similarity_df.columns[j])])
        if i != j:
            curr_influence_interaction += len(
                influence_df[(influence_df['influencer_main_genre'] ==
                              similarity_df.columns[i]) &
                             (influence_df['follower_main_genre'] ==
                              similarity_df.columns[j])])
        if curr_influence_interaction != 0:
            influence_interactions.append(curr_influence_interaction)
            labels.append((similarity_df.columns[i], similarity_df.columns[j]))
            similarities.append(similarity_df.iloc[i][j])

fig, ax = plt.subplots(1, 1)
ax.scatter(np.log(influence_interactions), similarities, s=10)

regr = linear_model.LinearRegression()
X = np.transpose([np.log(influence_interactions)])
Y = similarities
regr.fit(X, Y)  # use fit method
r_sqr = regr.score(X, Y)
betas = regr.coef_  # m
y_int = regr.intercept_  # b

y_hat = betas[0] * np.log(influence_interactions) + y_int
plt.plot(np.log(influence_interactions), y_hat, color='orange')  # y_hat, income
plt.xlabel('Log (number of interactions between genre)')
plt.ylabel('Average Similarity')
plt.savefig('../Graph/similarity_influence_relation.png',
            dpi=500, bbox_inches='tight')
plt.show()
plt.title('Relationship between number of interactions and similarity\n'
          f'between and within genre(R ^ 2: {r_sqr: .3f})')
