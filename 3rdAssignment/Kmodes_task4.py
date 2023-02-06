import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

movies = pd.read_csv('/Users/dhmako/Documents/Assignments in Managing Big Data/Third Assignment/movies.csv')

from kmodes.kmodes import KModes

cat = movies.drop(['movieId','title'],axis=1)

clusters = 200
sse = []

############ VERY LONG TO EXECUTE ############

for k in range(2, clusters+1):
    print("\t>>> Executing Kmode with k=", k, "clusters.....", end="")
    kmode = KModes(n_clusters=k, max_iter=500, n_init=10)
    # .fit() method actually does the kmeans clustering of wineData
    kmode.fit(cat)
    # .inertia_ contains the value of the objective function which is the sum of squared distances
    # across all clusters (SSE). Add the sum of squared distances into the list, so that
    # we can later diplay it
    sse.append(kmode.cost_)
    print("Done.")

print('\n K-finding iterations done...')

kmode.cluster_centroids_

print('\n Display plot to apply the Elbow method')

plt.plot(range(2,clusters+1),sse)
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')
plt.show()

# k = 100 looks like a good value...

print("Executing KModes with k=100 (100 clusters) to get FINAL clustering of the data...", end="")
kmode = KModes(n_clusters=100, max_iter=500, n_init=10)
clusters = kmode.fit(cat)
print("\nDone")

clusters.labels_

ratings = pd.read_csv('/Users/dhmako/Documents/Assignments in Managing Big Data/Third Assignment/ratings.csv')


ratings[ratings['userId']==135]['movieId']

len(clusters.cluster_centroids_)

movies['ClusterID'] = clusters.labels_

movies.head(5)

movies[movies['ClusterID']==5]['title']
ratings['ClusterID'] = clusters.labels_

 # Assign column ClusterID to ratings dataframe where ratings.movieid = movies.movieid

movies.reset_index()
movies.set_index('ClusterID')
merged_df = ratings.merge(movies[['movieId','ClusterID']], on='movieId', how='left')

merged_df.isna().sum(0)

    # Mean rating for every movie
ratings.groupby('movieId')['rating'].mean()

    # Mean rating for every cluster that has a movie user=198 rated
user198 = merged_df[merged_df['userId']==198].groupby('ClusterID')['rating'].mean().sort_values(ascending=False)

user198 = pd.DataFrame(user198)
user198.reset_index()

    # Drop clusters with less than 3.5 mean rating
mask = user198['rating']<3.5

user198 = user198.drop(user198[mask].index)
user198.head()
user198 = user198.reset_index()

if len(user198)==0:
    print("Sorry, no recommendations for you! The wise speak only of what they know")
else:
    print("See recommendations below...")

# Extract the clusters with average rating >= 3.5 for user 198
clusters_high_rating = user198[user198['rating'] >= 3.5]['ClusterID']

# Create an empty list to store the movie recommendations
movie_recs = []

# Iterate over clusters with average rating >= 3.5 for user 198
for cluster in clusters_high_rating:
    # Extract the movies from the merged_df dataframe that belong to the current cluster and that user 198 has not seen
    cluster_movies = merged_df[(merged_df['ClusterID'] == cluster) & (merged_df['userId'] != 198)]
    # Sort the movies by rating in descending order
    cluster_movies = cluster_movies.sort_values(by='rating', ascending=False)
    # Get the top 2 movies
    top_2_movies = cluster_movies.head(2)
    # Extract the movie titles
    top_2_titles = top_2_movies.merge(movies, on='movieId')['title']
    # Append the titles to the movie_recs list
    movie_recs.extend(top_2_titles)

# Print the message and the movie recommendations
print("You may also like the following movies:")
for title in movie_recs:
    print(title)

print("If you are in doubt, follow your nose...")