import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import pandas as pd

europe = pd.read_csv('/Users/dhmako/Documents/Assignments in Managing Big Data/Third Assignment/europe.csv')

europe.head()
europe.columns

# Scale the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(europe.drop(['Country'], axis=1))

# Perform hierarchical clustering
agg_cluster = AgglomerativeClustering(n_clusters=3)
agg_cluster.fit(data_scaled)

# Add the cluster labels to the original dataframe
europe['cluster_label'] = agg_cluster.labels_

# Print the dataframe with the added cluster labels
print(europe)


link = linkage(data_scaled, method='ward')

# Create the dendrogram
plt.figure(figsize=(20, 10))
dendrogram(link, labels=europe['Country'].values)
plt.show()