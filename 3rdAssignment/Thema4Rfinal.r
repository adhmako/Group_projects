graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");
setwd("/Users/andreasmoustakas/Downloads")

library("MASS")
library("klaR")
library(cluster)
library(dplyr)

movies <- read.csv("movies.csv")
ratings <- read.csv("ratings.csv")

cat <- movies[3:22]

#Find the appropriate k based on sum of squared errors
wss <- (nrow(cat)-1)*sum(apply(cat,2,var))
for (i in 2:200) wss[i] <- sum(kmeans(cat, centers=i)$withinss)
plot(1:200, wss, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Elbow Method For Optimal k")

#Appropriate k=100
kmodes_result <- kmodes(cat, modes = 100)

#New column with the clusterid (group number)
movies$clusterid <- kmodes_result$cluster

print(subset(movies,clusterid==13))



#Get movies that user198 has rated
user198_movies <- ratings %>% filter(userId == 198) %>% select(movieId, rating, timestamp)

#Get the clusters that the movies user198 has rated belong to 
user198_movies <- left_join(user198_movies, movies, by = "movieId") %>% select(movieId, rating, timestamp, clusterid)

#Group clusters by mean rating
cluster_mean_rating <- user198_movies %>% group_by(clusterid) %>% summarize(mean_rating = mean(rating))

#Drop clusters with mean rating less than 3.5
cluster_mean_rating <- cluster_mean_rating %>% filter(mean_rating >= 3.5)

#If there are no clusters with mean rating over 3.5 we can't recommend any movies!!
if (nrow(cluster_mean_rating) == 0) {
  print("Sorry, no recommendations for you! ~Do or do not there is no try.")
} else {
  print("See recommendations below")
}


#Get movies title and movieid from movies dataframe based on the same clusterid
movies_mean_ratings <- inner_join(movies, cluster_mean_rating, by = "clusterid")

#drop the categories columns
movies_mean_ratings <- movies_mean_ratings %>% select(-c(3:22))

#sort by descending rating order
movies_mean_ratings <- movies_mean_ratings %>% arrange(desc(mean_rating))

#top 2 rated movies of each cluster
top2_movies_per_cluster <- movies_mean_ratings %>% group_by(clusterid) %>% slice_head(n=2)

#If there are any movies user198 has already seen, drop them!
top2_movies_per_cluster <- top2_movies_per_cluster %>%
  filter(!movieId %in% user198_movies)


print("You may also like the following movies")

# show the title of the movies
top2_movies_per_cluster$title


########## Erwthma 2 ##########
graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

europe <- read.csv("europe.csv")

# Extract the values from the data frame
values <- europe[, -1]

# Calculate the distance matrix
dist_matrix <- dist(values)

# Perform hierarchical clustering on the distance matrix
hc <- hclust(dist_matrix)

# Increase the plot size and adjust the margins
par(cex = 0.7, mar = c(0, 4, 4, 2))

# Plot the dendrogram with the country names
plot(hc, labels = europe[, 1])