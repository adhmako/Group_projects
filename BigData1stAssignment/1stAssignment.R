##########  Erwthma 3a ###########
library(MASS)


graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

data("Cars93")

summary(Cars93)

colSums(is.na(Cars93)) #where are the missing values if any?
df<- na.omit(Cars93) #remove the missing values
cars.pca <- princomp(df[,c(4:8,12:15,17:25)], cor = FALSE,scores = TRUE)

summary(cars.pca)

plot(cars.pca)
biplot(cars.pca)

##########  Erwthma 4a ###########
graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

euclideanDistance<- function(x, y) sqrt(sum((x - y)^2))

#a
x<-c(1,2,3,4,5,6)
y<-c(1,2,3,4,5,6)
print("Euclidean distance between x and y is: ")
euclideanDistance(x,y)

#b
x=c(-0.5, 1, 7.3, 7, 9.4, -8.2, 9, -6, -6.3)
y=c(0.5, -1, -7.3, -7, -9.4, 8.2, -9, 6, 6.3)
print("Euclidean distance between x and y is: ")
euclideanDistance(x,y)

#c
x=c(-0.5, 1, 7.3, 7, 9.4, -8.2)
y=c(1.25, 9.02, -7.3, -7, 5, 1.3)
print("Euclidean distance between x and y is: ")
euclideanDistance(x,y)

#d
x=c(0, 0, 0.2)
y=c(0.2, 0.2, 0)
print("Euclidean distance between x and y is: ")
euclideanDistance(x,y)

##########  Erwthma 4b ###########
x=c(25000,14,7)
y=c(42000,17,9)
z=c(55000,22,5)
n=c(27000,13,11)
m=c(58000,21,13) ##target of comparison

eucldistvector=c(euclideanDistance(x,m),euclideanDistance(y,m),euclideanDistance(z,m),euclideanDistance(n,m))
print("The user that is most similiar to user with id=5 is")
which.min(eucldistvector)


##########  Erwthma 5 ###########
install.packages(lsa)

graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

library(SnowballC) #essential for lsa
library(lsa) #for cosine similiriaty function

cossim <- function(A,B) { (sum(A*B))/sqrt((sum(A^2))*(sum(B^2))) }

#a
x=c(9.32, -8.3, 0.2)
y=c(-5.3, 8.2, 7)
cossim(x,y)

#b
x=c(6.5, 1.3, 0.3, 16, 2.4, -5.2, 2, -6, -6.3)
y=c(0.5, -1, -7.3, -7, -9.4, 8.2, -9, 6, 6.3)
cossim(x,y)

#c
x=c(-0.5, 1, 7.3, 7, 9.4, -8.2)
y=c(1.25, 9.02, -7.3, -7, 15, 12.3)
cossim(x,y)

#d
x=c(2, 8, 5.2)
y=c(2, 8, 5.2)
cossim(x,y)


##########  Erwthma 6 ###########
graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

#I interpret distance between two character vectors as 1 when values between two vectors of the same column are different
nominalDistance <- function(x,y) sum(a==FALSE)

#a
x <-c("Green","Potato","Ford")
y <-c("Tyrian purple","Pasta","Opel")
a<-x==y
nominalDistance(x,y)

#b
x <-c("Eagle", "Ronaldo","Real madrid" , "Prussian blue", "Michael Bay")
y <-c("Eagle", "Ronaldo", "Real madrid", "Prussian blue", "Michael Bay")
a<-x==y
nominalDistance(x,y)

#c
x <-c("Werner Herzog", "Aquirre, the wrath of God", "Audi", "Spanish red")
y <-c("Martin Scorsese", "Taxi driver", "Toyota", "Spanish red")
a<-x==y
nominalDistance(x,y)



