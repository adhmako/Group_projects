library(gdata)
library(readxl)
graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");
data <- read.table("communities.data", fileEncoding="UTF-8", sep=",", header=FALSE)
d1 <- read_excel("varnames.xlsx")
names <- d1$names
colnames(data) <- names

idx <- data == "?"
is.na(data) <- idx

df<-na.omit(data)

y<- df[, "ViolentCrimesPerPop"]

X<- cbind( rep(1, 123), df[, "medIncome"], df[,"whitePerCap"], df[,"blackPerCap"], df[,"HispPerCap"], df[,"NumUnderPov"], df[,"PctUnemployed"], df[,"HousVacant"], df[,"MedRent"], df[,"NumStreet"] ) 



X <- as.matrix(X)
y <- as.matrix(y)

n_obs <- 123
learning_rate <- 0.13
batch_size <- 32
n_iter <- 10000
beta_hat_sgd <- matrix(0, nrow=n_iter, ncol=ncol(X))
beta_hat_gd <- matrix(0, nrow=n_iter, ncol=ncol(X))
beta_hat_norm_eq <- solve(t(X) %*% X) %*% t(X) %*% y

for(iter in seq_len(n_iter - 1)) {
  ## Keep it simple: sample a random subset of batch_size rows on every iteration
  row_idx <- sample(seq_len(n_obs), size=batch_size)
  residuals <- y[row_idx] - X[row_idx, ] %*% beta_hat_sgd[iter, ]
  gradient <- -(2 / batch_size) * t(X[row_idx, ]) %*% residuals
  beta_hat_sgd[iter + 1, ] <- beta_hat_sgd[iter, ] - learning_rate * gradient
}

beta_hat_sgd[n_iter, ]

plot(beta_hat_sgd[, c(2, 3)], type="l", col="red", xlab="beta_1", ylab="beta_2")
lines(beta_hat_sgd[, c(2, 3)], type="l", col="blue", lty=2)
points(x=beta_hat_norm_eq[2], y=beta_hat_norm_eq[3], col="black", pch=4, cex=3)

