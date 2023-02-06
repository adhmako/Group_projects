# Install
install.packages("tm")  # for text mining
install.packages("SnowballC") # for text stemming
install.packages("syuzhet") # for sentiment analysis
install.packages("stringr") # replace all occurrences of a specific string or character within another string
install.packages("caTools") # install and load the caTools library
install.packages("e1071") # contains the Naive Bayes function
install.packages("caret") #for confusion matrix 

# Load
library(tm)# Library "tm" is used to remove stopwords (We use stopwords for english)
library(SnowballC)# Library "SnonwballC" is used to stem words
library(stringr)
library(syuzhet)
library(dplyr)
library(caTools)
library(e1071)
library(caret)

graphics.off() ; rm(list = ls(all = TRUE)) ; cat("\014");

# Pre-processing
# Create the functions to be used for the pre-processing of data

# Clean up any HTML element
cleantext <- function(text) {
  
  # Remove the HTML elements
  return(gsub("<.*?>", "", text))
}

# Remove words that contain symbols that
# are not letters or numbers
is_spec <- function(text){
  
  # Remove the symbols
  text = str_replace_all(text, "[^[:alnum:]]", " ")
  
  return (text)
}

# Convert to lowercase
to_lower <- function(text) {
  text <- tolower(text)
  return(text)
}


# Remove stopwords
rem_stopwords <- function(text) {
  # Convert the input to a Corpus
  text <- Corpus(VectorSource(text))
  
  # Remove the stopwords
  text <- tm_map(text, removeWords, stopwords("SMART"))
  
  # Convert the Corpus back to character vector
  text <- unlist(sapply(text, as.character))
  
  return(text)
}


# Stem words
stem_txt <- function(text) {
  # Convert the input to a Corpus
  text <- Corpus(VectorSource(text))
  
  # Stem the words
  text <- tm_map(text, stemDocument, language = "english")
  
  # Convert the Corpus back to character vector
  text <- unlist(sapply(text, as.character))
  
  return(text)
}

remove_numbers <- function(text) {
  # Use gsub() to replace all digits with an empty string
  text <- gsub("[0-9]", "", text)
  return(text)
}


# Read the text file 
data <- read.csv('IMDBDataset.csv', stringsAsFactors = FALSE)

data$sentiment = as.factor( data$sentiment)



# Preprocessing
data$review <- cleantext(data$review)

data$review <- is_spec(data$review)

data$review <- to_lower(data$review)

data$review <- rem_stopwords(data$review)

data$review <- stem_txt(data$review)

data$review <- remove_numbers(data$review)



corpus <- Corpus(VectorSource(data$review))


dtm <- DocumentTermMatrix(corpus)

inspect(dtm)

# Create a vector saving the number of those dtm terms that
# occur at least 1000 times
freqTerms <- findFreqTerms(dtm, 1000)

# Create a new data frame keeping all rows and only the frequent terms
dtm_freq <- dtm[ , freqTerms]

# Convert the dtm with frequent terms to matrix
dtm_freq <- as.matrix(dtm_freq)

# Create a vector containing the values of the class attribute
y <- data$sentiment

# Create a df containing the frequent term outcomes and the class attribute
newData = data.frame(dtm_freq, y)


# Build a term-document matrix
TextDoc_dtm <- DocumentTermMatrix(data)
dtm_m <- as.matrix(TextDoc_dtm)


# create a data frame for training and testing
#critics_df <- data.frame(review = data$review,
                        # sentiment = as.factor(data$sentiment))



# split the data into training and testing sets
set.seed(1234) # for reproducibility
#sentiment_cols <- select(critics_df, -sentiment)
#split <- sample.split(critics_df$sentiment, 0.8)
#split <- sample.split(critics_df$sentiment, SplitRatio = 0.8)
#train_data <- sentiment_cols[sample.split(critics_df$sentiment, 0.8), ]
#test_data <- sentiment_cols[!sample.split(critics_df$sentiment, 0.8), ]
split_index <- sample(1:nrow(data), round(nrow(data) * 0.8))
train_data <- data[split_index, "review"]
test_data <- data[-split_index, "review"]

# Create a vector containing the values of the class attribute
y_train <- data[split_index, "sentiment"]

# Create the model #"naiveBayes" function
NaiveBayesModel <- naiveBayes(train_data, y_train)

#"naiveBayes" function
#NaiveBayesModel <- naiveBayes(review ~ . , data = critics_df)


#"predict" function to predict the class of new data using the trained model
predictions <- predict(NaiveBayesModel, test_data)

#predict function on the control dataset using the trained model
#control_predictions <- predict(NaiveBayesModel, test_data)

#can create a variable that contains a Boolean value indicating whether 
#each observation was correctly classified.
#correct_predictions <- control_predictions == train_data$rewiew

#calculate the accuracy by taking the mean of the correct_predictions variable, 
#which will give the proportion of correctly classified observations.
accuracy <- mean(predictions == data[-split_index, "sentiment"])
print(accuracy)

#Alternative way for accuracy
confusionMatrix(predictions, data[-split_index, "sentiment"])



