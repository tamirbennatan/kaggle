################
# Script for Mercari Price Prediction Challenge Submission
# Author: Tamir Bennatan
################

# packages
library(Matrix)
library(dplyr)
library(tidyr)
library(reshape2)
library(stringr)
library(tidytext)
library(xgboost)

#setwd("~/Desktop/my-playground/Mercari-Price-Prediction/")

# Load training data
print("Loading training data...")
train <- read.csv("data/train.tsv", row.names = NULL, sep = "\t")
# ORIGNIAL.TRAIN = train

train = ORIGINAL.TRAIN

# function to apply preliminary cleaning tranformations
normalize.data <- function(data){
      # convert the types of factors to characters
      data$name <- as.character(data$name)
      data$brand_name <- as.character(data$brand_name)
      data$item_description <- as.character(data$item_description)
      data$category_name <- as.character(data$category_name) 
      # convert the empty string to naive NA values
      data <- data %>% 
            filter(price > 0) %>%
            mutate(name = ifelse(name == "", NA, name),
                   brand_name = ifelse(brand_name == "", NA, brand_name), 
                   item_description = ifelse(item_description == "", NA, item_description), 
                   category_name = ifelse(category_name == "", NA, category_name)) %>%
            replace_na(list(category_name = "novalue/novalue/novalue", 
                            brand_name = "novalue")) %>%
            mutate(name = str_replace_all(name, fixed(" "), "_"),
                   brand_name = str_replace_all(brand_name, fixed(" "), "_"), 
                   category_name = str_replace_all(category_name, "[^a-zA-Z0-9/]", "_")) %>%
            mutate(brand_name = str_replace_all(brand_name, "[^a-zA-Z0-9]", "_"))
      

      # lowercase brand and item description values
      data <- data %>%
            mutate(brand_name = str_to_lower(brand_name), 
                   item_description = str_to_lower(item_description))
      # fill item descriptions of `no item description` to NA
      data <- data %>%
            mutate(item_description = ifelse(item_description == "no description yet", NA, item_description))
      return (data)
}


# apply transform
print("Normalizing training types...")
train <- normalize.data(train)

# function for splitting categories
split.category <- function(data){
      # split the category into a hierarchy
      data <- data %>%
            mutate(category_name = str_to_lower(category_name)) %>%
            separate(col = category_name, 
                     into = c("high_category", "mid_category", "low_category"), 
                     sep = "/", 
                     remove = FALSE) %>%
            unite(mid_low_categories, mid_category, low_category, sep = "__", remove = FALSE)
      return(data)
}
print("Splitting training categories...")
train <- split.category(train)


# levels for high category factor
high.category.levels <- train %>%
      count(high_category) %>% 
      .$high_category 

# function for encoding categories
category.onehot <- function(data){
      # first, fill any categories that aren't one of the factors to NA
      data <- data %>%
            mutate(high_category = ifelse(high_category %in% high.category.levels, high_category, "novalue"))
      
      # Now, encode as a factor 
      data$high_category <- factor(data$high_category, levels = high.category.levels )
      
      return(data)
}

print("Encoding categories as factors...")
train <- category.onehot(train)

# encode low/mid categories
low.mid.category.levels = train %>%
      count(mid_low_categories, sort = TRUE) %>%
      top_n(99) %>%
      .$mid_low_categories %>%
      c(., "other")

# function for encoding mid categories
low.mid.categories.onehot <- function(data){
      # convert any categories not in levels to "other"
      data <- data %>%
            mutate(mid_low_categories = ifelse(mid_low_categories %in% low.mid.category.levels, mid_low_categories, "other"))
      
      # change to a factor
      data$mid_low_categories <- factor(data$mid_low_categories, levels = low.mid.category.levels )
      
      return(data)
}

train <- low.mid.categories.onehot(train)


# encode the brand name
brand.levels <- train %>%
      mutate(brand_name = case_when(
            brand_name == "air jordan" ~ "jordan",
            brand_name == "beats by dr. dre" ~ "beats", 
            TRUE ~ brand_name)
      ) %>%
      count(brand_name, sort = TRUE) %>%
      top_n(70) %>%
      .$brand_name %>%
      c(., "other")

brand.onehot <- function(data){
      # convert brands to "other" if they're not in the labels
      data <- data %>% 
            mutate(brand_name = case_when(
                  brand_name == "air Jordan" ~ "jordan",
                  brand_name == "beats by dr. dre" ~ "beats", 
                  TRUE ~ brand_name)
            ) %>%
            mutate(brand_name = ifelse(brand_name %in% brand.levels, brand_name, "other"))
      
      # add a column for whether or not the brand is NA
      data <- data %>%
            mutate(missing_brand = brand_name == "novalue")
      
      # convert to a factor 
      data$brand_name <- factor(data$brand_name, levels = brand.levels )
      
      
      return(data)
}

print("Encoding training brands as factors...")
train <- brand.onehot(train)


# categories of item condition
condition.levels = c(1,2,3,4,5,6)


condition.onehot <- function(data){
      # if the condition is not in one of the levels, cast it to NA
      data <- data %>%
            mutate(item_condition_id = ifelse(item_condition_id %in% condition.levels, item_condition_id, 6))
      
      # convert to factor
      data$item_condition_id <- factor(data$item_condition_id, levels = condition.levels )
      
      
      return(data)
}

print("Encoding training condition as factors...")
train <- condition.onehot(train)

shipping.levels = c(0,1)


shipping.binary.var <- function(data){
      # fill values with 0 
      data <- data %>%
            replace_na(list(shipping = 0))
      
      # convert to a factor
      data$shipping <- factor(data$shipping,levels = shipping.levels )
      
      return(data)
}

print("Encoding shipping as factor...")
train <- shipping.binary.var(train)


basic.text.features <- function(data){
      data <- data %>%
            mutate(description.missing = is.na(item_description), 
                   title_contains_brand = str_detect(name, as.character(brand_name)), 
                   description_contains_rm = str_detect(item_description, fixed("[rm]"))) %>%
            replace_na(list(title_contains_brand = FALSE, description_contains_rm = FALSE))
      
      return(data)
      
}

print("Extracting basic text features...")
train <- basic.text.features(train) 

set.seed(8)
tmp = train %>%
      mutate(price.bin = ntile(price, n = 10))  %>%
      group_by(price.bin) %>%
      sample_n(20000) %>%
      ungroup() %>%
      unnest_tokens(word, item_description) %>%
      anti_join(stop_words)

binned.averages = tmp %>%
      mutate(num.postings =  n_distinct(train_id)) %>%
      group_by(word) %>%
      summarize(num.postings = first(num.postings),
                posts.with.word = n_distinct(train_id)) %>%
      mutate(avg.posts.contain.word = posts.with.word/num.postings) %>%
      ungroup() %>%
      inner_join(
            tmp %>%
                  group_by(price.bin) %>%
                  mutate(num.posts.bin = n_distinct(train_id)) %>%
                  group_by(price.bin, word) %>%
                  summarize(num.posts.bin = first(num.posts.bin), 
                            posts.with.word.bin = n_distinct(train_id)) %>%
                  ungroup() %>%
                  mutate(avg.posts.contain.word.bin = posts.with.word.bin/num.posts.bin) 
            
      ) %>%
      mutate(inter.average.diff = avg.posts.contain.word.bin - avg.posts.contain.word) %>%
      arrange(desc(abs(inter.average.diff))) %>%
      filter(!is.na(word)) 

variance.words <- binned.averages %>%
      filter(posts.with.word > 2000) %>%
      group_by(price.bin) %>%
      top_n(25, wt = abs(inter.average.diff)) %>%
      ungroup() %>%
      select(word)  %>%
      unique() %>%
      rename(variance.words = word) %>%
      filter(variance.words != "rm") %>%
      .$variance.words


variance.words.onehot <- function(data){
      for (i in 1:length(variance.words)){
            newcol = paste("w_",str_replace_all(variance.words[[i]], "[^a-z]", ""), sep = "")
            data[newcol] = str_detect(data$item_description, variance.words[[i]])
            
            data[newcol][is.na(data[newcol])] <- FALSE
      }
      return(data)
}

print("Extracting high variance words from train...")
train <- variance.words.onehot(train)

print("Selecting training columns...")
filter.columns <- function(data)
      
      return(
            select(data, -name, -category_name, -mid_category, 
                   -low_category, -item_description)
      )


train <- filter.columns(train)


# get a sparse matrix
print("Encoding training data s sparse matrix...")
sparse_matrix = sparse.model.matrix(price~.-1, data = select(train, -train_id))

#store the response in a seperate vector
response = train$price
median.price = median(train$price)
# isolate columns for testing
testcols = select(train, -train_id, -price) %>% colnames() %>% c("test_id", .)

# rm(train)

print("Training...")
# best.model = xgboost(data = sparse_matrix,label = response, 
#                      max_depth = 22,
#                      min_child_weight = 15,
#                      eta = .05,
#                      gamma = .3,
#                      nrounds = 150, print_every_n = 10)

best.model = xgboost(data = sparse_matrix,label = response, 
                     max_depth = 12,
                     min_child_weight = 6,
                     gamma = .3,
                     nrounds = 30, 
                     print_every_n = 2)


print("------0-------")

print("Loading test data...")
test <- read.csv("data/test.tsv", row.names = NULL, sep = "\t")

ORIGINAL.TEST <- test

# normalize types
print("Normalizing types...")
test <- normalize.data(test)
# split categories
print("Splitting categories...")
test <- split.category(test)
# encode the brands
print("Encoding categories...")
test <- category.onehot(test)
test <- low.mid.categories.onehot(test)
print("Encoding brands...")
test <- brand.onehot(test)
# encode the condition
print("Encoding condition and shipping...")
test <- condition.onehot(test)
test <- shipping.binary.var(test)
print("Extracting basic text features...")
test <- basic.text.features(test) 
print("Extracting high variance words...")
test <- variance.words.onehot(test)

test <- select(test, testcols)

print("Converting test to sparse matrix...")
sparse_test <- sparse.model.matrix(test_id~.-1,test)

print("Predictions")
pred <- predict(best.model,sparse_test)

print("Writing results...")
output = select(test, test_id)
output$price <- pred

output <- output %>%
      mutate(price = ifelse(price < 0, median.price , price))


# write results
write.csv(output, "submission.csv", row.names = FALSE)
