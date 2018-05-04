logistic.learning.curve <-  function(df.train, formula, seed = 1, classification.error = TRUE){
      
      # Define training and cross validation sets
      set.seed(seed)
      train <- sample(1:nrow(df.train), round(.8*nrow(df.train)), replace = FALSE)
      df.cv <- df.train[-train, ]
      
      # take subsets in the data to train classifier with, each with an additional
      # chunk of the data one fiftieth the size of the training set. Save errors. 
      
      errors <- list()
      chunk.size <- floor(nrow(df.train)/50)
      
      for( i in 1:50){
            
            # Take a subset of training set
            training.subset <- (df.train[train,])[1:(i*chunk.size),]
            
            # Fit a logistic regression classifier with inputed formula
            logistic.model <- glm(formula = formula, data = training.subset, family = binomial)
            
            # Predict responses for training subset and cross validation set
            training.subset.probs <- predict(logistic.model, newdata <- training.subset, type = "response")
            cv.probs <- predict(logistic.model, newdata = df.cv, type = "response")
            
            training.subset.predictions  <-  rep(0, length(training.subset.probs))
            cv.predictions <- rep(0, length(cv.probs))
            
            training.subset.predictions[training.subset.probs > .5] = 1
            cv.predictions[cv.probs > .5] = 1
            
            # store true responses
            Y.train <- (df.train[train, ])[1:(i*chunk.size), "Survived"]
            Y.cv <- df.cv$Survived
            
            # Store error. If parameter 'classification.error' is TRUE (default), use classificaton error. 
            # Otherwise use F1 score. 
            
            if(classification.error){
                  errors[[i]] <- list(chunk = i, 
                                      training.error = classification.error(
                                            pred = training.subset.predictions, Y = Y.train),
                                      cv.error = classification.error(
                                            pred = cv.predictions, Y = Y.cv)
                  )
            }
            else{
                  errors[[i]] <- list(chunk = i, 
                                      training.error = F1.score(
                                            pred = training.subset.predictions, Y = Y.train), 
                                      cv.error = F1.score(
                                            pred = cv.predictions, Y = Y.cv)
                  )
            }
            
      }
      
      return(errors)
}