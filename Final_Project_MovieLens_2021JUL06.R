---
  title: "Final_Project_MovieLens"
author: "Alfredo Nasiff"
date: "07/06/2021"
output: pdf_document
---
  
```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

\newpage

# Introduction

The Movie Lens project is part of the HarvardX: PH125.9x Data Science: Capstone
course. It closely follows the Data Science textbook by Rafael Irizarry 
particularly; Chapter 33.7 Recommendation systems, and Chapter 33.9 
Regularization, both sub chapters of the "Large Dataset" chapter.

The aim of the project is to develop and train a recommendation machine learning
algorithm to predict a rating given by a users to a movie in the dataset. The 
Residual Mean Square Error (RMSE) will be used to evaluate the accuracy of 
the algorithm.

This report will present an overview of the data, analysis, results and a 
conclusion.

## Dataset

The data is downloaded as per instruction from the MovieLens 10M dataset. 
The column "genres" was separated into as many columns as the maximum number of 
genres found in the classification of one movie. The "timestamp" column was 
mutated to "year" and two other columns were created, one with the year of the 
premier of the film, "premier_date", by extracting the information from the 
field "title" and the second one, "rating_age", by calculating the longevity of 
the rating by subtracting the "premier_date" to the "year". All this changes 
with the purpose of creating fields with additional predictive power.

```{r, echo=TRUE, message=FALSE}
##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(anytime)) install.packages("anytime", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(anytime)
library(recosystem)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "year"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

#determining the maximum number of genres per row
number_genres <- max(str_count(movielens$genres,"[|]")) + 1

movielens <- movielens %>% 
  mutate(
    year = lubridate::year(anytime(year)),
    premier_date = as.numeric(str_extract(str_extract(title, "[(]\\d{4}[)]"), "\\d{4}")),
    rating_age = year - premier_date
    
  ) %>% 
  separate(
    col = genres,
    into = paste0("genre",c(1:(number_genres))),
    sep = "[|]"
  )

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
```

The edx set is used for training and testing, and the validation set is used for
final validation to simulate the new data.

Here, we split the edx set in 2 parts: the training set and the test set.

The model building is done in the training set, and the test set is used to test
the model. When the model is complete, we use the validation set to calculate 
the final RMSE. We use the same procedure used to create edx and validation 
sets.

The training set will be 90% of edx data and the test set will be the remaining 
10%.

```{r}
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.1, list = FALSE)
train_set <- edx[-test_index,]
temp <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Add rows removed from test set back into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

rm(test_index, temp, removed)
```

An introductory review of the dataset is performed in order to familiarize 
ourselves.

Edx dataset contains rows corresponding to users ratings of a movie. The set 
contains the variables; "userId", "movieId", "rating", "year", "title", 
"genre1", "genre2", "genre3", "genre4", "genre5", "genre6", "genre7", "genre8", 
"premier_date" and "rating_age".

```{r, echo=TRUE}
# Summarise Data
head(edx, 5)
nrow(edx)
```

Summarising the dataset reveals a well formatted set with no missing values. 
Movies are rated between 0.5 and 5.0, with 9000055 rows in total. The minimum 
value of the "rating_age" is -2, meaning the year of the rating was lower than 
the premier date of the film, which is seemingly illogical. The cases found with 
"rating_age" values below 0 were 175 and were taken into the analysis, 
considering the meaning of the value is basically the same as if it had been 
"0".

```{r, echo=TRUE}
summary(edx)
```

The dataset contains 10,677 unique movies, 69,878 unique users, and 21 unique
genres:

```{r, echo=TRUE}
# Movies, Users and Genres in Database
edx %>% summarise(
  uniq_movies = n_distinct(movieId),
  uniq_users = n_distinct(userId),
  uniq_genres = length(unique(c(unique(genre1),unique(genre2),unique(genre2),
                            unique(genre2),unique(genre2),unique(genre2),
                            unique(genre2),unique(genre2))))
)
```

Note that if we are predicting the rating for movie i by user u, it happens that
any other rating made by u or any other rate received by i could be manifesting 
some pattern and therefore can be used as predictors, but neither the number of 
movies i rated by u, nor the users rating a single movie are constants, 
resulting in that in this machine learning challenge each outcome Y has a 
different set of predictors.

Lets look at some of the general properties of the data to better understand 
the challenges.

The first thing noticed is that some movies get rated more than others. Below 
is the distribution. Our second observation is that some users are more active 
than others at rating movies. Also, the graph "Year" below show that in the 
firsts years, users were more condescending rating the films. Regarding the date
of the premier, older films get better rated. Concerning the longevity of 
the rating, it is observed a variability of 0.6 stars in the rating and last, 
the average rating among the distinct genres move from just below 3.3 stars 
for Horror films to above 4 stars for Film-Noir movies.

```{r, echo=TRUE}
# Ratings Users and Number of Ratings

edx %>% count(movieId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins=100) +
  scale_x_log10() +
  xlab("# Ratings") +
  ylab("MovieId Count") +
  ggtitle("Movies") +
  theme(plot.title = element_text(hjust = 0.5))

edx %>% count(userId) %>%
  ggplot(aes(n)) +
  geom_histogram(color = "black", bins=60) +
  scale_x_log10() +
  xlab("# Ratings") +
  ylab("UserId Count") +
  ggtitle("Users") +
  theme(plot.title = element_text(hjust = 0.5))

edx %>% group_by(year) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Year") +
  theme(plot.title = element_text(hjust = 0.5))

edx %>% filter(rating_age >= 0) %>% #filtering out 175 cases where rating_age < 0
  group_by(rating_age) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(rating_age, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Rating Age") +
  theme(plot.title = element_text(hjust = 0.5))

edx %>% group_by(premier_date) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(premier_date, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Premier Date") +
  theme(plot.title = element_text(hjust = 0.5))

gnr1 <- edx %>% group_by(genre1) %>%summarize(rating1 = sum(rating),n1 = n())
gnr2 <- edx %>% group_by(genre2) %>%summarize(rating2 = sum(rating),n2 = n())
gnr3 <- edx %>% group_by(genre3) %>%summarize(rating3 = sum(rating),n3 = n())
gnr4 <- edx %>% group_by(genre4) %>%summarize(rating4 = sum(rating),n4 = n())
gnr5 <- edx %>% group_by(genre5) %>%summarize(rating5 = sum(rating),n5 = n())
gnr6 <- edx %>% group_by(genre6) %>%summarize(rating6 = sum(rating),n6 = n())
gnr7 <- edx %>% group_by(genre7) %>%summarize(rating7 = sum(rating),n7 = n())
gnr8 <- edx %>% group_by(genre8) %>%summarize(rating8 = sum(rating),n8 = n())

gnrs_list <- list(gnr1,gnr2,gnr3,gnr4,gnr5,gnr6,gnr7,gnr8)
gnr_total <- rbindlist(gnrs_list,use.names = F)
gnr_summ <- gnr_total %>% na.omit() %>% group_by(genre1) %>% 
  summarize(rating_genre = sum(rating1) / sum(n1))

gnr_summ %>% 
  ggplot(aes(genre1,rating_genre)) +
  geom_point() +
  geom_smooth() +
  ggtitle("Genre") +
  theme(plot.title = element_text(hjust = 0.5),
        axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

```
The challenge ahead is to identify those patterns and use them to predict the 
rating of a movie.

\pagebreak

# Analysis and Results
The Residual Mean Square Error (RMSE) is the error function that will be used 
to measure accuracy and quantify the typical error we make when predicting the 
movie rating. RMSE defined;

$$ RMSE = \sqrt{\frac{1}{N}\displaystyle\sum_{u,i} (\hat{y}_{u,i}-y_{u,i})^{2}} $$

where; N is the number of users, movie ratings, and the sum incorporating the 
total combinations.

For our model we are going to define function RMSE:

```{r}
RMSE <- function(true_ratings, predicted_ratings){
    sqrt(mean((true_ratings - predicted_ratings)^2))
  }
```

## Simple Prediction Based on Mean Rating
We can use a model based approach to answer this. A model that assumes the same 
rating for all movies and users with all the differences explained by random 
variation would look like this:

$$ Y_{u,i} = \mu + \epsilon_{u,i} $$
where $Y_{u,i}$ is the prediction, with $\epsilon_{u,i}$ the independent error 
sampled from the same distribution centered at 0 and $\mu$, the expected "true" 
rating for all movies. We know that the estimate that minimizes the RMSE is the 
least squares estimate of $\mu$ and, in this case, is the average of all 
ratings:

```{r, echo=TRUE}
## Simple Prediction based on Mean Rating
mu <- mean(train_set$rating)
mu
```

If we predict all unknown ratings with $\mu$ we obtain the following RMSE:

```{r}
rmse_naive <- RMSE(test_set$rating, mu)
rmse_naive
## Save Results in Data Frame
rmse_results = tibble(method = "Mean Predictor", RMSE = rmse_naive)
rmse_results %>% knitr::kable()
```

Investigating the dataset allows for more advanced analysis and rating 
predictions with smaller error.

## Movie Effects Model
As we all know, some movies are rated higher than others. This intuition, that 
different movies are rated differently, was confirmed by the graph showed in the
Dataset chapter. We can add to our previous model the term $b_i$ to represent 
average ranking for movie $i$:

$$ Y_{u,i} = \mu + b_i + \epsilon_{u,i} $$
where $b_i$ is the bias for each movie $i$.

The reason for using this approach is due to the fact that, as the data set is 
so large the use of the lm() function would be very slow in computing the $b_i$
of each movie, which otherwise can be calculated like $Y_{u,i} - \hat\mu$:
```{r, echo=TRUE}
# Simple model taking into account the movie effects, b_i
mu <- mean(train_set$rating)

movie_avgs <- train_set %>%
  group_by(movieId) %>%
  summarise(b_i = mean(rating - mu))
```

We can see that these estimates vary substantially:

```{r}
qplot(b_i, data = movie_avgs, bins = 45, color = I("black"))
```

Remember that $\hat{\mu}$=3.5, so a ${b_i}$=1.5 implies a perfect five star 
rating. Lets see how much our prediction improves once we use 
$\hat{Y}_{u,i} = \hat{\mu} + \hat{b_i}$:

```{r}
predicted_ratings <- mu + test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  pull(b_i)
rmse_model_movie_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_movie_effects
# Add Results in Data Frame
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie Effects Model",
                          RMSE = rmse_model_movie_effects))
rmse_results %>% knitr::kable()
```

The Movie Effect Model; predicting the movie rating with both bias, $b_i$, and 
mean, $\mu$ gives an improved prediction with a lower RMSE value.

## Movie and User Effects Model

The next step is to incorporate the individual User Effects, $b_u$, in to the 
model. Acknowledging each user inherent bias to mark all films higher or lower. 
Lets compute the average rating for user $u$ for those that have rated over 100 
movies:

```{r}
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 45, color = "black")
```

Notice that there is substantial variability across users as well: some users 
are very cranky and others love every movie. This implies that a further 
improvement to our model may be:

$$ Y_{u,i} = \mu + b_i + b_u + \epsilon_{u,i} $$

where $b_u$ is the bias for each user $u$. Now if a cranky user (negative $b_u$)
rates a great movie (positive $b_i$), the effects counter each other and we may 
be able to correctly predict that this user gave this great movie a 3 rather 
than a 5. We will compute an approximation by computing $\mu$ and $b_i$ and 
estimating $b_u$ as the average of ${Y}_{u,i} - \hat{\mu} - \hat{b_i}$:

```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
user_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = mean(rating - mu - b_i))
```

We can now construct predictors and see how much the RMSE improves:

```{r}
predicted_ratings_i_u <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
rmse_model_user_effects <- RMSE(test_set$rating, predicted_ratings_i_u)
rmse_model_user_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie and User Effects Model",
                                     RMSE = rmse_model_user_effects))
rmse_results %>% knitr::kable()
```

Incorporating the user bias into the model resulted in a further reduced RMSE.
Lets now add the "year" effect into the equation:

```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
year_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>% 
  group_by(year) %>%
  summarise(b_y = mean(rating - mu - b_i - b_u))
```

We can now construct predictors and see how much the RMSE improves:

```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>% 
  mutate(pred = mu + b_i + b_u + b_y) %>%
  pull(pred)
rmse_model_year_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_year_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User and Year Effects Model",
                                     RMSE = rmse_model_year_effects))
rmse_results %>% knitr::kable()
```

Now lets try incorporating the premier date effect on the ratings of the films.

```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
premier_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  group_by(premier_date) %>%
  summarise(b_p = mean(rating - mu - b_i - b_u - b_y))
```

We can now construct predictors and see how much the RMSE improves:

```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  mutate(pred = mu + b_i + b_u + b_y + b_p) %>%
  pull(pred)
rmse_model_premier_date_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_premier_date_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year and 
                                     Premier Date Effects Model",
                                     RMSE = rmse_model_premier_date_effects))
rmse_results %>% knitr::kable()
```

Lets see if we can improve any further adding the effect of the rating_age 
variable in the equation.

```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
rating_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  group_by(rating_age) %>%
  summarise(b_r = mean(rating - mu - b_i - b_u - b_y - b_p))
```

We can now construct predictors and see how much the RMSE improves:

```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r) %>%
  pull(pred)
rmse_model_rating_age_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_rating_age_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date
                                     and Rating Age Effects Model",
                                     RMSE = rmse_model_rating_age_effects))
rmse_results %>% knitr::kable()
```
Finally, we are going to incorporate the genres variables in the analysis

```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre1_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  group_by(genre1) %>%
  summarise(b_g1 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>% 
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1) %>%
  pull(pred)
rmse_model_genre1_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre1_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 Effects Model",
                                 RMSE = rmse_model_genre1_effects))
rmse_results %>% knitr::kable()
```

Furthermore, we will go with incorporating genre2-8 into the equation:
  
  ```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre2_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  group_by(genre2) %>%
  summarise(b_g2 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r - b_g1))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1 + b_g2) %>%
  pull(pred)
rmse_model_genre2_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre2_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 and 2 Effects Model",
                                 RMSE = rmse_model_genre2_effects))
rmse_results %>% knitr::kable()
```

Incorporating Genre3 variable:
  
  ```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre3_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  group_by(genre3) %>%
  summarise(b_g3 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r - b_g1 -
                          b_g2))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1 + b_g2 + b_g3) %>%
  pull(pred)
rmse_model_genre3_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre3_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 to 3 Effects Model",
                                 RMSE = rmse_model_genre3_effects))
rmse_results %>% knitr::kable()
```

Incorporating Genre4 variable:
  
  ```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre4_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  group_by(genre4) %>%
  summarise(b_g4 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r - b_g1 -
                          b_g2 - b_g3))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1 + b_g2 + b_g3 + 
           b_g4) %>%
  pull(pred)
rmse_model_genre4_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre4_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 to 4 Effects Model",
                                 RMSE = rmse_model_genre4_effects))
rmse_results %>% knitr::kable()
```

Incorporating Genre5 variable:
  
  ```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre5_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  group_by(genre5) %>%
  summarise(b_g5 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r - b_g1 -
                          b_g2 - b_g3 - b_g4))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  left_join(genre5_avgs, by="genre5") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1 + b_g2 + b_g3 + 
           b_g4 + b_g5) %>%
  pull(pred)
rmse_model_genre5_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre5_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 to 5 Effects Model",
                                 RMSE = rmse_model_genre5_effects))
rmse_results %>% knitr::kable()
```

Incorporating Genre6 variable:
  
  ```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre6_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  left_join(genre5_avgs, by="genre5") %>%
  group_by(genre6) %>%
  summarise(b_g6 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r - b_g1 -
                          b_g2 - b_g3 - b_g4 - b_g5))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  left_join(genre5_avgs, by="genre5") %>%
  left_join(genre6_avgs, by="genre6") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1 + b_g2 + b_g3 + 
           b_g4 + b_g5 + b_g6) %>%
  pull(pred)
rmse_model_genre6_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre6_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 to 6 Effects Model",
                                 RMSE = rmse_model_genre6_effects))
rmse_results %>% knitr::kable()
```

Incorporating Genre7 variable:
  
  ```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre7_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  left_join(genre5_avgs, by="genre5") %>%
  left_join(genre6_avgs, by="genre6") %>%
  group_by(genre7) %>%
  summarise(b_g7 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r - b_g1 -
                          b_g2 - b_g3 - b_g4 - b_g5 - b_g6))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  left_join(genre5_avgs, by="genre5") %>%
  left_join(genre6_avgs, by="genre6") %>%
  left_join(genre7_avgs, by="genre7") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1 + b_g2 + b_g3 + 
           b_g4 + b_g5 + b_g6 + b_g7) %>%
  pull(pred)
rmse_model_genre7_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre7_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 to 7 Effects Model",
                                 RMSE = rmse_model_genre7_effects))
rmse_results %>% knitr::kable()
```

Incorporating Genre8 variable:
  
  ```{r, echo=TRUE}
# Movie and User Effects Model
# Simple model taking into account the user effects, b_u
genre8_avgs <- train_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>%
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  left_join(genre5_avgs, by="genre5") %>%
  left_join(genre6_avgs, by="genre6") %>%
  left_join(genre7_avgs, by="genre7") %>%
  group_by(genre8) %>%
  summarise(b_g8 = mean(rating - mu - b_i - b_u - b_y - b_p - b_r - b_g1 -
                          b_g2 - b_g3 - b_g4 - b_g5 - b_g6 - b_g7))
```

We can now construct predictors and see how much the RMSE improves:
  
  ```{r}
predicted_ratings <- test_set %>%
  left_join(movie_avgs, by="movieId") %>%
  left_join(user_avgs, by="userId") %>%
  left_join(year_avgs, by="year") %>%
  left_join(premier_avgs, by="premier_date") %>% 
  left_join(rating_avgs, by="rating_age") %>% 
  left_join(genre1_avgs, by="genre1") %>%
  left_join(genre2_avgs, by="genre2") %>%
  left_join(genre3_avgs, by="genre3") %>%
  left_join(genre4_avgs, by="genre4") %>%
  left_join(genre5_avgs, by="genre5") %>%
  left_join(genre6_avgs, by="genre6") %>%
  left_join(genre7_avgs, by="genre7") %>%
  left_join(genre8_avgs, by="genre8") %>%
  mutate(pred = mu + b_i + b_u + b_y + b_p + b_r + b_g1 + b_g2 + b_g3 + 
           b_g4 + b_g5 + b_g6 + b_g7 + b_g8) %>%
  pull(pred)
rmse_model_genre8_effects <- RMSE(test_set$rating, predicted_ratings)
rmse_model_genre8_effects
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Movie, User, Year, Premier Date,
                                     Rating Age and Genre1 to 8 Effects Model",
                                 RMSE = rmse_model_genre8_effects))
rmse_results %>% knitr::kable()
```

Up to this point, the best predictive model as per the lowest RMSE result is the
"Movie, User, Year, Premier Date,	Rating Age and Genre1 to 6 Effects Model".

## Regularization

Regularization allows for reduced errors caused by movies with few ratings which
can influence the prediction and skew the error metric. The method uses a tuning
parameter, $\lambda$, to minimise the RMSE. Therefore we are going to be 
modifying $b_i$, $b_u$, $b_y$, $b_p$, $b_r$, $b_g1$, $b_g2$, $b_g3$, $b_g4$, 
$b_g5$ and $b_g6$ for movies with limited ratings.

```{r, echo=TRUE}
# Predict via regularization, movie, user, year, premier date, rating age and
# genre1 to 6 effect model
# (as per https://rafalab.github.io/dsbook 33.9 Regularization)
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  
  mu <- mean(train_set$rating)
  
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarise(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- train_set %>%
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarise(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_y <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    group_by(year) %>%
    summarise(b_y = sum(rating - b_u - b_i - mu)/(n()+l))
  
  b_p <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    group_by(premier_date) %>%
    summarise(b_p = sum(rating - b_y - b_u - b_i - mu)/(n()+l))
  
  b_r <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    group_by(rating_age) %>%
    summarise(b_r = sum(rating - b_p - b_y - b_u - b_i - mu)/(n()+l))
  
  b_g1 <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    left_join(b_r, by="rating_age") %>%
    group_by(genre1) %>%
    summarise(b_g1 = sum(rating - b_r - b_p - b_y - b_u - b_i - mu)/(n()+l))
  
  b_g2 <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    left_join(b_r, by="rating_age") %>%
    left_join(b_g1, by="genre1") %>%
    group_by(genre2) %>%
    summarise(b_g2 = sum(rating - b_g1 - b_r - b_p - b_y - b_u - b_i - mu)/
                (n()+l))
  
  b_g3 <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    left_join(b_r, by="rating_age") %>%
    left_join(b_g1, by="genre1") %>%
    left_join(b_g2, by="genre2") %>%
    group_by(genre3) %>%
    summarise(b_g3 = sum(rating - b_g2 - b_g1 - b_r - b_p - b_y - b_u - b_i -
                           mu)/(n()+l))
  
  b_g4 <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    left_join(b_r, by="rating_age") %>%
    left_join(b_g1, by="genre1") %>%
    left_join(b_g2, by="genre2") %>%
    left_join(b_g3, by="genre3") %>%
    group_by(genre4) %>%
    summarise(b_g4 = sum(rating - b_g3 - b_g2 - b_g1 - b_r - b_p - b_y - b_u -
                           b_i - mu)/(n()+l))
  b_g5 <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    left_join(b_r, by="rating_age") %>%
    left_join(b_g1, by="genre1") %>%
    left_join(b_g2, by="genre2") %>%
    left_join(b_g3, by="genre3") %>%
    left_join(b_g4, by="genre4") %>%
    group_by(genre5) %>%
    summarise(b_g5 = sum(rating - b_g4 - b_g3 - b_g2 - b_g1 - b_r - b_p - b_y -
                           b_u - b_i - mu)/(n()+l))
  
  b_g6 <- train_set %>%
    left_join(b_i, by="movieId") %>%
    left_join(b_u, by="userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    left_join(b_r, by="rating_age") %>%
    left_join(b_g1, by="genre1") %>%
    left_join(b_g2, by="genre2") %>%
    left_join(b_g3, by="genre3") %>%
    left_join(b_g4, by="genre4") %>%
    left_join(b_g5, by="genre5") %>%
    group_by(genre6) %>%
    summarise(b_g6 = sum(rating - b_g5 - b_g4 - b_g3 - b_g2 - b_g1 - b_r - b_p -
                           b_y - b_u - b_i - mu)/(n()+l))
  
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_y, by="year") %>%
    left_join(b_p, by="premier_date") %>%
    left_join(b_r, by="rating_age") %>%
    left_join(b_g1, by="genre1") %>%
    left_join(b_g2, by="genre2") %>%
    left_join(b_g3, by="genre3") %>%
    left_join(b_g4, by="genre4") %>%
    left_join(b_g5, by="genre5") %>%
    left_join(b_g6, by="genre6") %>%
    mutate(pred = mu + b_g6 + b_g5 + b_g4 + b_g3 + b_g2 + b_g1 + b_r + b_p +
             b_y + b_u + b_i) %>%
    pull(pred)
  
  return(RMSE(test_set$rating, predicted_ratings))
  
})

rmse_regularization <- min(rmses)
rmse_regularization
```

```{r, echo=TRUE}
# Plot RMSE against Lambdas to find optimal lambda
qplot(lambdas, rmses)
lambda <- lambdas[which.min(rmses)]
lambda
rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Regularized Effects Model",
                                 RMSE = rmse_regularization))
rmse_results %>% knitr::kable()
```

Regularization of a Movie, User, Year, Premier Date, Rating Age, Genres 1 to 6
Effect model has lead to the lowest RMSE of the prediction methods for the 
MovieLens ratings system.

\pagebreak

# Results and Discussion

The final values of the prediction models are shown below;

```{r, echo=TRUE}
rmse_results %>% knitr::kable()
```

The final model optimized for the prediction is the following;

$$ Y_{u,i} = \mu + b_{i,n,\lambda} + b_{u,n,\lambda} + b_{y,n,\lambda} + 
  b_{p,n,\lambda} + b_{r,n,\lambda} + b_{g1,n,\lambda} + b_{g2,n,\lambda} + 
  b_{g3,n,\lambda} + b_{g4,n,\lambda} + b_{g5,n,\lambda} + b_{g6,n,\lambda} + 
  \epsilon_{u,i} $$
  With 0.8633239 the lowest value of RMSE predicted.

# Final Validation

Now the work will be tested in the validation set using the Regularization Model
which was the one who gave the lowest RMSE.

```{r}

mu <- mean(train_set$rating)

b_i <- edx %>%
  group_by(movieId) %>%
  summarise(b_i = sum(rating - mu)/(n()+lambda))

b_u <- edx %>%
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarise(b_u = sum(rating - b_i - mu)/(n()+lambda))

b_y <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  group_by(year) %>%
  summarise(b_y = sum(rating - b_u - b_i - mu)/(n()+lambda))

b_p <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  group_by(premier_date) %>%
  summarise(b_p = sum(rating - b_y - b_u - b_i - mu)/(n()+lambda))

b_r <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  group_by(rating_age) %>%
  summarise(b_r = sum(rating - b_p - b_y - b_u - b_i - mu)/(n()+lambda))

b_g1 <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  left_join(b_r, by="rating_age") %>%
  group_by(genre1) %>%
  summarise(b_g1 = sum(rating - b_r - b_p - b_y - b_u - b_i - mu)/(n()+lambda))

b_g2 <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  left_join(b_r, by="rating_age") %>%
  left_join(b_g1, by="genre1") %>%
  group_by(genre2) %>%
  summarise(b_g2 = sum(rating - b_g1 - b_r - b_p - b_y - b_u - b_i - mu)/
              (n()+lambda))

b_g3 <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  left_join(b_r, by="rating_age") %>%
  left_join(b_g1, by="genre1") %>%
  left_join(b_g2, by="genre2") %>%
  group_by(genre3) %>%
  summarise(b_g3 = sum(rating - b_g2 - b_g1 - b_r - b_p - b_y - b_u - b_i -
                         mu)/(n()+lambda))

b_g4 <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  left_join(b_r, by="rating_age") %>%
  left_join(b_g1, by="genre1") %>%
  left_join(b_g2, by="genre2") %>%
  left_join(b_g3, by="genre3") %>%
  group_by(genre4) %>%
  summarise(b_g4 = sum(rating - b_g3 - b_g2 - b_g1 - b_r - b_p - b_y - b_u -
                         b_i - mu)/(n()+lambda))
b_g5 <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  left_join(b_r, by="rating_age") %>%
  left_join(b_g1, by="genre1") %>%
  left_join(b_g2, by="genre2") %>%
  left_join(b_g3, by="genre3") %>%
  left_join(b_g4, by="genre4") %>%
  group_by(genre5) %>%
  summarise(b_g5 = sum(rating - b_g4 - b_g3 - b_g2 - b_g1 - b_r - b_p - b_y -
                         b_u - b_i - mu)/(n()+lambda))

b_g6 <- edx %>%
  left_join(b_i, by="movieId") %>%
  left_join(b_u, by="userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  left_join(b_r, by="rating_age") %>%
  left_join(b_g1, by="genre1") %>%
  left_join(b_g2, by="genre2") %>%
  left_join(b_g3, by="genre3") %>%
  left_join(b_g4, by="genre4") %>%
  left_join(b_g5, by="genre5") %>%
  group_by(genre6) %>%
  summarise(b_g6 = sum(rating - b_g5 - b_g4 - b_g3 - b_g2 - b_g1 - b_r - b_p -
                         b_y - b_u - b_i - mu)/(n()+lambda))

predicted_ratings <- validation %>%
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_y, by="year") %>%
  left_join(b_p, by="premier_date") %>%
  left_join(b_r, by="rating_age") %>%
  left_join(b_g1, by="genre1") %>%
  left_join(b_g2, by="genre2") %>%
  left_join(b_g3, by="genre3") %>%
  left_join(b_g4, by="genre4") %>%
  left_join(b_g5, by="genre5") %>%
  left_join(b_g6, by="genre6") %>%
  mutate(pred = mu + b_g6 + b_g5 + b_g4 + b_g3 + b_g2 + b_g1 + b_r + b_p +
           b_y + b_u + b_i) %>%
  pull(pred)

rmse_results <- bind_rows(rmse_results, 
                          tibble(method="Validated Regularized Effects Model",
                                 RMSE = RMSE(validation$rating, predicted_ratings)))
rmse_results %>% knitr::kable()
```

# Matrix Factorization

We will now try the matrix factorization approach using the recosystem package. 
For that we are going to extract from the edx table only three variables 
("MovieId", "UserId" and "ratings") so that a matrix can be created like this:
  
  ```{r}
set.seed(2021, sample.kind = "Rounding")
# Convert the train and test sets into recosystem input format
train_mf <-  with(train_set, data_memory(user_index = userId, 
                                         item_index = movieId, 
                                         rating = rating))
test_mf  <-  with(test_set,  data_memory(user_index = userId, 
                                         item_index = movieId, 
                                         rating = rating))

#Creating a model object (a Reference Class object in R) by calling the function
#Reco()
r = Reco()

#tunning "train_mf"
opts = r$tune(train_mf, opts = list(lrate = c(0.1, 0.2), costp_l1 = 0, 
                                    costq_l1 = 0, niter = 10))

#training the recommender model
r$train(train_mf, opts = c(opts$min))

#Making prediction on test set and calculating RMSE:
predicted_ratings_mf <- r$predict(test_mf, out_memory())
rmse_matrixfact <- RMSE(test_set$rating, predicted_ratings_mf)
rmse_results <- bind_rows(rmse_results,
                          tibble(method="Matrix Factorization",  
                                 RMSE = rmse_matrixfact))
rmse_results 
```

# Conclusion
A machine learning algorithm to predict the ratings from the Movie Lens dataset
was constructed. The optimal model incorporated the effects of user, movie, year
of the rating, premier date of the film, rating longevity and genres bias in the
model and these variables were regularized to eliminate movies with a low number
of ratings. The final model validated gives an RMSE of 0.8640019. In addition, a
matrix factorization model was tested using the 'recosystem' package, providing
a lower RMSE.