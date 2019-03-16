library(sparklyr) 
library(ggplot2) 
library(dplyr)
sc <- spark_connect(master = "local") 
iris_tbl <- copy_to(sc, iris, "iris", overwrite = TRUE) 
iris_tbl
lm_model <- iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  ml_linear_regression(Petal_Length ~ Petal_Width)

iris_tbl %>%
  select(Petal_Width, Petal_Length) %>%
  collect %>%
  ggplot(aes(Petal_Length, Petal_Width)) +
  geom_point(aes(Petal_Width, Petal_Length), size = 2, alpha = 0.5) +
  geom_abline(aes(slope = coef(lm_model)[["Petal_Width"]],
                  intercept = coef(lm_model)[["(Intercept)"]]),
              color = "red") +
  labs(
    x = "Petal Width",
    y = "Petal Length",
    title = "Linear Regression: Petal Length ~ Petal Width",
    subtitle = "Use Spark.ML linear regression to predict petal length as a function of petal width."
  )

pca_model <- tbl(sc, "iris") %>%
  select(-Species) %>%
  ml_pca()
print(pca_model)

#b. Apply random forest, logistic regression using Spark R
#c. Predict for new dataset

#Random Forest 
#Use Spark's Random Forest to perform regression or multiclass classification.
rf_model <- iris_tbl %>%
  ml_random_forest(Species ~ Petal_Length + Petal_Width, type = "classification")

rf_predict <- ml_predict(rf_model, iris_tbl) %>%
  ft_string_indexer("Species", "Species_idx") %>%
  collect

table

partitions <- tbl(sc, "iris") %>%
  sdf_partition(training = 0.75, test = 0.25, seed = 1099)

fit <- partitions$training %>%
  ml_linear_regression(Petal_Length ~ Petal_Width)

estimate_mse <- function(df){
  ml_predict(fit, df) %>%
    mutate(resid = Petal_Length - prediction) %>%
    summarize(mse = mean(resid ^ 2)) %>%
    collect
}

sapply(partitions, estimate_mse)

ft_string2idx <- iris_tbl %>% 
  ft_string_indexer("Species", "Species_idx") %>% 
  ft_index_to_string("Species_idx", "Species_remap") %>% 
  collect

table(ft_string2idx$Species, ft_string2idx$Species_remap)  

ft_string2idx <- iris_tbl %>% 
  sdf_mutate(Species_idx = ft_string_indexer(Species)) %>% 
  sdf_mutate(Species_remap = ft_index_to_string(Species_idx)) %>% 
  collect 

ft_string2idx %>% 
  select(Species, Species_idx, Species_remap) %>% 
  distinct

beaver <- beaver2 
beaver$activ <- factor(beaver$activ, labels = c("Non-Active", "Active")) 
copy_to(sc, beaver, "beaver")

beaver_tbl <- tbl(sc, "beaver")

glm_model <- beaver_tbl %>%
  mutate(binary_response = as.numeric(activ == "Active")) %>%
  ml_logistic_regression(binary_response ~ temp)

glm_model

mtcars_tbl <- copy_to(sc, mtcars, "mtcars")

partitions <- mtcars_tbl %>% 
  filter(hp >= 100) %>% 
  sdf_mutate(cyl8 = ft_bucketizer(cyl, c(0,8,12))) %>% 
  sdf_partition(training = 0.5, test = 0.5, seed = 888)

fit <- partitions$training %>%
  ml_linear_regression(mpg ~ wt + cyl)

summary(fit)


  
