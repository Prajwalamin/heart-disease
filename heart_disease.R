# Installing packages
install.packages("tidyverse")
install.packages("FactoMineR")
install.packages("factoextra")
install.packages("cluster")
install.packages("plotly")
install.packages("keras")
install.packages("tensorflow")
install.packages("dplyr")
install.packages("caret")
install.packages("StatMatch")
install.packages("ggplot2")
install.packages("ggmap")
install.packages('dataMaid')
install.packages('SmartEDA')



# Libraries
library(tidyverse)
library("FactoMineR")
library("factoextra")
library(cluster)
library(plotly)
library(keras)
library(tensorflow)
library(dplyr)
library(caret)
library(StatMatch)
library(ggplot2)
library(ggmap)
library(rsample)
library(RANN)
library(dataMaid)
library(SmartEDA)
library(rpart)
library(rpart.plot)

# Setting the path

famd_path <- 'datasets/famd_data.csv'
tsne_path <- 'datasets/tsne_data.csv'
data_path <- "datasets/heart_noencoding.csv"
data_normal <- 'datasets/heart_processed.csv'
data_raw <- 'datasets/heart_dataset.csv'

# Reading the dataset

famd_data <- read.csv(famd_path)
tsne_data <- read.csv(tsne_path)
heart <- read.csv(data_path)
heart_normal <- read.csv(data_normal)

view(heart)

# Exploratory Data Analysis

# data report of eda using 'dataMaid' function


heart %>% summary()
heart_normal %>% summary()

ggplot(data = heart, aes(x = cp)) +
  stat_count(fill = "steelblue") +
  labs(title = "Chest Pain Type Bar Plot", x = "Chest Pain Type", y = "Count") +
  theme_minimal()


# Boxplots (OUTLIERS)

boxplot(heart$chol)

summary(heart)

# DATA PROCESSING

# Convert categorical variables to factors
heart <- heart %>%
  mutate_if(is.character, as.factor)

# Convert logical variables to factors
heart <- heart %>%
  mutate_if(is.logical, as.factor)


str(heart)

summary(heart)






# PIPELINE

# Keep target variable aside
target_data <- heart$target

# 1. Without encoding
data_pipeline <- function(data) {
  
  # Separate numerical and categorical variables
  numerical_vars <- names(heart)[sapply(heart, is.numeric)]
  categorical_vars <- names(heart)[sapply(heart, is.factor)]
  
  # Remove target variable from numerical_vars
  numerical_vars <- setdiff(numerical_vars, "target")
  
  # Imputation for numerical variables using k-NN
  preprocessor_num <- preProcess(data[numerical_vars], method = "knnImpute") # Use k-NN for imputation
  numerical_imputed <- predict(preprocessor_num, newdata = data[numerical_vars])
  # Replace with desired imputation method for numerical variables
  
  # Imputation for categorical variables
  categorical_imputed <- data %>%
    select(all_of(categorical_vars)) %>%
    mutate_all(~ ifelse(is.na(.), mode(.), as.character(.)))  # Replace with desired imputation method for categorical variables
  
  # Combine the imputed variables
  imputed_data <- bind_cols(numerical_imputed, categorical_imputed)
  
  # Exclude scaling for the target variable
  numerical_vars_no_target <- setdiff(numerical_vars, "target")
  
  preprocessor <- preProcess(imputed_data[numerical_vars_no_target], method = c("center", "scale"))  # Scale the data by centering and scaling
  
  # Apply scaling transformation to numerical variables
  imputed_data[numerical_vars] <- predict(preprocessor, newdata = imputed_data[numerical_vars])
  
  # Return the scaled data
  return(imputed_data)
}

heart_proc <- data_pipeline(heart)
heart_proc$target <- target_data
view(heart_proc)



# 2. One Hot Encoding

data_pipeline_ohe <- function(data) {
  
  # Separate numerical and categorical variables
  numerical_vars <- names(data)[sapply(data, is.numeric)]
  categorical_vars <- names(data)[sapply(data, is.factor)]
  
  # Remove target variable from numerical_vars
  numerical_vars <- setdiff(numerical_vars, "target")
  
  # Imputation for numerical variables using k-NN
  preprocessor_num <- preProcess(data[numerical_vars], method = "knnImpute") # Use k-NN for imputation
  numerical_imputed <- predict(preprocessor_num, newdata = data[numerical_vars])
  
  # Imputation for categorical variables
  categorical_imputed <- data %>%
    select(all_of(categorical_vars)) %>%
    mutate_all(~ ifelse(is.na(.), mode(.), as.character(.)))  # Replace with desired imputation method for categorical variables
  
  # Combine the imputed variables
  imputed_data <- bind_cols(numerical_imputed, categorical_imputed)
  
  # Scaling
  preprocessor <- preProcess(imputed_data[numerical_vars], method = c("center", "scale"))  # Scale the data by centering and scaling
  
  # Apply scaling transformation to numerical variables
  imputed_data[numerical_vars] <- predict(preprocessor, newdata = imputed_data[numerical_vars])
  
  # One-hot encoding for categorical variables
  dummies <- dummyVars(~., data = imputed_data[categorical_vars])
  cat_encoded_data <- predict(dummies, newdata = imputed_data[categorical_vars])
  
  # Combine scaled numerical data and encoded categorical data
  final_data <- bind_cols(imputed_data[numerical_vars], cat_encoded_data)
  
  # Return the final data
  return(final_data)
}

heart_ohe <- data_pipeline_ohe(heart)
heart_ohe$target <- target_data
view(heart_ohe)




# <--  FAMD Analysis  -->

heart_proc <- select(heart, -target)
view(heart_proc)

# this works with mixed data so processed data without any encoding is passed
FAMD(heart_proc, ncp = 10, sup.var = NULL, ind.sup = NULL, graph = FALSE)
res.famd <- FAMD(heart_proc, ncp=10, graph = FALSE)
print(res.famd)

# Visualization
eig.val <- get_eigenvalue(res.famd)
head(eig.val,10)

fviz_screeplot(res.famd, n = 10)

var <- get_famd_var(res.famd)
var


# Plot of variables (Correlation)
fviz_famd_var(res.famd, repel = TRUE)
# Contribution to the first dimension
fviz_contrib(res.famd, "var", axes = 1)
# Contribution to the second dimension
fviz_contrib(res.famd, "var", axes = 2)
# Contribution to the second dimension
fviz_contrib(res.famd, "var", axes = 5)

# quantitative variables
quanti.var <- get_famd_var(res.famd, "quanti.var")
quanti.var 

fviz_famd_var(res.famd, "quanti.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
              repel = TRUE)

# Color by cos2 values: quality on the factor map
fviz_famd_var(res.famd, "quanti.var", col.var = "cos2",
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"), 
              repel = TRUE)

# qualitative variables
fviz_famd_var(res.famd, "quali.var", col.var = "contrib", 
              gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07")
)



# Finding the Optimal number of clusters using NBclust

# Install and load the NbClust package
library(NbClust)

#FAMD
# Assuming your data is stored in a dataframe named 'data'
result_famd <- NbClust(famd_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")

# Print the suggested number of clusters
cat("Suggested number of clusters:", result$Best.nc, "\n")

#t-SNE
# Assuming your data is stored in a dataframe named 'data'
result_tsne <- NbClust(tsne_data, distance = "euclidean", min.nc = 2, max.nc = 10, method = "kmeans")

# Print the suggested number of clusters
cat("Suggested number of clusters:", result$Best.nc, "\n")





# K-means

# Extract the factor scores from the FAMD result
factor_scores <- res.famd$ind$coord

# Remove the cluster column
heart <- select(heart, -cluster)

# Perform Clustering (e.g., k-means)
set.seed(123)
k <- 2  # Number of desired clusters
kmeans_result <- kmeans(factor_scores, centers = k)

# Get the cluster assignments for each patient
cluster_assignments <- kmeans_result$cluster

# Add cluster assignments back to the original dataset
heart$cluster <- cluster_assignments

# View the segmented patient groups
table(heart$cluster)

# Create a hover text with patient details
hover_text <- with(heart, paste("Patient ID:", rownames(heart), "<br>",
                                     "Age:", age, "<br>",
                                     "Sex:", sex, "<br>",
                                     "cp:", cp, "<br>",
                                     "trestbps:", trestbps, "<br>",
                                     "Cholestrol:", chol, "<br>",
                                     "fbs:", fbs, "<br>",
                                     "oldpeak:", oldpeak, "<br>",
                                     "thalach:", thalach, "<br>",
                                     "exang:", exang, "<br>",
                                     "Slope:", slope, "<br>",
                                     "Heart Disease?:", target))

# Plot the clusters using Plotly
plot_data <- data.frame(x = pca_data[, 1], y = pca_data[, 2], cluster = as.factor(cluster_assignments), hover_text)
plot_data <- data.frame(x = factor_scores[, 1], y = factor_scores[, 2], cluster = as.factor(cluster_assignments), hover_text)


plot <- plot_ly(data = plot_data, x = ~x, y = ~y, color = ~cluster, text = ~hover_text,
                type = "scatter", mode = "markers", hoverinfo = "text",
                marker = list(size = 10, line = list(color = "white", width = 0.05)))

plot <- plot %>% layout(title = "Patient Segmentation",
                        xaxis = list(title = "Dimension 1"),
                        yaxis = list(title = "Dimension 2"))

plot





# EDA Post-clustering

# 1. Descriptive statistics

cluster_data <- split(heart[, c("age", "sex", "chol", "cp","fbs", "target")], cluster_assignments)

# Compute summary statistics for each cluster
cluster_summary <- lapply(cluster_data, summary)

# View the summary statistics for each cluster
for (i in seq_along(cluster_summary)) {
  cat("Cluster", i, "\n")
  print(cluster_summary[[i]])
  cat("\n")
}



# 2. Distribution of heart disease patients.

heart_disease <- table(heart$cluster, heart$target)
print(heart_disease)

table(heart$cp, heart$target)

qa# Create a contingency table of cluster counts
cluster_table <- table(cluster_assignments)

# Calculate the mean age in each cluster
mean_age <- aggregate(age ~ cluster_assignments, data = heart, FUN = mean)

# Display the contingency table and mean age in each cluster
print(mean_age)


# 3. Histogram for types of cp levels in each cluster

# Create the grouped bar graph comparing 'cp' against clusters
bar_graph <- plot_ly(heart, x = ~cp, color = ~factor(cluster), type = "histogram") %>%
  layout(barmode = "group", xaxis = list(title = "Chest Pain (cp)"), yaxis = list(title = "Count"))
bar_graph



