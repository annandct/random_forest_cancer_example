04_Compare_Models.md
================
2025-10-31

``` r
# 1. SETUP LIBRARIES
#library(tidymodels)  # For the core modeling workflow
library(tidyverse)   # For general data manipulation and loading
library(tidymodels)
library(dplyr)
library(tidyr)
library(here)        # For robust file paths
#library(glmnet)      # The 'engine' for penalized regression
library(vip)         # For variable importance plots
#library(conflicted)  # To manage function name conflicts
```

## Compare Machine Learning Methods

``` r
#load from models
rf_output <- read_rds(here("_models", "final_rf_output.rds"))
lasso_output <- read_rds(here("_models", "final_lasso_output.rds"))
svm_output <- read_rds(here("_models", "final_svm_output.rds"))
```

``` r
#combine results
combined_metrics <- bind_rows(
  rf_output$metrics %>% mutate(model = "Random Forest"),
  lasso_output$metrics %>% mutate(model = "Lasso Regression"),
  svm_output$metrics %>% mutate(model = "SVM with RBF Kernel")
)

glimpse(combined_metrics)
```

    ## Rows: 9
    ## Columns: 5
    ## $ .metric    <chr> "accuracy", "roc_auc", "brier_class", "accuracy", "roc_auc"…
    ## $ .estimator <chr> "binary", "binary", "binary", "binary", "binary", "binary",…
    ## $ .estimate  <dbl> 0.95774648, 0.98834005, 0.03663178, 0.95070423, 0.98855205,…
    ## $ .config    <chr> "Preprocessor1_Model1", "Preprocessor1_Model1", "Preprocess…
    ## $ model      <chr> "Random Forest", "Random Forest", "Random Forest", "Lasso R…

``` r
#metrics from preds
# Metrics from ML Course
ml_mets <- metric_set(accuracy, precision, recall)
#create F1 Score Formula
f1_score <- function(precision, recall) {
  2 * (precision * recall) / (precision + recall)
}

combined_predictions <- bind_rows(
  rf_output$preds %>%
    ml_mets(truth = diagnosis, estimate = .pred_class) %>%
    mutate(model = "Random Forest"),
  lasso_output$preds %>%
    ml_mets(truth = diagnosis, estimate = .pred_class) %>%
    mutate(model = "Lasso Regression"),
  svm_output$preds %>%
    ml_mets(truth = diagnosis, estimate = .pred_class) %>%
    mutate(model = "SVM with RBF Kernel")
)

plotting_predictions <- combined_predictions %>% 
  pivot_wider(id_cols = model, names_from = .metric, values_from = .estimate) %>% 
  group_by(model) %>% 
  mutate(F1 = f1_score(precision, recall)) %>% #to compute F1
  pivot_longer(cols = c(accuracy, precision, recall, F1), names_to = "metric", values_to = "value")

plotting_predictions %>% 
ggplot(aes(x=model, y=value, fill=metric)) +
  geom_bar(stat="identity", position="dodge") +
  labs(title="Metrics by Model",
       x="Model",
       y="Metrics") +
  theme_minimal()
```

![](../_plot_images/unnamed-chunk-4-1.png)<!-- -->

## Generate Confusion Matrix for Each Model

# Generate and print the confusion matrix

conf_matrix \<- conf_mat( test_predictions, truth = diagnosis, estimate
= .pred_class, options = list(positive = “M”) \# Set positive = “M” )

print(conf_matrix) autoplot(conf_matrix, type = “heatmap”)

``` r
#CONF matrix for each model, and then a heatmap plot of each confusion matrix
library(purrr)
library(ggplot2)
model_outputs <- list(
  "Random Forest" = rf_output,
  "Lasso Regression" = lasso_output,
  "SVM with RBF Kernel" = svm_output
)
confusion_matrices <- map(model_outputs, function(output) {
  conf_mat(
    output$preds,
    truth = diagnosis,
    estimate = .pred_class,
    options = list(positive = "M")
  )
})

# Plot confusion matrices
patchwork::wrap_plots(
map2(confusion_matrices, names(confusion_matrices), function(cm, model_name) {
  autoplot(cm, type = "heatmap") + ggtitle(paste(model_name))
})
)
```

![](../_plot_images/unnamed-chunk-5-1.png)<!-- -->

``` r
#visualize comparison
#str_extract text in model before first space " "
str_extract(combined_metrics$model, "^[^ ]+")
```

    ## [1] "Random" "Random" "Random" "Lasso"  "Lasso"  "Lasso"  "SVM"    "SVM"   
    ## [9] "SVM"

``` r
ggplot(combined_metrics, aes(x = str_extract(model, "^[^ ]+"), y = .estimate, fill = model)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~.metric, scales = "free_y") +
  labs(title = "Comparison of Machine Learning Methods on BCWDD",
       x = "Model",
       y = "Metric Value") +
  theme_minimal()
```

![](../_plot_images/unnamed-chunk-6-1.png)<!-- -->
