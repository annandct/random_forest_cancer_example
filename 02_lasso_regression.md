02_lasso_regression.md
================
2025-10-31

## OUTLINE OF ANALYSIS

Proposed Method 2: - (Implement Lasso): Implement Lasso Logistic
Regression. - Use Lasso (L1) Logistic Regression, specifically to see if
a sparse model can be interpretable for this dataset without sacrificing
too much accuracy. - Use cross-validation to select the optimal
regularization parameter (lambda). - The resulting model’s coefficients
will be analyzed to produce a ranked list of the most important
diagnostic features.

``` r
# 1. SETUP LIBRARIES
library(tidymodels)  # For the core modeling workflow
library(tidyverse)   # For general data manipulation and loading
library(here)        # For robust file paths
library(glmnet)      # The 'engine' for penalized regression
library(vip)         # For variable importance plots
library(conflicted)  # To manage function name conflicts

# Set seed: 
set.seed(8888) 
```

### 1. Load and Prepare Data

``` r
#load data
cancer_data <- readRDS(here("_data", "cancer_data_processed.rds")) #processed in00 data_wrangling script.
str(cancer_data)
```

    ## tibble [568 × 32] (S3: tbl_df/tbl/data.frame)
    ##  $ id                     : num [1:568] 842302 842517 84300903 84348301 84358402 ...
    ##  $ diagnosis              : Factor w/ 2 levels "B","M": 2 2 2 2 2 2 2 2 2 2 ...
    ##  $ radius_mean            : num [1:568] 18 20.6 19.7 11.4 20.3 ...
    ##  $ texture_mean           : num [1:568] 10.4 17.8 21.2 20.4 14.3 ...
    ##  $ perimeter_mean         : num [1:568] 122.8 132.9 130 77.6 135.1 ...
    ##  $ area_mean              : num [1:568] 1001 1326 1203 386 1297 ...
    ##  $ smoothness_mean        : num [1:568] 0.1184 0.0847 0.1096 0.1425 0.1003 ...
    ##  $ compactness_mean       : num [1:568] 0.2776 0.0786 0.1599 0.2839 0.1328 ...
    ##  $ concavity_mean         : num [1:568] 0.3001 0.0869 0.1974 0.2414 0.198 ...
    ##  $ concave_points_mean    : num [1:568] 0.1471 0.0702 0.1279 0.1052 0.1043 ...
    ##  $ symmetry_mean          : num [1:568] 0.242 0.181 0.207 0.26 0.181 ...
    ##  $ fractal_dimension_mean : num [1:568] 0.0787 0.0567 0.06 0.0974 0.0588 ...
    ##  $ radius_se              : num [1:568] 1.095 0.543 0.746 0.496 0.757 ...
    ##  $ texture_se             : num [1:568] 0.905 0.734 0.787 1.156 0.781 ...
    ##  $ perimeter_se           : num [1:568] 8.59 3.4 4.58 3.44 5.44 ...
    ##  $ area_se                : num [1:568] 153.4 74.1 94 27.2 94.4 ...
    ##  $ smoothness_se          : num [1:568] 0.0064 0.00522 0.00615 0.00911 0.01149 ...
    ##  $ compactness_se         : num [1:568] 0.049 0.0131 0.0401 0.0746 0.0246 ...
    ##  $ concavity_se           : num [1:568] 0.0537 0.0186 0.0383 0.0566 0.0569 ...
    ##  $ concave_points_se      : num [1:568] 0.0159 0.0134 0.0206 0.0187 0.0188 ...
    ##  $ symmetry_se            : num [1:568] 0.03 0.0139 0.0225 0.0596 0.0176 ...
    ##  $ fractal_dimension_se   : num [1:568] 0.00619 0.00353 0.00457 0.00921 0.00511 ...
    ##  $ radius_worst           : num [1:568] 25.4 25 23.6 14.9 22.5 ...
    ##  $ texture_worst          : num [1:568] 17.3 23.4 25.5 26.5 16.7 ...
    ##  $ perimeter_worst        : num [1:568] 184.6 158.8 152.5 98.9 152.2 ...
    ##  $ area_worst             : num [1:568] 2019 1956 1709 568 1575 ...
    ##  $ smoothness_worst       : num [1:568] 0.162 0.124 0.144 0.21 0.137 ...
    ##  $ compactness_worst      : num [1:568] 0.666 0.187 0.424 0.866 0.205 ...
    ##  $ concavity_worst        : num [1:568] 0.712 0.242 0.45 0.687 0.4 ...
    ##  $ concave_points_worst   : num [1:568] 0.265 0.186 0.243 0.258 0.163 ...
    ##  $ symmetry_worst         : num [1:568] 0.46 0.275 0.361 0.664 0.236 ...
    ##  $ fractal_dimension_worst: num [1:568] 0.1189 0.089 0.0876 0.173 0.0768 ...

### 2. Data Splitting

We use the same stratified split parameters and seed as the SVM script
to ensure the train/test sets are identical.

``` r
# 3. DATA SPLITTING
# Create the stratified data split (75% training, 25% testing)
data_split <- initial_split(cancer_data, prop = 0.75, strata = diagnosis)

# Create the training and testing data frames
train_data <- training(data_split)
test_data  <- testing(data_split)

# Check the proportions in each set
train_data %>% count(diagnosis) %>% mutate(prop = n/sum(n))
```

    ## # A tibble: 2 × 3
    ##   diagnosis     n  prop
    ##   <fct>     <int> <dbl>
    ## 1 B           267 0.627
    ## 2 M           159 0.373

``` r
test_data %>% count(diagnosis) %>% mutate(prop = n/sum(n))
```

    ## # A tibble: 2 × 3
    ##   diagnosis     n  prop
    ##   <fct>     <int> <dbl>
    ## 1 B            89 0.627
    ## 2 M            53 0.373

### 3. Preprocessing Recipe

This recipe is nearly identical to the SVM one. Scaling and centering
are **critical** for Lasso regression, as the penalty is applied
directly to the coefficients’ values.

``` r
# 4. PREPROCESSING RECIPE
# We are predicting 'diagnosis' using all other variables (.).
# The only predictor we *don't* want is 'id', so we remove it.
# We then normalize (center and scale) ALL numeric predictors.
lasso_recipe <- recipe(diagnosis ~ ., data = train_data) %>%
  update_role(id, new_role = "ID") %>% # Keep ID but don't use it
  step_normalize(all_numeric_predictors()) # Center and scale
```

### 4. Cross-Validation Folds

We create the *exact same* 10-fold cross-validation folds from our
training data, as the `set.seed()` was identical.

``` r
# 5. CROSS-VALIDATION FOLDS
# Create 10-fold cross-validation folds from the training data
cv_folds <- vfold_cv(train_data, v = 10, strata = diagnosis)
```

### 5. Model Specification & Tuning

Here we define the `logistic_reg` model. We set `mixture = 1` to specify
*pure Lasso* (L1 penalty) and tell `tidymodels` we want to `tune()` the
`penalty` (lambda) parameter.

``` r
# 6. MODEL SPECIFICATION
# We specify a logistic regression model.
# 'mixture = 1' makes it a pure Lasso model (L1 penalty).
# We set the 'engine' to 'glmnet'.
# We will tune the 'penalty' (lambda) hyperparameter.
lasso_spec <- logistic_reg(penalty = tune(), mixture = 1) %>%
  set_engine("glmnet") %>%
  set_mode("classification")

# 7. CREATE WORKFLOW
# Combine the recipe and model spec
lasso_workflow <- workflow() %>%
  add_recipe(lasso_recipe) %>%
  add_model(lasso_spec)

# 8. HYPERPARAMETER TUNING
# Create a grid of 50 different 'penalty' (lambda) values to try
lasso_grid <- grid_regular(penalty(), levels = 50)
#ctrl <- control_grid(save_workflow = TRUE)
# Run the tuning process
lasso_tune_results <- tune_grid(
  lasso_workflow,
  resamples = cv_folds,
  grid = lasso_grid,
  control = control_grid(save_workflow = TRUE),
  metrics = metric_set(accuracy, precision, recall))

# Show the top 5 best-performing penalty values
show_best(lasso_tune_results, metric = "accuracy")
```

    ## # A tibble: 5 × 7
    ##    penalty .metric  .estimator  mean     n std_err .config              
    ##      <dbl> <chr>    <chr>      <dbl> <int>   <dbl> <chr>                
    ## 1 3.56e- 3 accuracy binary     0.979    10 0.00655 Preprocessor1_Model38
    ## 2 2.22e- 3 accuracy binary     0.976    10 0.00611 Preprocessor1_Model37
    ## 3 5.69e- 3 accuracy binary     0.976    10 0.00611 Preprocessor1_Model39
    ## 4 1   e-10 accuracy binary     0.972    10 0.00909 Preprocessor1_Model01
    ## 5 1.60e-10 accuracy binary     0.972    10 0.00909 Preprocessor1_Model02

``` r
# Select the single best penalty value
best_lasso_params <- select_best(lasso_tune_results, metric = "accuracy")

#fit best model ^^ this is duplication of code from above, but worth seeing/saving the params. Can also fit with the params
best_lasso_fit <- fit_best(lasso_tune_results)
tidy(best_lasso_fit)
```

    ## # A tibble: 31 × 3
    ##    term                estimate penalty
    ##    <chr>                  <dbl>   <dbl>
    ##  1 (Intercept)           -0.545 0.00356
    ##  2 radius_mean            0     0.00356
    ##  3 texture_mean           0     0.00356
    ##  4 perimeter_mean         0     0.00356
    ##  5 area_mean              0     0.00356
    ##  6 smoothness_mean        0     0.00356
    ##  7 compactness_mean       0     0.00356
    ##  8 concavity_mean         0.118 0.00356
    ##  9 concave_points_mean    0     0.00356
    ## 10 symmetry_mean          0     0.00356
    ## # ℹ 21 more rows

### 6. Finalize and Evaluate Model

We update our workflow with the *best* penalty and fit it to the full
training set, then evaluate on the test set.

``` r
# 9. FINALIZE WORKFLOW
# Update the workflow with the best penalty we just found
# same as best_lasso_fit...
final_lasso_workflow <- finalize_workflow(lasso_workflow,
  #lasso_workflow,
  best_lasso_params
) 

# 10. FIT ON TRAIN, EVALUATE ON TEST
# Use 'last_fit' to fit on the full training set and evaluate on the test set
final_lasso_fit <- last_fit(
  final_lasso_workflow,
  data_split
)
```

### 7. Review Performance Metrics

Let’s see how our final tuned Lasso model performed on the test data.

``` r
# 11. COLLECT METRICS
# Get the metrics from our test-set evaluation
test_metrics <- collect_metrics(final_lasso_fit)
print(test_metrics)
```

    ## # A tibble: 3 × 4
    ##   .metric     .estimator .estimate .config             
    ##   <chr>       <chr>          <dbl> <chr>               
    ## 1 accuracy    binary        0.951  Preprocessor1_Model1
    ## 2 roc_auc     binary        0.989  Preprocessor1_Model1
    ## 3 brier_class binary        0.0340 Preprocessor1_Model1

``` r
# Get the predictions to build a confusion matrix
test_predictions <- collect_predictions(final_lasso_fit)

# Generate and print the confusion matrix
# We again set positive = "M" for consistency
conf_matrix <- conf_mat(
  test_predictions,
  truth = diagnosis,
  estimate = .pred_class,
  options = list(positive = "M") 
)

print(conf_matrix)
```

    ##           Truth
    ## Prediction  B  M
    ##          B 88  6
    ##          M  1 47

``` r
autoplot(conf_matrix, type = "heatmap")
```

![](../_plot_images/metrics-1.png)<!-- -->

### 8. Feature Importance (Interpretability)

This is the key step for the Lasso model. We will extract the
coefficients from our final model to see which features it “kept” (gave
a non-zero coefficient) and which it “threw away” (set to zero).

``` r
# 12. EXTRACT FEATURE IMPORTANCE
#library(vip)
# Extract the fitted model object
fitted_lasso_model <- extract_workflow(final_lasso_fit)

# Use the 'vip' package to create a variable importance plot
# This shows the non-zero coefficients
print("Plotting Variable Importance...")
```

    ## [1] "Plotting Variable Importance..."

``` r
vip(fitted_lasso_model, num_features = 20) +
  labs(title = "Lasso Variable Importance",
       subtitle = "Features with non-zero coefficients")
```

![](../_plot_images/vip-1.png)<!-- -->

``` r
# We can also 'tidy' the model to see the coefficients as a data frame
print("Model Coefficients:")
```

    ## [1] "Model Coefficients:"

``` r
tidy(best_lasso_fit) %>%
  dplyr::filter(estimate != 0) %>%  # Filter to show only non-zero features
  arrange(desc(abs(estimate))) # Sort by importance
```

    ## # A tibble: 13 × 3
    ##    term                   estimate penalty
    ##    <chr>                     <dbl>   <dbl>
    ##  1 radius_worst             4.35   0.00356
    ##  2 radius_se                2.21   0.00356
    ##  3 texture_worst            1.39   0.00356
    ##  4 smoothness_worst         1.08   0.00356
    ##  5 concavity_worst          0.907  0.00356
    ##  6 concave_points_worst     0.693  0.00356
    ##  7 symmetry_worst           0.636  0.00356
    ##  8 (Intercept)             -0.545  0.00356
    ##  9 fractal_dimension_mean  -0.447  0.00356
    ## 10 symmetry_se             -0.247  0.00356
    ## 11 compactness_se          -0.212  0.00356
    ## 12 concavity_mean           0.118  0.00356
    ## 13 texture_se              -0.0142 0.00356

### 9. Save Model Artifacts

``` r
# 13. SAVE ARTIFACTS
# Save the final, fitted workflow
saveRDS(
  list("model" = final_lasso_fit, "metrics" = test_metrics, "preds" = test_predictions),
  file = here("_models", "final_lasso_output.rds")
)
```
