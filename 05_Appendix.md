
## Workflow Reproducibility and Model Comparison

This script outlines the ‘tidymodels’ workflow used for the project.

``` r
#library(tidymodels)   For the core modeling framework
#library(tidyverse)    For data manipulation and visualization
#library(vip)          Used for variable importance plots
#library(kernlab)      SVM Library
```

### (A) Dataset Description

Breast Cancer Wisconsin (Diagnostic) Dataset (BCWDD) - Dimensions: 568
observations (rows) and 32 variables (columns). - Target Variable:
‘diagnosis’ (Factor w/ 2 levels: “B” for Benign, “M” for Malignant). -
Predictors: 30 numerical features (e.g., radius_mean, texture_mean,
area_worst). - ID Variable: ‘id’ (removed during preprocessing).

### (B) Tidymodels Approach for Reproducibility

``` r
#Set a single seed for all processes to ensure reproducibility
 set.seed(8888)

# 1. Data Splitting
#  The data was split into training (75%) and testing (25%) sets.
#  Stratification on 'diagnosis' was used to maintain class balance.
 data_split <- initial_split(cancer_data, prop = 0.75, strata = diagnosis)
 train_data <- training(data_split)
 test_data  <- testing(data_split)

# 2. Cross-Validation Folds
#   10-fold cross-validation was created from the *training data*
#   This same set of folds ('cv_folds') was used to tune all three models.
  cv_folds <- vfold_cv(train_data, v = 10, strata = diagnosis)

# 3. Preprocessing Recipe
#   A single, consistent recipe was defined to be applied to all models.
#   This ensures all models received the exact same processed data.
  norm_recipe <- recipe(diagnosis ~ ., data = train_data) %>%
   step_rm(id) %>%   The 'id' column is not a predictor
   step_normalize(all_numeric_predictors())  Center and scale all 30 predictors


# 4. Model Specifications for Tuning
# These are the "blueprints" for the models, with parameters set to 'tune()'.
 
  # --- Model 1: Random Forest ---
 rf_spec_tune <- rand_forest(
   mtry = tune(),   Number of predictors to sample
   trees = 1000,    1000 trees is generally sufficient
   min_n = tune()   Minimum node size
 ) %>%
   set_engine("ranger", importance = "permutation") %>%
   set_mode("classification")

  # --- Model 2: Lasso (L1) Logistic Regression ---
 lasso_spec_tune <- logistic_reg(
   penalty = tune(),  The L1 regularization penalty (lambda)
   mixture = 1        mixture = 1 specifies a pure Lasso model
 ) %>%
   set_engine("glmnet") %>%
   set_mode("classification")

  # --- Model 3: Support Vector Machine (RBF Kernel) ---
 svm_spec_tune <- svm_rbf(
   cost = tune(),       The cost of misclassification
   rbf_sigma = tune()   The gamma/sigma of the RBF kernel
 ) %>%
   set_engine("kernlab") %>%
   set_mode("classification")

# 5. Model Fitting Routine (Example)
# Each model was then tuned using a workflow, tune_grid, and the 'cv_folds'.
  # The example below is for the Random Forest.

  rf_workflow_tune <- workflow() %>%
    add_model(rf_spec_tune) %>%
    add_recipe(norm_recipe)
 
  rf_tune_results <- tune_grid(
    rf_workflow_tune,
    resamples = cv_folds,
    grid = 20  Test 20 candidate hyperparameter combinations
  )
 
  #  The 'best' model was selected, finalized, and fit one last time
  #  on the full training set and evaluated on the test set.
  best_rf_params <- select_best(rf_tune_results, metric = "accuracy")
 
  final_rf_fit <- last_fit(
    finalize_workflow(rf_workflow_tune, best_rf_params),
    data_split
  )
```

### (C) Final Tables Comparing Model Outcomes

The data below is hard-coded from the final, verified results from
running the ‘last_fit()’ evaluation on the test set (n=142).

<table class="table table-striped table-hover table-condensed" style="color: black; width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="border-bottom:hidden;padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="4">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

Final Performance Metrics (Test Set)

</div>

</th>

</tr>

<tr>

<th style="text-align:left;">

model
</th>

<th style="text-align:right;">

accuracy
</th>

<th style="text-align:right;">

precision
</th>

<th style="text-align:right;">

recall
</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

Random Forest
</td>

<td style="text-align:right;">

0.958
</td>

<td style="text-align:right;">

0.980
</td>

<td style="text-align:right;">

0.906
</td>

</tr>

<tr>

<td style="text-align:left;">

Lasso Regression
</td>

<td style="text-align:right;">

0.951
</td>

<td style="text-align:right;">

0.979
</td>

<td style="text-align:right;">

0.887
</td>

</tr>

<tr>

<td style="text-align:left;">

SVM (RBF)
</td>

<td style="text-align:right;">

0.958
</td>

<td style="text-align:right;">

0.980
</td>

<td style="text-align:right;">

0.906
</td>

</tr>

</tbody>

</table>

<table class="table table-striped table-hover table-condensed" style="color: black; width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="border-bottom:hidden;padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="5">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

Final Confusion Matrix (Test Set)

</div>

</th>

</tr>

<tr>

<th style="text-align:left;">

model
</th>

<th style="text-align:right;">

true_negative_B
</th>

<th style="text-align:right;">

false_positive_M
</th>

<th style="text-align:right;">

false_negative_B
</th>

<th style="text-align:right;">

true_positive_M
</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

Random Forest
</td>

<td style="text-align:right;">

88
</td>

<td style="text-align:right;">

1
</td>

<td style="text-align:right;">

5
</td>

<td style="text-align:right;">

48
</td>

</tr>

<tr>

<td style="text-align:left;">

Lasso Regression
</td>

<td style="text-align:right;">

88
</td>

<td style="text-align:right;">

1
</td>

<td style="text-align:right;">

6
</td>

<td style="text-align:right;">

47
</td>

</tr>

<tr>

<td style="text-align:left;">

SVM (RBF)
</td>

<td style="text-align:right;">

88
</td>

<td style="text-align:right;">

1
</td>

<td style="text-align:right;">

5
</td>

<td style="text-align:right;">

48
</td>

</tr>

</tbody>

</table>

<table class="table table-striped table-hover table-condensed" style="color: black; width: auto !important; margin-left: auto; margin-right: auto;">

<thead>

<tr>

<th style="border-bottom:hidden;padding-bottom:0; padding-left:3px;padding-right:3px;text-align: center; " colspan="3">

<div style="border-bottom: 1px solid #ddd; padding-bottom: 5px; ">

Final Confusion Matrix (Test Set)

</div>

</th>

</tr>

<tr>

<th style="text-align:left;">

model
</th>

<th style="text-align:left;">

parameter_1
</th>

<th style="text-align:left;">

parameter_2
</th>

</tr>

</thead>

<tbody>

<tr>

<td style="text-align:left;">

Random Forest
</td>

<td style="text-align:left;">

mtry = 10
</td>

<td style="text-align:left;">

min_n = 6
</td>

</tr>

<tr>

<td style="text-align:left;">

Lasso Regression
</td>

<td style="text-align:left;">

penalty = 0.00356
</td>

<td style="text-align:left;">

NA
</td>

</tr>

<tr>

<td style="text-align:left;">

SVM (RBF)
</td>

<td style="text-align:left;">

cost = 21.5
</td>

<td style="text-align:left;">

rbf_sigma = 0.000318
</td>

</tr>

</tbody>

</table>
