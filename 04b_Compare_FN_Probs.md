04_Compare_FN_Probs.md
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
# 1. Load the Model Artifacts
rf_output <- readRDS(here("_models", "final_rf_output.rds"))
svm_output <- readRDS(here("_models", "final_svm_output.rds"))

# 2. Extract Prediction Data Frames
# We assume row alignment is preserved due to identical set.seed(8888) usage
rf_preds <- rf_output$preds
svm_preds <- svm_output$preds

# 3. Identify False Negatives
# (Patients who have Malignant cancer (M), but the model predicted Benign (B))
# We find the row indices where this occurred in the Random Forest model
# (Since both models had the same 5 FN, we can just use one to find the indices)
fn_indices <- which(rf_preds$diagnosis == "M" & rf_preds$.pred_class == "B")
fnK_indices <- which(svm_preds$diagnosis == "M" & svm_preds$.pred_class == "B")

union_index <- union(fn_indices, fnK_indices)

# 4. Construct the Comparison Table
missed_cases <- data.frame(
  #Patient_Index = fn_indices,
  #Truth = "Malignant",
  
  Patient_Index = union_index,
  Truth = "Malignant",
  
  # Extract the Probability of Malignancy (.pred_M) for these rows
  # Note: A score of 0.49 is a "close call", a score of 0.10 is a "total miss"
  #RF_Prob_Malignant  = rf_preds$.pred_M[fn_indices],
  #SVM_Prob_Malignant = svm_preds$.pred_M[fn_indices]
  
  RF_FN.RF_Prob_Malignant  = rf_preds$.pred_M[union_index],
  RF_FN.SVM_Prob_Malignant = svm_preds$.pred_M[union_index],
  
  SVM_FN.RF_Prob_Malignant  = rf_preds$.pred_M[union_index],
  SVM_FN.SVM_Prob_Malignant = svm_preds$.pred_M[union_index]
  
)


missed_cases
```

    ##   Patient_Index     Truth RF_FN.RF_Prob_Malignant RF_FN.SVM_Prob_Malignant
    ## 1            18 Malignant               0.2155000              0.156560440
    ## 2            23 Malignant               0.4932000              0.435289763
    ## 3            32 Malignant               0.4523167              0.947403046
    ## 4            38 Malignant               0.4616500              0.811117809
    ## 5            72 Malignant               0.0446000              0.001472448
    ## 6           135 Malignant               0.4952500              0.833362822
    ## 7             5 Malignant               0.5184833              0.439595109
    ## 8            55 Malignant               0.8908500              0.183578065
    ##   SVM_FN.RF_Prob_Malignant SVM_FN.SVM_Prob_Malignant
    ## 1                0.2155000               0.156560440
    ## 2                0.4932000               0.435289763
    ## 3                0.4523167               0.947403046
    ## 4                0.4616500               0.811117809
    ## 5                0.0446000               0.001472448
    ## 6                0.4952500               0.833362822
    ## 7                0.5184833               0.439595109
    ## 8                0.8908500               0.183578065

``` r
# 1. Load the Model Artifacts
rf_output <- readRDS(here("_models", "final_rf_output.rds"))
svm_output <- readRDS(here("_models", "final_svm_output.rds"))

# 2. Extract Prediction Data Frames
# We assume row alignment is preserved due to identical set.seed(8888) usage
rf_preds <- rf_output$preds
svm_preds <- svm_output$preds

# 3. Identify False Negatives (Missed Cancers)
# Truth is Malignant (M), but Predicted Class was Benign (B)
rf_missed_indices <- which(rf_preds$diagnosis == "M" & rf_preds$.pred_class == "B")
svm_missed_indices <- which(svm_preds$diagnosis == "M" & svm_preds$.pred_class == "B")

# Find the UNION of all difficult cases (missed by at least one model)
difficult_cases_indices <- union(rf_missed_indices, svm_missed_indices)

# 4. Construct the Comparison Table
missed_cases <- data.frame(
  Patient_Index = difficult_cases_indices,
  Truth = "Malignant",
  
  # Extract Probabilities
  RF_Prob_M  = rf_preds$.pred_M[difficult_cases_indices],
  SVM_Prob_M = svm_preds$.pred_M[difficult_cases_indices]
)

# 5. Categorize the Failure Mode
# We classify each case as "Missed by Both", "RF Only", or "SVM Only"
missed_cases <- missed_cases %>%
  mutate(
    RF_Status = ifelse(RF_Prob_M < 0.5, "MISSED", "Caught"),
    SVM_Status = ifelse(SVM_Prob_M < 0.5, "MISSED", "Caught"),
    
    Category = case_when(
      RF_Status == "MISSED" & SVM_Status == "MISSED" ~ "Hard Case (Both Missed)",
      RF_Status == "MISSED" & SVM_Status == "Caught" ~ "RF Failed / SVM Success",
      RF_Status == "Caught" & SVM_Status == "MISSED" ~ "RF Success / SVM Failed"
    ),
    
    # Who was more confident? (Higher prob = 'less wrong')
    Stronger_Model = ifelse(RF_Prob_M > SVM_Prob_M, "Random Forest", "SVM")
  ) %>%
  arrange(Category)

# 6. Print the Table
print("Analysis of Difficult Cases (False Negatives by at least one model):")
```

    ## [1] "Analysis of Difficult Cases (False Negatives by at least one model):"

``` r
knitr::kable(missed_cases, digits = 3, caption = "Comparison of Models on Difficult Cases")
```

| Patient_Index | Truth | RF_Prob_M | SVM_Prob_M | RF_Status | SVM_Status | Category | Stronger_Model |
|---:|:---|---:|---:|:---|:---|:---|:---|
| 18 | Malignant | 0.215 | 0.157 | MISSED | MISSED | Hard Case (Both Missed) | Random Forest |
| 23 | Malignant | 0.493 | 0.435 | MISSED | MISSED | Hard Case (Both Missed) | Random Forest |
| 72 | Malignant | 0.045 | 0.001 | MISSED | MISSED | Hard Case (Both Missed) | Random Forest |
| 32 | Malignant | 0.452 | 0.947 | MISSED | Caught | RF Failed / SVM Success | SVM |
| 38 | Malignant | 0.462 | 0.811 | MISSED | Caught | RF Failed / SVM Success | SVM |
| 135 | Malignant | 0.495 | 0.833 | MISSED | Caught | RF Failed / SVM Success | SVM |
| 5 | Malignant | 0.518 | 0.440 | Caught | MISSED | RF Success / SVM Failed | Random Forest |
| 55 | Malignant | 0.891 | 0.184 | Caught | MISSED | RF Success / SVM Failed | Random Forest |

Comparison of Models on Difficult Cases

``` r
# 7. Visualization of the Discrepancies
missed_cases_long <- missed_cases %>%
  select(Patient_Index, RF_Prob_M, SVM_Prob_M, Category) %>%
  pivot_longer(cols = c(RF_Prob_M, SVM_Prob_M), 
               names_to = "Model", 
               values_to = "Probability")

ggplot(missed_cases_long, aes(x = as.factor(Patient_Index), y = Probability, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_hline(yintercept = 0.5, linetype = "dashed", color = "black") +
  annotate("text", x = 0.1, y = 0.54, label = "Threshold", color = "black", hjust=0, size=3) +
  facet_wrap(~Category, scales = "free_x") +
  labs(
    title = "Ensemble Disagreement on Edge Cases",
    #subtitle = "Cases where one model succeeds while the other fails highlight the benefit of using both.",
    x = "Patient Index",
    y = "Prob. of Malignancy"
  ) +
  #scale_fill_discrete(palette = "greys") +
  scale_fill_manual(values = c("RF_Prob_M" = "grey50", "SVM_Prob_M" = "grey80")) +
  theme_minimal() +
  theme(legend.position = "bottom")
```

![](_plot_images/unnamed-chunk-3-1.png)<!-- -->
