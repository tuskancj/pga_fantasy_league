# Baseline Datagolf Predictions

Valero Texas Open Cut Predictions

![Datagolf.com AUC](images/auc_datagolf.png)

![Datagolf.com Confusion Matrix](images/conf_matrix_datagolf.png)

-   0.66 AUC and F1 Score of 0.61 is the baseline that will try to be improved.

# Data 



# Model 1

-   averaged strokes gained measures from 4 months of prior tournaments, all available rounds
-   golfer random effects logistic regression

![Model 1 AUC](images/auc_model_1.png)

![Model 1 Confusion Matrix](images/conf_matrix_model_1.png)

-   0.60 AUC and F1 Score of 0.69

Model 1's F1 score of 0.69 is higher than Data Golf's F1 Score. However, the AUC is slightly lower at 0.60.

The ratio of golfers predicted to make the cut is heavy at 0.68 which slightly invalidates F1 score since the predicted values are imbalanced. AUC is more reliable, so this model has not improved performance over Data Golf. With that said, 148 golfers is a large number. Combined with the predicted ratio to make the cut, this results in 101 golfers which is way higher than the \~65 standard for Valero.

The threshold can be increased for predicted probability to make the cut to decrease the number of golfers that make the cut.This model produces an F1 score of 0.69 which is a good start as this is better than Data Golf's prediction. However, the AUC is slightly lower at 0.60 (vs 0.66).

The F1 score decreases when the predicted probability threshold is increased from 0.5 to 0.55 which suggests this model is overfit or not ideal. It might make more sense to look at a linear regression model for score after 2 rounds, then select the lowest 65 to 70 predicted scores as the cut line (for ties). This number can be adjusted depending on what the model is used for (e.g. for betting, a tighter threshold to limit losses)

# Current Research




