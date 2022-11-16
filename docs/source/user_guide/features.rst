About training and hyperparameters optimization
==========================================================

During training and hyperparameters optimization the following steps happen:

1. Models are trained with currently optimal hyperparameters on all but last folds from *cv*
2. After training models make prediction on test parts of corresponding validation folds
3. Quality of predictions is assessed with *validation metric*
4. Obtained validation metrics are averaged
5. One more model is trained with currently optimal hyperparameters, but now on test fold.
6. Test model makes predictions on test part of test fold.
7. Validation metrics, trained models and obtained metrics are logged to a server.
8. Average validation metric is used as metric representing quality of current hyperparameters
9. Optuna updates optimal hyperparameters and steps are repeated

