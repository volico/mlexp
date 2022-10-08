.. _ParamsFunc:

About params_func
==========================

**params_func** is used to set hyperparameters for training and to set a set of hyperparameters for optimization.

**params_func** have to accept only 1 argument and return dictionary of hyperparameters,
dictionary content depends on the type of the model you are training.

params_func input
##########################

**params_func** should accept only 1 argument, during training
`optuna.trial.Trial <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial>`_
object will be passed to this argument. You can use this object for hyperparameters optimization.

For instance, `optuna.trial.Trial <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial>`_
has method *.suggest_float* which is used for finding optimal value of hyperparameter of float type.
All methods of optuna.trial.Trial you can find in
`optuna documentation <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html#optuna.trial.Trial>`_.

Example:

.. code-block:: python

    def params_func(trial):
        return(
            {
                'model_params': {'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
                                 'feature_fraction': trial.suggest_float('feature_fraction', 0.01, 1),
                                 'min_child_samples': trial.suggest_int('min_child_samples', 2, 256)}

            }
        )


During hyperparameters optimization:

- *boosting* will be chosen from ['gbdt', 'dart', 'goss'] during hyperparameter optimization
- *feature_fraction* will be chosen from uniform distribution from 0.01 to 1 during hyperparameter optimization
- *min_child_samples* will be chosen itegers from 2 to 256 during hyperparameter optimization

params_func for LGB model
##########################

In case you use mlexp.trainer.lgb_trainer, **params_func** have to return a dictionary with the following keys:

- **model_params** should contain a dictionary with `hyperparameters for lightgbm model <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_, except for data set parameters
- **lgb_data_set_params** should contain a dictionary with `parameters for lightgbm.Dataset <https://lightgbm.readthedocs.io/en/latest/Parameters.html>`_, except for *data* and *label*, which are passed from training data automatically

Example:

.. code-block:: python

    def params_func(trial):
        return(
            {
                'model_params': {'objective': trial.suggest_categorical('objective', ['huber', 'fair', 'l2', 'l1', 'mape']),
                                 'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
                                 'n_jobs': -1,
                                 'n_estimators': 500,
                                 'random_state': random_state,
                                 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.01, 1),
                                 'feature_fraction': trial.suggest_float('feature_fraction', 0.01, 1),
                                 'min_child_samples': trial.suggest_int('min_child_samples', 2, 256),
                                 'num_leaves': trial.suggest_int('num_leaves', 2, 256),
                                 'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.5)},
                'lgb_data_set_params': {'feature_name': ['first_feature', 'second_feature'],
                                        'categorical_feature': ['first_feature']},

            }
        )

params_func for scikit-learn model
####################################################

In case you use mlexp.trainer.sklearn_trainer, **params_func** have to return a dictionary with the following keys:

- **model_params** should contain a dictionary with hyperparameters for scikit-learn model

Example for
`sklearn.linear_model.Ridge <https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html>`_:

.. code-block:: python

    def params_func(trial):
        return(
            {
                'model_params': {'random_state': random_state,
                                 'alpha': trial.suggest_float('alpha', 0.01, 1)},
            }
        )

params_func for pytorch-lightning neural network
####################################################

In case you use mlexp.trainer.torch_trainer, **params_func** have to return a dictionary with the following keys:

- **model_params** should contain a dictionary with hyperparameters for your *nn_model* class from module passed to *nn_model_module* parameter
- **EarlyStopping_params** should contain a dictionary with parameters for `pytorch_lightning.callbacks.EarlyStopping <https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.callbacks.EarlyStopping.html#pytorch_lightning.callbacks.EarlyStopping>`_
- **trainer_params** should contain a dictionary with parameters for `pytorch_lightning.Trainer <https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api>`_
- **data_loaders_params** should contain a dictionary with hyperparameters for your *train_val_data_loaders* function from module passed to *data_loaders_module* parameter


Example:

.. code-block:: python

    def params_func(trial):
        return(
            {
                'model_params': {'optimizer': 'SGD',
                                 'objective': 'cross_entropy',
                                 'lr': trial.suggest_float('lr', 0.001, 1),
                                 'vocab_size': 500,
                                 'embedding_size': 150,
                                 'weight_decay': 0.05},
                'trainer_params': {'min_epochs': 2,
                                   'max_epochs': 10,
                                   'num_sanity_val_steps': 0,
                                   'progress_bar_refresh_rate': 0,
                                   'gpus': 1},
                'data_loaders_params': {'return_wide_format': True},
                'EarlyStopping_params': {'min_delta': 0.001,
                                         'patience': 3,
                                         'monitor': 'validation'}
            }
        )