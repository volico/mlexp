# MLexp

MLexp is the framework to automate model training, hyperparameters 
optimization and logging results to a server, which allows to speed up 
hypothesis testing and helps to maintain the reproducibility of experiments.

## Installation

### Installation with pip

```bash
$ pip install mlexp
```

To support training on pytorch models gpu, before installation of MLexps you 
have to install torch of version compatible with installed pytorch_lightning 
with support of gpu.

## Documentation

Documentation is hosted on [Read the Docs](https://mlexp.readthedocs.io/en/latest/)

## Simple example


### Train model, optimize hyperparameters and log results

In this example we will train lightgbm model, optimize hyperparameters
and log results to mlflow server.

First of all, we have to start mlflow server. Run the following command
in your console.

```bash
$ mlflow server --port 5000
```

Running this command will start mlflow server on port 5000 in the
current directory.

Now we should define function which returns set of parameters.

``` {.python}
def params_func(trial):
    return(
        {
            'model_params': {'boosting': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
                             'feature_fraction': trial.suggest_float('feature_fraction', 0.01, 1),
                             'min_child_samples': trial.suggest_int('min_child_samples', 2, 256),
                             'n_estimators': 500},
            'lgb_data_set_params': {}
        }
    )
```

According to this function:

-   *boosting* will be chosen from ['gbdt', 'dart', 'goss']
    during hyperparameter optimization
-   *feature_fraction* will be chosen from uniform distribution from
    0.01 to 1 during hyperparameter optimization
-   *min_child_samples* will be chosen integers from 2 to 256 during
    hyperparameter optimization
-   no parameters will be passed to lightgbm.Dataset, except for *data*
    and *label*, which are passed from training data automatically

Now we can use python API of mlexp.trainers.LgbTrainer to train model,
optimize hyperparameters and log results to mlflow server.

``` {.python}
from mlexp.trainers import LgbTrainer
from sklearn.metrics import mean_squared_error

# Initialise trainer object
trainer = LgbTrainer(
    # MSE will be used as validation metric
    validation_metric = mean_squared_error,
    # MSE should be minimised during hyperparameters optimization
    direction = 'minimize',
    # Before logging to server files will be saved to /home/logged_files
    saved_files_path = r'/home/logged_files',
    # During training model on test fold n_estimators will be set to the mean n_estimators on validation_folds
    use_average_n_estimators_on_test_fold = True,
    # During hyperparameters' optimization, mean metric on validation fold will be optimized
    optimization_metric = 'metric_mean_cv')

# Initialise mlflow run
trainer.init_run(
    # Init run on mlflow server
    logging_server = 'mlflow',
    # Run will be started in experiment 'example_exp'
    experiment_name = 'example_exp',
    # URI of mlflow server (it will be printed in console after starting mlflow server)
    tracking_uri = 'http://127.0.0.1:5000/',
    # Let's set run_name to 'Example. LGBM' and add tag Hypothesis = 1
    mlflow_run_params = {'run_name': 'Example. LGBM', 'tags': {'Hypothesis': '1'}},
    # Let's also log current_data_config.txt to mlflow server
    upload_files = ['/home/current_data_config.txt'])

# Hyperparameters' sampler
sampler = optuna.samplers.TPESampler(seed = 1234)

# Start model training, hyperparameters optimization and logging results to mlflow server
trainer.train(
    X = X,
    y = y,
    cv = cv_list,
    # Will optimize hyperparameters during 20 iterations
    n_trials = 20,
    params_func = params_func,
    sampler = sampler)
```

### Model inference

Let's get trained model, parameters and hyperparameters from mlflow
server with the help of mlexp.inference.LgbInference

``` {.python}
from mlexp.inference import LgbInference

# Initialize inference object
inference = LgbInference(
    # Let's download models and aprameters to /home/downloaded_files
    downloaded_files_path = '/home/downloaded_files',
    inference_server_params={
    'tracking_uri': 'http://127.0.0.1:5000/',
    'run_id': '1325ca558ec14277b0f39b0f8134d17e'},
    server='mlflow')
# Download params and model
inference.get_params_model(
    # Get parameters and models from best step of hyperparameters optimization (with minimal MSE)
    step = 'best',
    # Get model for test fold
    fold_num = 'test',
    # Get trained model
    trained_model = True)
```

After running this code we will get dictionary with downloaded
parameters and trained model.

## Contributing

For contributing guidelines check [CONTRIBUTING.md](CONTRIBUTING.md)