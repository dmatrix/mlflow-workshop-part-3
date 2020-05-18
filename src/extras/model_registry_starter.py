import os
import shutil
import pprint

from random import random, randint
import numpy as np
import matplotlib.pyplot as plt
import mlflow.sklearn
import mlflow
from sklearn.ensemble import RandomForestRegressor
from mlflow.tracking import MlflowClient
from cls.utils import Utils

import warnings

RANDOM_TEXTS = ["Looks, like I logged to the local store!",
                "This Mlflow thingy is way cool!",
                "Try the new MLflow Registry component",
                "Hello, This is part-3 Nice to virtual now!"]


def gen_random_text():
    """
    Random text for messages
    :return: return string message text
    """
    return RANDOM_TEXTS[randint(0, 3)]


def gen_random_scatter_plots(npoints):
    """
    Random scatter plot geneator to save as an MLflow artifact in an MLflow experiment
    run
    :param npoints: number of data points for the scatter plot
    :return: return a tuple
    """
    data = {'a': np.arange(npoints),
            'c': np.random.randint(0, npoints, npoints),
            'd': np.random.randn(npoints)}
    data['b'] = data['a'] + 10 * np.random.randn(npoints)
    data['d'] = np.abs(data['d']) * 100

    plt.clf()
    fig, ax = plt.subplots()

    plt.scatter('a', 'b', c='c', s='d', data=data)
    plt.xlabel('Entry for data in A')
    plt.ylabel('entry for data in B')
    return (fig, ax)


def print_experiment_details(experiment_id, run_id):
    """
    Method to print experiment run info and a specific run details
    :param experiment_id: MLflow experiment ID
    :param run_id: MLflow run ID within an experiment
    :return: none
    """
    print("Finished MLflow Run with run_id {} and experiment_id {}".format(run_id, experiment_id))

    # Use MlflowClient API to list experiments and run info
    client = MlflowClient()
    print("=" * 80)
    # Get a list of all experiments
    print("List of all Experiments")
    print("=" * 80)
    [print(pprint.pprint(dict(exp), indent=4))
     for exp in client.list_experiments()]
    print("=" * 80)
    print(f"List Run info for run_id={run_id}")
    print(pprint.pprint(dict(mlflow.get_run(run_id))))


def mlflow_run(params, run_name="LOCAL_REGISTRY"):
    """
    Function to start a run within a Default experiment
    :param params: ters used for the run, such as arguments to RandomForest scikit-learn
    :param run_name: label for the name of the run
    :return: experiment ID and run ID
    """

    with mlflow.start_run(run_name=run_name) as run:
        # Get the run and experimentid

        run_id = run.info.run_uuid
        experiment_id = run.info.experiment_id

        # Create our model type instance
        sk_learn_rfr = RandomForestRegressor(params)

        # Log params and metrics using the MLflow APIs
        mlflow.log_params(params)
        mlflow.log_metric("metric_1", random())
        mlflow.log_metric("metric_2", random() + 1)
        mlflow.log_metric("metric_3", random() + 2)

        # Set the notes for experiment and the Runs
        MlflowClient().set_experiment_tag(experiment_id,
                                          "mlflow.note.content",
                                          "This is experiment for getting started with MLflow ...")
        MlflowClient().set_tag(run_id,
                               "mlflow.note.content",
                               "This Run is for getting started with MLflow Model Registry ...")

        # Log the model at the same time
        mlflow.sklearn.log_model(
            sk_model=sk_learn_rfr,
            artifact_path="sklearn-model")

        # Create sample message artifact
        if not os.path.exists("messages"):
            os.makedirs("messages")
        with open("messages/message.txt", "w") as f:
            f.write(gen_random_text())

        mlflow.log_artifacts("messages")
        shutil.rmtree('messages')

        # Create scatter random plot artifacts file and log artifact
        for npoints in range(55, 70, 5):
            fig, ex = gen_random_scatter_plots(npoints)
            temp_file_name = Utils.get_temporary_directory_path("scatter-plot-", ".png")
            temp_name = temp_file_name.name
            try:
                fig.savefig(temp_name)
                mlflow.log_artifact(temp_name, "scatter_plots")
            finally:
                temp_file_name.close()  # Delete the temp file

        return (run_id, experiment_id)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print(mlflow.__version__)

    # Use sqlite:///mlruns.db as the local store
    local_registry = "sqlite:///mlruns.db"
    print(f"Running local model registry={local_registry}")
    mlflow.set_tracking_uri(local_registry)
    model_name = "sk-learn-random-forest"
    params = {"n_estimators": 4, "random_state": 42}
    run_id, experiment_id = mlflow_run(params)

    # Print experiment and run details
    print_experiment_details(experiment_id, run_id)
    # launch MLflow ui
    # mlflow ui --backend-store-uri sqlite:///mlruns.db
