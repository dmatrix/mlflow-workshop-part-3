import mlflow

from cls.rfr_model import RFRModel
from cls.utils import Utils

if __name__ == "__main__":
   # Use sqlite:///mlruns.db as the local store for tracking and registery
   mlflow.set_tracking_uri("sqlite:///mlruns.db")

   # Load and print dataset
   csv_path = "data/windfarm_data.csv"
   wind_farm_data = Utils.load_data(csv_path, index_col=0)
   Utils.print_pandas_dataset(wind_farm_data)

   # Get Validation data
   X_train, y_train = Utils.get_training_data(wind_farm_data)
   val_x, val_y = Utils.get_validation_data(wind_farm_data)

   # Train, fit and register our model
   params_list = [
      {"n_estimators": 100},
      {"n_estimators": 200},
      {"n_estimators": 300}]

   # Iterate over few different tuning parameters
   model_name = "SKLearnWeatherForestModel"
   for params in params_list:
      rfr = RFRModel.new_instance(params)
      print("Using paramerts={}".format(params))
      runID = rfr.mlflow_run(X_train, y_train, val_x, val_y, model_name)
      print("MLflow run_id={} completed with MSE={} and RMSE={}".format(runID, rfr.mse, rfr.rsme))


