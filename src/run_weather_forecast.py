import mlflow

from cls.rfr_model import RFRModel
from cls.utils import Utils

"""
This program will use the MLflow Model Registry to build a machine learning application that forecasts the 
daily power output of a wind farm. Wind farm power output depends on weather conditions: generally, more energy 
is produced at higher wind speeds. Accordingly, the machine learning models used in program predict power output 
based on weather forecasts with three features: wind direction, wind speed, and air temperature.

It uses altered data from the National WIND Toolkit dataset provided by NREL, which is publicly available and cited as follows:

Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. Overview and Meteorological Validation of the Wind Integration National Dataset Toolkit (Technical Report, NREL/TP-5000-61740). Golden, CO: National Renewable Energy Laboratory.

Draxl, C., B.M. Hodge, A. Clifton, and J. McCaa. 2015. "The Wind Integration National Dataset (WIND) Toolkit." Applied Energy 151: 355366.

Lieberman-Cribbin, W., C. Draxl, and A. Clifton. 2014. Guide to Using the WIND Toolkit Validation Code (Technical Report, NREL/TP-5000-62595). Golden, CO: National Renewable Energy Laboratory.

King, J., A. Clifton, and B.M. Hodge. 2014. Validation of Power Output for the WIND Toolkit (Technical Report, NREL/TP-5D00-61714). Golden, CO: National Renewable Energy Laboratory.
"""

if __name__ == "__main__":
   # Use sqlite:///mlruns.db as the local store for tracking and registery
   mlflow.set_tracking_uri("sqlite:///mlruns.db")

   # Load and print dataset
   csv_path = "data/windfarm_data.csv"

   # Use column 0 (date) as the index
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
   model_name = "PowerForecastingModel"
   for params in params_list:
      rfr = RFRModel.new_instance(params)
      print("Using paramerts={}".format(params))
      runID = rfr.mlflow_run(X_train, y_train, val_x, val_y, model_name, register=True)
      print("MLflow run_id={} completed with MSE={} and RMSE={}".format(runID, rfr.mse, rfr.rsme))

   # Load test data
   score_weather_cvs = "data/score_windfarm_data.csv"
   score_df = Utils.load_data(score_weather_cvs,index_col=0)
   score_df = score_df.drop(columns=["power"])

   # Our JSON payload for scoring the model
   # Use as payload on the REST call to the deployed model
   # on the local host
   print(score_df.to_json(orient="records"))


