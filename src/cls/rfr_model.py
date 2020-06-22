import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

class RFRModel():
   def __init__(self, params={}):
      self.rf = RandomForestRegressor(**params)
      self.params = params
      self._mse = None
      self._rsme = None

   @classmethod
   def new_instance(cls, params={}):
      return cls(params)

   @property
   def model(self):
      return self.rf

   @property
   def mse(self):
      return self._mse

   @mse.setter
   def mse(self, value):
      self._mse = value

   @property
   def rsme(self):
      return self._rsme

   @rsme.setter
   def rsme(self, value):
      self._rsme = value

   def mlflow_run(self, X_train, y_train, val_x, val_y, model_name,
                  run_name="Random Forest Regressor: Power Forecasting Model",
                  register=False, verbose=False):
      with mlflow.start_run(run_name=run_name) as run:
         # Log all parameters
         mlflow.log_params(self.params)

         # Train and fit the model
         self.rf.fit(X_train, y_train)
         y_pred = self.rf.predict(val_x)

         # Compute metrics
         self._mse = mean_squared_error(y_pred, val_y)
         self._rsme = np.sqrt(self._mse)

         if verbose:
            print("Validation MSE: %d" % self._mse)
            print("Validation RMSE: %d" % self._rsme)

         # log params and metrics
         mlflow.log_params(self.params)
         mlflow.log_metric("mse", self._mse)
         mlflow.log_metric("rmse", self._rsme)

         # Specify the `registered_model_name` parameter of the
         # function to register the model with the Model Registry. This automatically
         # creates a new model version for each new run
         mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="sklearn-model",
            registered_model_name=model_name) if register else mlflow.sklearn.log_model(
               sk_model=self.model,
               artifact_path="sklearn-model")

         run_id = run.info.run_id

      return run_id
