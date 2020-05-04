import pandas as pd
import tempfile

class Utils:
   @staticmethod
   def load_data(path, index_col=0):
      df = pd.read_csv(path, index_col=0)
      return df

   @staticmethod
   def get_training_data(df):
      training_data = pd.DataFrame(df["2014-01-01":"2018-01-01"])
      X = training_data.drop(columns="power")
      y = training_data["power"]
      return X, y

   @staticmethod
   def get_validation_data(df):
      validation_data = pd.DataFrame(df["2018-01-01":"2019-01-01"])
      X = validation_data.drop(columns="power")
      y = validation_data["power"]
      return X, y

   @staticmethod
   def get_temporary_directory_path(prefix, suffix):
      """
      Get a temporary directory and files for artifacts
      :param prefix: name of the file
      :param suffix: .csv, .txt, .png etc
      :return: object to tempfile.
      """

      temp = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix)
      return temp

   @staticmethod
   def print_pandas_dataset(d, n=5):
      """
      Given a Pandas dataFrame show the dimensions sizes
      :param d: Pandas dataFrame
      :return: None
      """
      print("rows = %d; columns=%d" % (d.shape[0], d.shape[1]))
      print(d.head(n))
