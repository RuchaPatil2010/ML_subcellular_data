import pandas as pd 

FILE_LOCATION = "../utils/"

def get_occur_data():
  """
  Reads data from the 'occur.csv' file and returns it as a DataFrame.

  Returns:
      DataFrame: The contents of the 'occur.csv' file.
  """
  return pd.read_csv(FILE_LOCATION + 'occur.csv')
