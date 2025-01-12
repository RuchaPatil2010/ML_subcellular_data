import pandas as pd 

FILE_LOCATION = "../utils/"

def get_occur_data():
  """
  Reads data from the 'occur.csv' file and returns it as a DataFrame.

  Returns:
      DataFrame: The contents of the 'occur.csv' file.
  """
  return pd.read_csv(FILE_LOCATION + 'occur.csv')

def read_data():
  """
  Reads multiple data files from the specified file location.

  Returns:
    tuple: A tuple containing the following DataFrames:
        - g_data (DataFrame): Data from 'g_data.csv'.
        - occur_data (DataFrame): Data from 'occur.csv'.
        - comp_occur_data (DataFrame): Data from 'comp_occur.csv'.
        - attributes_data (DataFrame): Data from 'attributes.csv', with the 
                                       second row as the header.
  """
  g_data = pd.read_csv(FILE_LOCATION + 'g_data.csv')
  occur_data = pd.read_csv(FILE_LOCATION + 'occur.csv')
  comp_occur_data = pd.read_csv(FILE_LOCATION + 'comp_occur.csv')
  attributes_data = pd.read_csv(FILE_LOCATION + 'attributes.csv', header=1)

  return (g_data, occur_data, comp_occur_data, attributes_data)

def clean_data():
  """
  Cleans and processes data for gram-positive datasets and other related files.

  Steps performed:
    - Drops columns with all NaN values.
    - Removes duplicate columns.
    - Sets appropriate headers and reassigns the existing header to the first row.
    - Adjusts indexing for proper data alignment.
  
  Returns:
    DataFrame: The cleaned version of `g_data`.
  """
  # Read all datasets
  g_data, occur_data, comp_occur_data, attributes_data = read_data()

  # Clean and preprocess `g_data`
  df_g = g_data
  # Drop columns with all NaN values
  g_data_cleaned = df_g.dropna(axis=1, how='all')

  # Drop duplicate columns based on their content
  g_data_cleaned = g_data_cleaned.T.drop_duplicates().T

  # Copy the current headers for reassignment
  copied_headers = g_data_cleaned.columns.copy()

  # Define new column headers
  columns_n_data = ['Number', 'Label', 'Name', 'Protein'] 
  g_data_cleaned.columns = columns_n_data

  # Move the original header to the first row
  g_data_cleaned.loc[-1] = copied_headers
  g_data_cleaned.index = g_data_cleaned.index + 1
  g_data_cleaned.sort_index(inplace=True)

  return g_data_cleaned
