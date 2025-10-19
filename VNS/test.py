import pandas as pd

row_i = pd.read_csv('rotated_data.csv', skiprows=1, nrows=1, header=None).iloc[0]
print(row_i[1])