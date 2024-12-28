import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv('employee_data.csv')

# print(df)
# print(df.info())
# print(df.describe())

df2 = pd.read_json('employee_data.json')

print(df2)
print(df2.info())
print(df2.describe())


