import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Salary_Data.csv')

x=dataset.iloc[:,[0]].values
y=dataset.iloc[:,[1]].values