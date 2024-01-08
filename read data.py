import os
import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', 50)

train_raw = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2019_ontime.csv')
print("Training size: ", len(train_raw))
display(train_raw.head())

test_raw = pd.read_csv('/kaggle/input/flight-delay-prediction/Jan_2020_ontime.csv')
print("Testing size: ", len(test_raw))
display(test_raw.head())
