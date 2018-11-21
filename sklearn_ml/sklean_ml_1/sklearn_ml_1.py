import os
import pandas as pd
import numpy as np
HOUSING_PATH = "datasets/housing"
def load_housing_data(housing_path = HOUSING_PATH):
    csv_path = HOUSING_PATH+"/housing.csv"
    return pd.read_csv(csv_path)

def split_train_test(data,test_ratio):
    shuffler_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffler_indices[:test_set_size]
    train_indices = shuffler_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]


housing = load_housing_data()
# print(housing[:5])
# print(housing.info())
print(housing["ocean_proximity"].value_counts())
import matplotlib.pyplot as plt
# housing.hist(bins=50,figsize=(20,15))
# plt.show()

# #创建测试集



train_set, test_set = split_train_test(housing,0.2)
print(len(train_set),"train +" ,len(test_set),"test")