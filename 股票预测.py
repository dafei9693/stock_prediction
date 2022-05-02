import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.preprocessing import StandardScaler  # 标准化数据
from sklearn.model_selection import train_test_split


X_train = pd.read_csv("data.csv", encoding="gbk")
X_train = X_train.values
#X_test = np.array([[4044.3842,4047.5566,4000.9008,4009.7831,4017.5941,26.7901,13814876100,2.10E+11]])

y_train = pd.read_csv("label.csv", encoding='gbk')
y_train = y_train.values

X_train,X_test,y_train,y_test=train_test_split(X_train,y_train)



#y_test = np.array([[1.3432]])

ss = StandardScaler()
X_train_scaled = ss.fit_transform(X_train)
X_test_scaled = ss.transform(X_test)
K = 3
KN = KNeighborsRegressor(n_neighbors=K)
KN.fit(X_train_scaled,y_train)
predictions = KN.predict(X_test_scaled)
print('预测的数据:',predictions)
print('MAE为',mean_absolute_error(y_test,predictions))  # 平均绝对误差
print('MSE为',mean_squared_error(y_test,predictions))  # 均方误差
