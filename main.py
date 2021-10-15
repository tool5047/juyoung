import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/강원도 코로나 확진자 현황.csv', encoding='cp949')

df['확진시기'] = df['확진시기'].index
df['누적백신접종률(1차)'] = df['누적백신접종률(1차)'].apply(lambda x: float(x[:-1]) / 100)
df['누적백신접종률(2차)'] = df['누적백신접종률(2차)'].apply(lambda x: float(x[:-1]) / 100)

X = df[[col for col in df if col != '시군명' and col != '확진자수']]
y = df['확진자수']

model = LinearRegression()
model.fit(X, y)

today = [[571, 1281, 0.39, 0.145]]  # 확산시기, 누적확진자수, 누적백신접종률(1차), 누적백신접종률(2차)
today_predict = model.predict(today)
print(today_predict)

