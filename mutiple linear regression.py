import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/강원도 코로나 확진자 현황.csv', encoding='cp949')
df['확진시기'] = [idx for idx in range(len(df['확진시기']))]
df['누적백신접종률(1차)'] = [float(per[:-1])/100 for per in df['누적백신접종률(1차)']]
df['누적백신접종률(2차)'] = [float(per[:-1])/100 for per in df['누적백신접종률(2차)']]
print(df.head(-1))

X = df[[col for col in df if col != '시군명' and col != '확진자수' and col != "누적확진자수"]]
y = df['확진자수']
print(X, y)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

model = LinearRegression()
model.fit(X, y)

today = [[582, 0.1, 0.1]]  # 확산시기, 누적백신접종률(1차), 누적백신접종률(2차)
today_predict = model.predict(today)

y_predict = model.predict(x_test)

print(today_predict)
print(model.coef_, model.intercept_)

plt.scatter(df['누적백신접종률(2차)'], df['확진자수'], alpha=0.5)
plt.show()

print(model.score(x_train, y_train))
