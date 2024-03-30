import pandas as pd
import numpy as np
import seaborn as sns
df = pd.read_csv("weather.csv", encoding='utf-8', delimiter=';')

# Посмотрим на сами данные
df.head(10)
# и на столбцы
df.info()

# В наборе присутсвует дата как строка, много цифровых значений, представленных в виде строки и очень много незаполненных значений.
# Всё это надо чистить. Начнём с преобразования столбца со временем

# Вначале переименуем его для удобства работы в time
df = df.rename(columns = {'Местное время в Рязани':'time'})
df['time'] = pd.to_datetime(df['time'], format='%d.%m.%Y %H:%M')

# Также можно воспользоваться группировкой по дням, месяцам, годам
df.groupby(pd.Grouper(key="time", freq="D")).mean()

# Столбец N - цифровая информация об облачности, прдстваленная строкой.
# Давайте переведем столбец в число. Прежде всего, посомтрите, какие данные в нём присутствуют
df['N'].unique().tolist()

# Замените столбец на новый, в котором облачость представлена числом с плавающей точкой.
# Строки с nan и неопределённостью удалите из выборки

def set_clounds(value):
    if value == 'Облаков нет.':
        return 0
    if value == '10%  или менее, но не 0':
        return 5
    elif value == '20–30%.':
        return 15
    elif value == '40%.':
        return 40
    elif value == '50%.':
        return 50
    elif value == '60%.':
        return 60
    elif value == '70 – 80%.':
        return 65
    elif value == '90  или более, но не 100%':
        return 95
    elif value == '100%.':
        return 100
    else:
      return np.nan

df['clouds'] = df['N'].map(set_clounds)
df = df.dropna(subset=['clouds'])
df = df.drop(['N'], axis=1)
df.info()