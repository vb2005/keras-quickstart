import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Activation, Dense
from matplotlib import pyplot as plt

df = pd.read_csv('Arrhythmia.csv', sep=";",header=None)
data = df.to_numpy()

for i in range(452):
  if data[i,276] <= 1:
    data[i,276] = 0
  else:
    data[i,276] = 1
	
X = data[:,:276]
Y = data[:,276]

model = Sequential()

# Первый слой - 274 нейрона (по размеру входного вектора). Далее данные приводятся при помощи функции активации Sigmoid
model.add(Dense(274,"sigmoid"))

# Второй слой и последующие задают глубину обучения и количество весовых коэффициентов.
# Далеко не всегда разумно бесконечно наращивать глубину сети. Иногда сеть и из 2х слоёв
# Справляется с указанной задачей
model.add(Dense(200,"exponential"))
model.add(Dense(200,"sigmoid"))

# Выходной слой с 1 значением. У нас это индикатор наличия или отсутствия заболевания
model.add(Dense(1,"sigmoid"))

model.compile("Adam","MSE")            # Сборка графа. Указание оптимизатора и функции потерь
history = model.fit(                   # Результат обучения сохраняется в переменную history
    X,Y,                               # Обучающая выборка
    batch_size=32,                     # Размер посылки
    epochs=50,                         # Число эпох. Используем 50.
    validation_split=0.2)              # Выборка для валидации
	
# Визуализация графика
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь')
plt.ylabel('Потери (меньше - лучше)')
plt.xlabel('Число эпох')
plt.legend(['Обучающая', 'Тестовая'], loc='upper left')
plt.show()

# Ручной просчёт точности работы нейронной сети:

# Получаем предсказанные на основании нейронной сети значения
pred_Y = model.predict(X).ravel()

# Вычисляем сумму модулей поэлементной разницы двух массивов и делим на количество примеров
difference = pow((pred_Y - Y),2).sum() / len(pred_Y)

# Получаем итоговую точность работы на указанных примерах. Больше - лучше
print('Точнось работы:', (100-difference*100), '%')