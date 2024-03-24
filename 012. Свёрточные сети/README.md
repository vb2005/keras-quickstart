# Свёрточные сети. Классификация изображений
В данной работе рассматривается задача классификации изображений. Для сокращения размерности признаков, работа с изображениями осуществляется при помощи окон свёртки. Каскад свёрточных блоков в сочетании с прореживаем дает на выходе одномерное пространство признаков (Обычно, небольшой размерности). Далее, с этим пространством можно работать как с простым перцептроном, как в предыдущей работе

## Подготовка данных
Для эксперимента мы будем использовать классический набор данных cats/dogs. Скачать архив можно по ссылке: https://www.kaggle.com/c/dogs-vs-cats/code
Прочитаем все изображения в массив X, а в Y сформируем ID-класса
``` python
import os, tqdm, cv2, sklearn
import numpy as np

# Массивы данных
X = []
Y = []

# Размер изображения на входе
size = 96

# Количество классов
classes = 2

# Режим цвета
color_mode = cv2.IMREAD_COLOR #COLOR GRAYSCALE

# Прочитаем все файлы из каталога
cats_files = [f for f in os.listdir('C:/Python/cats-dogs/train') if f.endswith(('.png', '.jpg', '.jpeg', '.gif')) and f.startswith(('cat')) ]
dog_files = [f for f in os.listdir('C:/Python/cats-dogs/train') if f.endswith(('.png', '.jpg', '.jpeg', '.gif')) and f.startswith(('dog')) ]

for file in tqdm.tqdm(cats_files):
  img = cv2.resize(cv2.imread(os.path.join('C:/Python/cats-dogs/train', file), color_mode), (size, size))
  X.append(list(np.array(img)))
  Y.append([0])

for file in tqdm.tqdm(dog_files):
  img = cv2.resize(cv2.imread(os.path.join('C:/Python/cats-dogs/train', file), color_mode), (size, size))
  X.append(list(np.array(img)))
  Y.append([1])


# Преобразуем данные к Numpy-массиву
X = np.array(X)
Y = np.array(Y)

# и перемешаем
X, Y = sklearn.utils.shuffle(X, Y)
```

``` python
# Проверить данные можно при помощи
X[60]
#Y[60]
```
Теперь данные в массиве Y надо перевести в формат one-hot. Данный код поможет преобразовать массив с индексами классов в формат one-hot, так как нейронная сеть должна возратить не номер класса, а вероятности.
``` python
One-hot encoding
# [0] => [1, 0, ..., 0]
# [1] => [0, 1, ..., 0]
# ...
# [n] => [0, 0, ..., 1]
from keras.utils import to_categorical
import tensorflow as tf
Y_en = tf.keras.utils.to_categorical(Y, num_classes=2, dtype='float32')
```

## Построение модели

Я предлагаю исследовать разные виды моделей и снять основные их характеристики. Далее представлена архитектура свёрточной сети [96, 96, 3] в [1]. Её можно модифицировать для получения наилучших результатов.
``` python
import keras
from keras.optimizers import *
from keras.models import Model, Sequential
from keras.layers import *

# Конфигурируем модель сети
def build_model(size, classes):
    # Модель последовательная...
    model = Sequential()

    # размером SIZE*SIZE*3. В первом блоке свёртка окном 3*3 разных окон 36 штук...
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(size, size, 3)))
    # Активация ReLU
    model.add(Activation("elu"))
    # Подвыборка квадратом 3*3 максимума из квадрата
    model.add(MaxPooling2D(pool_size=(3, 3)))
    model.add(Dropout(0.15))

    # [32, size/3, size/3]
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    # [32, size/6, size/6]
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    # [64, size/12, size/12]
    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    # [128, size/24, size/24]
    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    # [64, size/48, size/48]
    model.add(Conv2D(512, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # [32, size/96, size/96]
    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("elu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax")) # Если будет несколько классов, то softmax
    #model.add(Activation("sigmoid"))

    # возвращаем модель
    return model
```

``` python
# Строим модель и запускаем обучение
model = build_model(size, classes)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[ 'accuracy', 'mse', 'mae'])
```
Запуск обучения
``` python
history = model.fit(X,Y_en,epochs=40, batch_size=256, validation_split=0.2)
```
Оцениваем результат работы сети традиционным графиком
``` python
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь')
plt.ylabel('Потери (меньше - лучше)')
plt.xlabel('Число эпох')
plt.legend(['Обучающая', 'Тестовая'], loc='upper left')
plt.show()
```
## Тестирование сети на пользовательском изображении
``` python
path = 'C:/Python/Test.jpg'
img = cv2.resize(cv2.imread(path, color_mode), (size, size))
img = np.array([img])
pred = model.predict(img)
print(pred)
```
## Самостоятельное задание
В рамках группы проведите исследование параметров, влияющих на сходимость и переобучение. Для исследования возьмите следующие параметры:
1. Вид активационной функции (linear, relu, elu, silu, leaky_relu, sigmoid)
2. Величина Dropout (10%, 25%, 50%, 75%, 90%)
3. Количество свёрток на разных слоях
4. Разные виды оптимизаторов (Adam, SGD, RMSProp)
5. Разный размер batch_size (1, 8, 32, 128)
6. Разный размер входного изображения
7. Разное число каналов (RGB, HSV, Grayscale)

В качестве анализируемых величин выберем (после 10 эпох):
1. На какой эпохе accuray стал выше 90% на train и на val
2. Какое наименьшее значение loss фиксировалось на train и на val
3. Время просчёта одного изображения
4. Построить графики loss_train, loss_val для своих кейсов
5. Определить на какой эпохе началось явное переобучение, и начиналось ли оно вовсе для выбранных кейсов

## Контрольные вопросы
1. Что такое свёртка?
2. Что такое подвыборка?
3. Опишите схему алгоритма или описание на любом языке программирования операции свёртки
4. Какие функции активации применяются в задачах обработки изображений?
5. Почему подвыборки стараются делать малым окном?
6. Что такое преобразование One-hot?
7. Как работает функция binary_crossentropy?
8. Почему не стоит применять MSE для задач классификации?
9. Как работает функция активации softmax?
10. Что такое прореживание? Зачем оно нужно?
