# Блок 1
## Загрузка набора данных и извлечение изображений из архива
Для обучения сети Вы можете использовать собственные наборы данных. Подготовьте по 300-500 изображений на каждый класс. Либо Вы можете использовать готовый набор данных с freecodecamp. Для этого скачайте его по ссылке и распакуйте в рабочий каталог.
https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip

# Блок 2
## Импорт необходимых библиотек
Если Вы еще не устанавливали Python и библиотеки, обратитесь к работе №0 для установки всего необходимого на Ваш ПК. 

``` python
import cv2, sklearn, os, keras
import numpy as np
import tensorflow as tf
from keras.optimizers import *
from keras.models import Model, Sequential
from keras.layers import *
from keras.utils import *
```

# Блок 3
## Объявление констант

``` python
# Размер входного изображения
size = 256

# Папка с обучающей выборкой
train_dir = "C:/Python/cats_and_dogs/train"

# Цвет изображения
CLR = cv2.IMREAD_COLOR

# Количество классов
classes = 2
```

# Блок 4
## Функция загрузки изображений
Все изображения из каталога загружаются в RAM в виде массивов X и Y. Массив X представляет собой изображения [img_count, size, size, 3]. Массив Y - индексы классов [img_count]


``` python
def create_train_data(train_dir, size, color_mode):
  # Массивы X и Y
  X = []
  Y = []

  #Индекс класса
  pp = 0

  # По всем классам(папкам), и по всем файлам в них
  for p1 in sorted(os.listdir(train_dir)):
    if not p1.startswith('.'):
      for img in os.listdir(os.path.join(train_dir, p1)):
            # Формируем путь к файлу
            path = os.path.join(os.path.join(train_dir, p1), img)
            # Изменяем размер изображения и цветовое пространство
            img = cv2.resize(cv2.imread(path, color_mode), (size, size))
            # Добавляем картинку в обучающую выборку
            X.append(list(np.array(img)))
            # А также её индекс
            Y.append([pp])
      print(pp,p1)
      pp=pp+1

  # Формируем и выгружаем массивы
  Y = np.array(Y)
  X = np.array(X)
  X, Y = sklearn.utils.shuffle(X, Y)
  return (X,Y)
```

# Блок 5
## Загрузка данных и преобразование выходов
Для классификационной сети требуется преобразование из формата
[3] => [0, 0, 0, 1, 0], [0] => [1, 0, 0, 0, 0]
Это преобразование one-hot Encoding
Подробнее тут: https://habr.com/ru/companies/karuna/articles/769366/

``` python
(X, Y) = create_train_data(train_dir, size, CLR)
Y_en = tf.keras.utils.to_categorical(Y, num_classes=classes, dtype='float32')
```

# Блок 6
## Построение модели сети
Это моя базовая модель классификации. Она быстро обучается и работает, при этом демонстрируя высокую точность, работает на слоях 6 типов: Conv2D, Activation, MaxPooling, Dense, Flatten, Dropout
О том, как все это устроено можно найти тут: https://keras.io/api/layers/

``` python
def build_model(size, classes):
    model = Sequential()

    model.add(Input((256,256,3)))
    #model.add(Conv2D(32, (3, 3), padding="same", input_shape=(size, size, 3)))
    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.15))

    model.add(Conv2D(32, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.15))

    model.add(Conv2D(256, (3, 3), padding="same"))
    model.add(Activation("elu"))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("elu"))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model
```

# Блок 7
## Строим модель и запускаем обучение
Метод Summary выводит архитектуру модели. Метод Compile собирает граф сети, добавляя в него необходимые служебные слои для работы оптимизатора. Для оценки качества работы сеть будет полагаться на функцию bce. Для задач классификации это наилучший вариант. Метод Fit запускает обучение. Сеть 200 раз прогонит через себя входные данные (X, Y_en) пакетами по 32. 20% От выборки не будет участвовать в обучении. Это валидация. На ней будем проверять качество работы

``` python
model = build_model(size, classes)
model.summary()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae'])
history = model.fit(X,Y_en,epochs=200, batch_size=32, validation_split=0.2)
```

# Блок 8
## Получение метрик сети в виде графика
Выводится график для функции потерь (loss = 'bce'). Тут чем меньше - тем лучше. Но если функция начинает расти для графика val_loss, значит пошло переобучение. Определите на какой эпохе это началось (и началось ли в Вашем случае) и в следующий раз запускайте обучение до этой эпохи

``` python
from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь')
plt.ylabel('Потери (меньше - лучше)')
plt.xlabel('Число эпох')
plt.legend(['Обучающая', 'Тестовая'], loc='upper left')
plt.yscale('log')
plt.show()
```

# Блок 9
## Тестирование сети
Укажите путь до любой картинки, которая не входила в обучающую выборку. Оцените результат
``` python
test = cv2.resize(cv2.imread('C:/Python/cat.jpg', CLR), (size, size))
test = tf.expand_dims(test, axis=0)
print(model.predict(test))
```

# Блок 10
## Экспорт модели сети
Можно попробовать сохранить сеть в формате TFLite. В таком формате она может быть запущена на различных мобильных устройствах.

``` python
import tensorflow as tf
from keras.models import model_from_json
keras.backend.clear_session()

# Сохраняем веса модели в TensorFlow формате
model.save('model_full.h5')
model = tf.keras.models.load_model('model_full.h5', compile=False)

# Конвертируем модель в TFLIte
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Сохраняем граф модели в файл
with tf.io.gfile.GFile('model.tflite', 'wb') as f:
  f.write(tflite_model)
```