import cv2, sklearn, os, keras
import numpy as np
import tensorflow as tf
from keras.optimizers import *
from keras.models import Model, Sequential
from keras.layers import *
from keras.utils import *

# Размер входного изображения
size = 256

# Папка с обучающей выборкой
train_dir = "C:/Python/cats_and_dogs/train"

# Цвет изображения
CLR = cv2.IMREAD_COLOR

# Количество классов
classes = 2

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

(X, Y) = create_train_data(train_dir, size, CLR)
Y_en = tf.keras.utils.to_categorical(Y, num_classes=classes, dtype='float32')

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

model = build_model(size, classes)
model.summary()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy', 'mse', 'mae'])
history = model.fit(X,Y_en,epochs=200, batch_size=32, validation_split=0.2)

from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь')
plt.ylabel('Потери (меньше - лучше)')
plt.xlabel('Число эпох')
plt.legend(['Обучающая', 'Тестовая'], loc='upper left')
plt.yscale('log')
plt.show()

test = cv2.resize(cv2.imread('C:/Python/cat.jpg', CLR), (size, size))
test = tf.expand_dims(test, axis=0)
print(model.predict(test))

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
