import os, tqdm, cv2, sklearn
import numpy as np
from keras.utils import to_categorical
import tensorflow as tf
import keras
from keras.optimizers import *
from keras.models import Model, Sequential
from keras.layers import *
from matplotlib import pyplot as plt

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
Y_en = tf.keras.utils.to_categorical(Y, num_classes=2, dtype='float32')


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
    
model = build_model(size, classes)
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[ 'accuracy', 'mse', 'mae'])
history = model.fit(X,Y_en,epochs=40, batch_size=256, validation_split=0.2)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Функция потерь')
plt.ylabel('Потери (меньше - лучше)')
plt.xlabel('Число эпох')
plt.legend(['Обучающая', 'Тестовая'], loc='upper left')
plt.show()

path = 'C:/Python/Test.jpg'
img = cv2.resize(cv2.imread(path, color_mode), (size, size))
img = np.array([img])
pred = model.predict(img)
print(pred)