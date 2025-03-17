# Работа №14
### Сегментационные нейронные сети

В данной работе рассматриваются архитектуры, которые способы выделять контуры искомых объектов (производить сегментацию). 

### Сведения о работе

Необходимое программное обеспечение: `SAS.Planet, CVAT (или GIMP/Photoshop)`

Количество заданий для самостоятельного выполнения: ![](https://img.shields.io/badge/1%20задание-red)

Время на выполнение: ![](https://img.shields.io/badge/180-минут-blue)

### Введение

Сегментационные нейронные сети — это класс моделей глубокого обучения, которые используются для разделения изображения на различные сегменты (области) в зависимости от содержания. Они широко применяются в задачах компьютерного зрения, таких как медицинская визуализация, автономное вождение, распознавание объектов и другие. Основная цель таких сетей — присвоить каждому пикселю изображения метку, указывающую, к какому классу или объекту он принадлежит.

**Основные типы сегментации:**
1. Семантическая сегментация:
Каждому пикселю изображения присваивается метка класса (например, "дорога", "человек", "дерево"). Не различает отдельные объекты одного класса (например, два человека на изображении будут отмечены одним и тем же классом).

2. Инстанс-сегментация:
Помимо семантической сегментации, различает отдельные объекты одного класса (например, два человека будут выделены как два разных объекта).

#### Примеры архитектур:
**U-Net:**

Популярна в медицинской визуализации.

Состоит из симметричного энкодера и декодера с пропускными соединениями (skip connections), которые помогают сохранить детали.

**DeepLab:**

Использует атроконволюции (atrous convolutions) для увеличения рецептивного поля без потери разрешения.

Применяет CRF для уточнения границ.

**Mask R-CNN:**

Расширяет Faster R-CNN, добавляя ветку для предсказания масок объектов.

Подходит для инстанс-сегментации.

## Подготовка набора данных

В данной работе я предлагаю рассмотреть сегменатцию спутниковых изображений. Для этого нам потребуется загрузить тайлы (фрагменты одного размера) спутниковых снимков и разметить их. Сегментационные сети могут выделять объекты нескольких классов (categorial), но мы рассмотрим только 2 класса: объект и фон. 
Объект будем выделять белым цветом, фон - черным. Для лучшей сходимости допускается применение серого - неопределенные пиксели. 

Для загрузки тайлов воспользуйтесь программой SAS.Planet. Разметку можно производить в любом графическом редакторе. 

Тестовый набор данных доступен по ссылке: https://raw.githubusercontent.com/vb2005/keras-quickstart/main/Datasets/forest.7z

## Подключение библиотек

Потребуется стандартный набор

``` python 
import os
import cv2
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
```
## Чтение датасета

Воспользуемся библиотекой OpenCV для чтения изображений.

``` python
IMG_FOLDER = "/content/forest/img"
MASK_FOLDER = "/content/forest/mask"
SIZE = 256
CLR = cv2.IMREAD_COLOR
GR = cv2.IMREAD_GRAYSCALE

def create_train_data():
  xx = []
  yy = []

  # Проходим по всем файлам
  for img in os.listdir(IMG_FOLDER):
        path = os.path.join(IMG_FOLDER, img)                      # получаем пути
        img = cv2.resize(cv2.imread(path, CLR), (SIZE, SIZE))     # читаем картинки
        xx.append(list(np.array(img)))                            # добавляем в массив

  for img in os.listdir(MASK_FOLDER):
        path = os.path.join(MASK_FOLDER, img)
        img = cv2.resize(cv2.imread(path, GR), (256, 256))
        img = img/255.0
        yy.append(list(np.array(img)))

  yy = np.expand_dims(yy,3)                                       # UNET может сразу в несколько классов. У нас только один. Поэтому нужно еще расширить размерность
  xx=np.array(xx)
  yy=np.array(yy)
  return (xx,yy)

(X_train2, Y_train2) = create_train_data()
```

## Модель UNET
Сетка отличается очень необычной структрой. Данные с верхних свёрток падают на нижние. Отсюда и название
![Описание изображения](https://user-images.githubusercontent.com/33135767/92586254-a46e9600-f2b3-11ea-8b24-bb838960dd90.png)

``` python
# По классике, UNET имеет размерность входа/выхода 128.
# Я для примера нарастил до 256 на 256. А вообще, если видеокарта позволяет, её можно даже в 512х512х512 обучать
def build():
  inputs = Input((256, 256, 3))
  s = Lambda(lambda x: x / 255) (inputs)


  c0 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (s)
  c0 = Dropout(0.5) (c0)
  c0 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c0)
  p0 = MaxPooling2D((2, 2)) (c0)

  c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p0)
  c1 = Dropout(0.5) (c1)
  c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
  p1 = MaxPooling2D((2, 2)) (c1)

  c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
  c2 = Dropout(0.5) (c2)
  c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
  p2 = MaxPooling2D((2, 2)) (c2)

  c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
  c3 = Dropout(0.5) (c3)
  c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
  p3 = MaxPooling2D((2, 2)) (c3)

  c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
  c4 = Dropout(0.5) (c4)
  c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
  p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

  c5 = Conv2D(256, (5, 5), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
  c5 = Dropout(0.5) (c5)
  c5 = Conv2D(256, (5, 5), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

  u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
  u6 = concatenate([u6, c4])
  c6 = Conv2D(128, (7, 7), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
  c6 = Dropout(0.5) (c6)
  c6 = Conv2D(128, (7, 7), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

  u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
  u7 = concatenate([u7, c3])
  c7 = Conv2D(64, (7, 7), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
  c7 = Dropout(0.5) (c7)
  c7 = Conv2D(64, (7, 7), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

  u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
  u8 = concatenate([u8, c2])
  c8 = Conv2D(32, (5, 5), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
  c8 = Dropout(0.5) (c8)
  c8 = Conv2D(32, (5, 5), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

  u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
  u9 = concatenate([u9, c1], axis=3)
  c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
  c9 = Dropout(0.5) (c9)
  c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

  u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c9)
  u10 = concatenate([u10, c0], axis=3)
  c10 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u10)
  c10 = Dropout(0.5) (c10)
  c10 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c10)

  outputs = Conv2D(1, (1, 1), activation='sigmoid') (c10)

  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer='Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
  return model

model = build()
```

## Обучение и проверка работы

``` python
for i in range(1000):
  model.fit(x=X_train2, y=Y_train2,batch_size=5,epochs=1)                     # Запускаем очередную эпоху
  img = cv2.resize(cv2.imread("/content/y5244.jpg", CLR), (SIZE, SIZE))      # По её окончанию читаем картинку
  raw = np.expand_dims(img, 0)                                                # Изменяем размерность до (1, 256, 256, 3)
  pred = model.predict(raw)                                                   # Получаем predict
  pred = pred.squeeze().reshape(256,256)*255                                  # predic t приводим к адекватной размерости в (256, 256)
  #cv2.imwrite("SomeResults.png", pred)                                        # сохраняем результат
  cv2.imshow('img', pred)  
```
