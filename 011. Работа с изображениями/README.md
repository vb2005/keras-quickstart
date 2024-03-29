# Основы работы с OpenCV

Перед началом работы загрузите в рабочий каталог файлы **rose.png** и **white.png**. Они пригодятся для работы

Подключите необходимые библиотеки
``` python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
```
## Блок 1. Открытие, отображение и сохранение изображения

``` python
# Загрузка цветного изображения
img = cv2.imread('rose.jpg')

# Получение информации о нём
print(img.shape)
print(img.dtype)
```

``` python
# Загрузка с преобразванием в монохромному формату
img_gray = cv2.imread('rose.jpg', cv2.IMREAD_GRAYSCALE)

# Получение информации о нём
print(img_gray.shape)
print(img_gray.dtype)
```

``` python
# Вывод изображения на экран
# Конструкция для выполнения на ПК
# cv2.imshow('demo', img)

# Конструкция для Google colab:
from google.colab.patches import cv2_imshow
cv2_imshow(img)
```

``` python
# Сохранение изображения в файл
cv2.imwrite('demo.jpg', img_gray)
```
## Блок 2. Работа с цветом
``` python
# Построение гистограммы
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv2.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()
```

``` python
# Изменение рабочей палитры:
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2_imshow(img_gray)
```
Также доступна работа в RGB, ARGB, GRAY, XYZ, YCrCb, HSV, Lab, Luv, HLS, YUV, Bayer

``` python
# Инверсия цвета
img_test = 255 - img
cv2_imshow(img_test)
```

``` python
# Изменение типа канала на int16
img_test = np.int16(img)
print(img.dtype)
```

``` python

```

``` python

```

