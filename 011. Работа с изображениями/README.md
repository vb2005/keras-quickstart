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

Загрузка цветного изображения и получение информации о нём
``` python
img = cv2.imread('rose.jpg')

print(img.shape)
print(img.dtype)
```

Загрузка с преобразванием в монохромному формату
``` python
img_gray = cv2.imread('rose.jpg', cv2.IMREAD_GRAYSCALE)

# Получение информации о нём
print(img_gray.shape)
print(img_gray.dtype)
```

Вывод изображения на экран
``` python
cv2.imshow('demo', img)

# Конструкция для Google colab:
# from google.colab.patches import cv2_imshow
# cv2_imshow(img)
```

Сохранение изображения в файл
``` python
cv2.imwrite('demo.jpg', img_gray)
```

## Блок 2. Работа с цветом

Построение гистограммы
``` python
color = ('b','g','r')
for i,col in enumerate(color):
 histr = cv2.calcHist([img],[i],None,[256],[0,256])
 plt.plot(histr,color = col)
 plt.xlim([0,256])
plt.show()
```

Изменение рабочей палитры:
``` python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('demo', img_gray)
```
Также доступна работа в RGB, ARGB, GRAY, XYZ, YCrCb, HSV, Lab, Luv, HLS, YUV, Bayer

Инверсия цвета
``` python
img_test = 255 - img
cv2.imshow('demo', img_test)
```

Изменение типа канала на int16
``` python
img_test = np.int16(img)
print(img.dtype)
```

Увеличение яркости картинки (на 50 ед.)
``` python
img_test = img_test + 50

# Отрезаем данные, которые не входят в uint8
img_test = np.clip(img_test, 0, 255)

# Конвертируем в диапазон uint8
img_test = np.uint8(img_test)
cv2.imshow('demo', img_test)
```

Изменим цвет розы. Для этого переведем её в палтру HSV
``` python
img_test = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
for x in range(img_test.shape[0]):
  for y in range(img_test.shape[1]):

    # Сместим розовый спектр на 50 ед.
    h = img_test[x, y, 0]
    if (h > 120):
      h = h - 50
    img_test[x,y,0] = h

# Вернемся обратно к палитре RGB
img_test = cv2.cvtColor(img_test, cv2.COLOR_HSV2BGR)
cv2.imshow('demo', img_test)
```

Задание №2. Исправление баланса белого.
Баланс белого - это один из параметров фотографии, определяющий и компенсирующий цветовую температуру источника света на фото. Фотоаппараты умеют делать это автоматически, хотя, и иногда ошибаются. В файле white.jpg - пример фото с нарушением баланса белого. Из-за этого фотография кажется более "синей", чем есть на самом деле. Причина этого -  избыточное значение в синем канале. Исправить баланс белого можно в одной из двух палтир: В палитре RGB - это величина B-кнаала, в палитре LAB - тоже канал B. Сравните результаты

``` python
# Бинаризация
ret,thresh1 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img_gray,127,255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img_gray,127,255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img_gray, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
 plt.subplot(2,3,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])

plt.show()

```

## Блок 3. Преобразования

Изменение размера изображения
``` python
# Указывается исходное изображение и новый размер
img_test = cv2.resize(img, (128,128))
cv2.imshow('demo', img_test)
```

Срезы. Получение фрамента
``` python
# Следующий код вырежет фрагмент картинки.
# Будет выбран квадрат от 100 до 200 пикселя по X
# и от 100 ло 200 пикслея по Y
img_test = img[100:200, 100:200, :]
cv2.imshow('demo', img_test)
```

Поворот и отражение
``` python
# Поворот можем сделать при помощи swapaxes
# Мы меняем координаты X на координаты Y
# Тем самым сделаем поворот на 270 град.
img_test = np.swapaxes(img,0,1)
cv2.imshow('demo', img_test)ython
```

``` python
# Отражение можно сделать через flip
img_test = np.flip(img_test, 0)
cv2.imshow('demo', img_test)
```

## Блок 4. Матрица гомографии
Работа с матрицей выделена в отдельный блок, поскольку она позволет применять различные виды деформаций к исходному изображению. В OpenCV существует метод warpPerspective, который позволяет применять матрицу гомографии. В матрице гомографии каждый коэффицент определяет какой-либо вид преобразования.

Сделаем функцию для применения матрицы, и рассмотрим результат на разных видах матрицы
``` python
def visualize(img, h_matrix):
  h_matrix = np.array(h_matrix, dtype = 'float32')
  img_test = cv2.warpPerspective(img, h_matrix, (1000,1000))
  cv2_imshow(img_test)
```

### Исходная матрица
Пример матрицы, которая сохранит исходное изображение без изменений

$$
  H_{default} =
  \left( {\begin{array}{cc}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1 \\
  \end{array} } \right)
$$

``` python
matrix = [[1, 0, 0],
          [0, 1, 0],
          [0, 0, 1]]

visualize(img, matrix)
```
### Масштабирование
Пример матрицы для измеенения размера. Размер меняется в относительных координатах и задаётся по главной диагонали

$$
  H_{scale} =
  \left( {\begin{array}{cc}
    s_{w} & 0 & 0 \\
    0 & s_{h} & 0 \\
    0 & 0 & 1 \\
  \end{array} } \right)
$$

``` python
# Растянуть в 2 раза по оси X
# Растянуть в 3 раза по оси Y
matrix = [[2, 0, 0],
          [0, 3, 0],
          [0, 0, 1]]

visualize(img, matrix)
```
$$
  H_{translate} =
  \left( {\begin{array}{cc}
    1 & 0 & c_{x} \\
    0 & 1 & c_{y} \\
    0 & 0 & 1 \\
  \end{array} } \right)
$$
``` python
# Пример матрицы, для переноса
# На 100 пикс. по оси X
# На -50 пикс. по оси Y
matrix = [[1, 0, 100],
          [0, 1, -50],
          [0, 0, 1]]

visualize(img, matrix)
```

``` python

```

``` python

```

``` python

```
