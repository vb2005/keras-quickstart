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
  cv2.imshow('demo',img_test)
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

### Перенос
Пример матрицы для переноса изображения. Смещение определяется в абсолютных значениях

$$
  H_{translate} =
  \left( {\begin{array}{cc}
    1 & 0 & c_{x} \\
    0 & 1 & c_{y} \\
    0 & 0 & 1 \\
  \end{array} } \right)
$$

``` python
# На 100 пикс. по оси X
# На -50 пикс. по оси Y
matrix = [[1, 0, 100],
          [0, 1, -50],
          [0, 0, 1]]

visualize(img, matrix)
```
### Наклон
Пример матрицы для наклона. Для получения адекватного результата старайтесь использовать маленькие значения

$$
  H_{slant} =
  \left( {\begin{array}{cc}
    1 & t_{x} & 0 \\
    t_{y} & 1 & 0 \\
    t_{x} & t_{y} & 1 \\
  \end{array} } \right)
$$

``` python
# Пример матрицы, для наклона (для удобства, вместе с переносом)
# На 0.05%  по оси X
matrix = [[1,      0.0005,   300],
          [0,           1,   300],
          [0.0005,      0,     1]]

visualize(img, matrix)
```

### Поворот на произвольный угол
Пример матрицы для поворота изображения.

$$
  H_{rotate} =
  \left( {\begin{array}{cc}
    cos(a) & -sin(a) & 0 \\
    sin(a) & cos(a) & 0 \\
    0 & 0 & 1 \\
  \end{array} } \right)
$$

``` python
from math import cos, sin, pi
# Пример матрицы, для поворота (для удобства, вместе с переносом)
angle = 45

# Перевод в радианы
a = angle / 180 * pi

matrix = [[cos(a), -sin(a),   300],
          [sin(a),  cos(a),   300],
          [0,            0,     1]]

visualize(img, matrix)
```

## Блок 5. Выделение значимых характеристик изображения
### Фильтр Собеля
``` python
# Работа осуществляется только в одноканальном режиме
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Собель - это частная производная с шагом ksize по выбранной переменной (x, y) или для всех сразу
sobelx = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) / 10
sobely = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) / 10
sobelxy = cv2.Sobel(src=img_gray, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5)

titles = ['Original Image','X','Y','XY']
images = [img_gray, sobelx, sobely, sobelxy]

for i in range(4):
 plt.subplot(2,2,i+1),plt.imshow(images[i],'gray',vmin=0,vmax=255)
 plt.title(titles[i])
 plt.xticks([]),plt.yticks([])

plt.show()
```

### Фильтр Canny
Фильтр Canny позволяет получить более тонкие контуры, в сравнении с Sobel, за счёт применения минимизации немаксимумов
``` python
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_canny = cv2.Canny(img,300,300)
cv2.imshow('demo', img_canny)
```

### Сглаживающие фильтры
Фильтр, в котором все значения внутри матрицы равны называется сглаживающи
``` pythonм
kernel = np.ones((5,5),np.float32)/25
print(kernel)
img_test = cv2.filter2D(img,-1,kernel)
cv2.imshow('demo', img_test)
```

Для него существует специальный метод:
``` python
img_test = cv2.blur(img,(5,5))
cv2.imshow('demo', img_test)
```

Обратите внимание на следующий фильтр.
Что напоминает результат его работы?
``` python
kernel = [[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]]
kernel = np.array(kernel, dtype = 'float32')
img_test = cv2.filter2D(img_gray,-1,kernel)
cv2.imshow('demo', img_test)
```

### Фильтр Гаусса
Данный фильтр намного лучше размывает изображения за счёт более низких значений по углам квадратной матрицы. Под капотом он использует распределение Гаусса

``` python
img_test = cv2.GaussianBlur(img,(5,5),0)
cv2.imshow('demo',  img_test)
```

### Медианный  фильтр
Пример вызова меданного фильтра. Данный фильтр записывает в качестве результата медианное значение из окрестностей n*n. Данный фильтр отлично удаляет импульсные помехи
``` python

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for i in range(5000):
  x = random.randint(0, img_gray.shape[0]-1)
  y = random.randint(0, img_gray.shape[1]-1)
  img_gray[x,y] = 0
  x = random.randint(0, img_gray.shape[0]-1)
  y = random.randint(0, img_gray.shape[1]-1)
  img_gray[x,y] = 255

img_test = cv2.medianBlur(img_gray,3)
cv2.imshow('demo', img_gray)
cv2.imshow('demo', img_test)
```

## Блок 6. Генерация изображений
Небольшой гайд по созданию градиентных изображений с использованием тригонометрии

Создадим холст для творчества
``` python
img = np.zeros((240,320,3),dtype='uint8')
```

Сделаем все пиксели одним цветом:
``` python
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
      img[i,j,0] = 255
      img[i,j,1] = 255
      img[i,j,2] = 0

cv2.imshow('demo', img)
```

Заполняем случайным шумом:
``` python
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    for с in range(img.shape[2]):
      img[i,j,с] = np.random.randn() % 255

cv2.imshow('demo', img)
```

Добавляем градиент:
``` python
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
      img[i,j,0] = i
      img[i,j,1] = 0
      img[i,j,2] = j

cv2.imshow('demo', img)
```

В прошлом примере произошёл выход за пределы формата uint8. Исправить диапазон значений можно через функции, которые имеют ограниченную ОДЗ, например, sin или cos
``` python
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    img[i,j,0] = cos((j+i) / 180 * 3.14) * 128 + 127
    img[i,j,1] = 0
    img[i,j,2] = 0

cv2.imshow('demo',img)
```

А теперь попробуем задействовать все каналы
``` python
for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    img[i,j,0] = cos((j+i)   / 180 * 3.14) * 128 + 127
    img[i,j,1] = cos((2*i)   / 180 * 3.14) * 128 + 127
    img[i,j,2] = cos((0.5*j) / 180 * 3.14) * 128 + 127

cv2.imshow('demo',img)
```

При этом можно работать в другом цветовом пространстве:
``` python
img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for i in range(img.shape[0]):
  for j in range(img.shape[1]):
    img[i,j,0] = cos(j / 180 * 3.14) * 90 + 90
    img[i,j,1] = 255
    img[i,j,2] = 255

img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

cv2.imshow('demo',img)
```
