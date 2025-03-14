# Работа №5
### Регрессия. Линия, экспонента, полином

В данной работе рассматриваются простейшие модели машинного обучения, построенные на регрессионном анализе. Предлагается изучить существующие решения в данной области, а также разработать собственные
### Сведения о работе

Для исследования данных моделей предлагается использовать одну из самых популярных библиотек в данной области: ``` sklearn ```

Для работы установите её при помощи команды:

```pip install scikit-learn```


Количество заданий для самостоятельного выполнения: ![](https://img.shields.io/badge/4%20задания-red)

Время на выполнение: ![](https://img.shields.io/badge/90-минут-blue)

## Введение
**Линейная регрессия** (англ. Linear regression) — используемая в статистике регрессионная модель зависимости одной (объясняемой, зависимой) переменной **y** от другой или нескольких других переменных (факторов, регрессоров, независимых переменных) **x** с линейной функцией зависимости.

Модель линейной регрессии является часто используемой и наиболее изученной в эконометрике. А именно изучены свойства оценок параметров, получаемых различными методами при предположениях о вероятностных характеристиках факторов, и случайных ошибок модели. Предельные (асимптотические) свойства оценок нелинейных моделей также выводятся исходя из аппроксимации последних линейными моделями. С эконометрической точки зрения более важное значение имеет линейность по параметрам, чем линейность по факторам модели.

**Линейная регрессия** – это построение такой прямой, которая с наименьшей ошибкой проходит по представленным точкам.

![image.png](data:image/png;base64,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)

## Загрузка данных
Данную работу предлагается проводить на 2х наборах данных: синтетическом, основанным на небольшом наборе регулярных точек и экспериментальном, основанный на исследованиях зависимостей в области металлургии (https://www.kaggle.com/datasets/rukenmissonnier/manufacturing-data-for-polynomial-regression/code)

``` python
# Необходимые библиотеки
import pandas as pd
import seaborn as sns
import numpy as np

# Пакет sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
```

``` python
# Синтетический набор, имитирующий поведение некоторой кривой
X = [1, 2, 3, 4, 5, 6, 7, 8]
Y = [1, 4, 9, 16, 25, 36, 49, 54]

df = pd.read_csv("manufacturing.csv")
```

![](https://img.shields.io/badge/задание%20№1-red) 
``` python
# Визуализируйте исходные данные.
# 1. Определите, какому типу кривой соответствует набор точек?
# 2. Постройте парные диаграммы для набора данных из `manufacturing.csv`. Определите визуально, какие из величин имеют явные зависимости
```

## Предобработка данных
Сейчас исходные данные представлены в виде списка и в виде таблицы Pandas. В таком формате они не могут быть переданы в математический аппарат регрессии. Во-первых, они должны быть типа Numpy-array, а во-вторых, должны быть представлены в виде двумерного массива. Выполним подобное преобразование и убедимся, что данные были успешно преобразованы

``` python
X = np.array(X).reshape(-1, 1)
Y = np.array(Y).reshape(-1, 1)

print(X.shape, Y.shape)
```

``` python
X_1 = df['Material Fusion Metric'].values.reshape(-1, 1)
Y_1 = df['Quality Rating'].values.reshape(-1, 1)

print(X_1.shape, Y_1.shape)
```

## Обучение линейной регрессии
Рассмотрим пример обучения линейной регрессии. Необходимо создать экземпляр класса `LinearRegression` и вызвать у него метод `fit`

``` python
model = LinearRegression()
lr_1 = model.fit(X, Y)
```

## Получение весов, прогноз и метрики

Все результаты обучения хранятся в переменной `lr_1` и мы можем к ним получить доступ через свойства.

``` python
# Тестовые данные
xx_1 = [1, 2]

# Преобразование типа массива
xx_1 = np.array(xx_1).reshape(-1, 1)

# Получение прогноза
yy_1 = model.predict(xx_1)

# Вывод данных
print('x = ', xx_1)
print('y = ', yy_1)
```

``` python
# Выделение коэффициентов при x:
k = lr_1.coef_[0][0]

# Выделение свободного члена b:
b = lr_1.intercept_[0]

# Вывод на экран
print('y =', k, '* x +', b)
```
![](https://img.shields.io/badge/задание%20№2-red) 
``` python
# Разработайте функцию MSE, которая для вводимой модели регрессии позволит оценить ошибку (СКО) её работы на наборе данных
```

## Обучение полиномиальной регрессии
Полиномиальная регрессия может быть представлена в виде многочлена n-ой степени $$ y = \sum_{i=1}^{n} w_i x^{i} + bias $$. Данное выражение может быть вполне представимо в том случае, если во множественную линейную регрессию подставить не самостоятельные величины, а значения `x` возведенную в нужную степень. Для решения такой задачи к `sklearn` есть готовое решение:

``` python
def CalcPolynomialRegression(X, Y, n):
  # Создаем объект, который поможет сгенерировать данные для множественной регрессии
  p = PolynomialFeatures(degree=n)
  X_poly = p.fit_transform(X)

  # Таким образом, функция создала набор степеней для исходного выражения
  print(X_poly)

  # А далее мы переходим к классической линейной регрессии и обучаем её
  model = LinearRegression()
  lr = model.fit(X_poly, Y)

  # Извлекаем коэффициенты, причём свободный член находится в свойстве coef_, 
  # поскольку здесь в наборе присутствовал столбец нулевой степени. Он и стал поправкой
  koef = []

  for i in range(n,-1,-1):
    koef.append(lr.coef_[0][i])

  return koef

result = CalcPolynomialRegression(X, Y, 2)
print("Результат работы метода: ", result)
```

## Самостоятельная работа
**Задача 0.** Внимательно изучить практический материал, выполнить 2 мини-кейса.

**Задача 1.** В наборе данных manufacturing.csv для 5 любых зависимостей попробуйте найти Линейную или полиномиальную регрессию с минимальной степенью свободы, которая бы удовлетворяла общему тренду данных.

**Задача 2.** Разработайте метод для обучения регрессии функции, отличной от прямой или полинома (например, экспонента или тригонометрические функции). Обучите на адекватных для этой функции наборе данных и сделайте выводы о возможности её применения для решения данной задачи.

**Задача 3.** Разработайте свой метод для обучения линейной регрессии. Проверьте на нём исходные данные и сравните с результатами, полученными в `sklearn`.

## Контрольные вопросы
1. Что такое регрессия?
2. Какие виды регрессии вы знаете?
3. В чём особенность множественной линейной регрессии?
4. Как оценить СКО по результатам работы регрессии?
