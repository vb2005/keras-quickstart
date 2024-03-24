# Pandas. Работа с данными 
## Подготовка
Для данной работы потребуется библиотека Pandas. Установите её, если работаете на локальном ПК, при помощи команды:

```
pip install pandas
```

Загрузите набор данных "Ирисы Фишера" [по ссылке](https://raw.githubusercontent.com/vb2005/keras-quickstart/main/Datasets/iris.csv). Данный набор является одним из наиболе известных наборов для обучения работе с данными. Подробнее о нём можно прочитать [тут](https://ru.wikipedia.org/wiki/%D0%98%D1%80%D0%B8%D1%81%D1%8B_%D0%A4%D0%B8%D1%88%D0%B5%D1%80%D0%B0).

Ирисы Фишера состоят из данных о 150 экземплярах ириса (150 строк), по 50 экземпляров из трёх видов — Ирис щетинистый (setosa), Ирис виргинский (virginica) и Ирис разноцветный (versicolor). Для каждого экземпляра измерялись четыре характеристики (в сантиметрах):

1. Длина наружной доли околоцветника (англ. sepal_length);
2. Ширина наружной доли околоцветника (англ. sepal_width);
3. Длина внутренней доли околоцветника (англ. petal-length);
4. Ширина внутренней доли околоцветника (англ. petal_width).

На основании этого набора исследуются алгоритмы классификации данных, определяющие вид растения по данным измерений. Это задача многоклассовой классификации, так как имеется три класса — три вида ириса.

## Чтение данных и статистическая информация

Загрузите в рабочий каталог набор данных по ссылке: https://raw.githubusercontent.com/vb2005/keras-quickstart/main/Datasets/iris.csv

Подключите библиотеку pandas к проекту:
``` python
import pandas as pd
```

``` python
# Загружаем данные из файла
df = pd.read_csv("iris.csv")
df.head()
```

``` python
# Или формируем вручную (3 строки для примера)
df2 = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 'versicolor'],
                    [0.3, 0.5, 0.1, 1.4, 'virginica'],
                    [0.4, 0.5, 0.4, 1.8, 'setosa']],
columns=['sepal_length','sepal_width','petal_length','petal_width','species'])
df2.head()
```

``` python
# Посмотреть размер массива
df.shape
```
``` python
# Посмотреть количество строк
len(df)
```
``` python
# Поличить информацию о столбцах
df.info()
```
``` python
# Поличить статистику по числовым столбцам
df.describe()
```
``` python
# Информация о количестве данных разных классов (указывается название класса)
df.value_counts("species")

# Или так
df.species.value_counts()
```
## Выборки и срезы
``` python
# Создание независимой копии
df_copy = df.copy(deep=True)
```
``` python
# Первые 10 строк
df[:10]
```
``` python
# Последние 2 строки
df[-2:]
```
``` python
# Строки со 2 по 9
df[2:10]
```
``` python
# Получить все уникальные значения столбца
df['species'].unique()

# Получить частоту повторения значений
df.species.value_counts()
```
``` python
# Проверяем отсуствующие(незаполненные) значения
df.isnull().sum()
```
``` python
# Выборка ирисов только двух видов:
df[df['species'].isin(['setosa', 'versicolor'])]
```
``` python
# Фильтр по числовому значению
df[df['sepal_length'] > 7]
```
``` python
# Сортировка данных
df.sort_values('sepal_length', ascending=False)
```
``` python
# Выбрать случайную строку
df.sample()
```
``` python
# Выбрать случайные 15% значений
df.sample(frac=0.15)
```
``` python
# Перемешать данные
df.sample(frac=1)
```
## Модификация данных
``` python
# Добавить столбец со значением по умолчанию
df['use'] = True
df
```
``` python
# Выбрать только указанные столбцы
df = df[['sepal_length','sepal_width','petal_length','petal_width','species']]
df
```
``` python
# Удаление столбцов
df.drop(['use'], axis=1)
```
``` python
# Перевести названия классов в формат One-Hot (подробнее: https://www.codecamp.ru/blog/one-hot-encoding-in-python/)
pd.get_dummies(df)
```
## Экспорт данных
``` python
# В файл CSV
df.to_csv('data.csv', index=False)
```
``` python
# Один столбец как список
l = df['species'].tolist()
l
```
``` python
# Перевод в Numpy-Array
np_array = df.values
np_array.shape
```
# Основано на матриалах авторов:
1. https://habr.com/ru/companies/ruvds/articles/494720/
2. https://pythonist.ru/pandas-tutorial/