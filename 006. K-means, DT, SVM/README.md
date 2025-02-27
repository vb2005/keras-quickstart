# Кластеризация, Деревья и SVM

В данном разделе рассмотрим существующие методы из раздела машинного обучения, такие как k-means, desigion tree, svm. Эти методы широко применяются в задачах, для которых характерно явное разделение данных на основе одного или нескольких признаков. 

В этой работе мы предолжим использовать библиотеку машинного обучения: ```sklearn```.

Количество заданий для самостоятельного выполнения: ![](https://img.shields.io/badge/4%20задания-red)

Время на выполнение: ![](https://img.shields.io/badge/90-минут-blue)

``` python
# Подключим к проекту все необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import datasets, svm

import warnings
warnings.filterwarnings('ignore')
```

## Введение
В данной работе мы поработаем с набором Ирисы Фишера. Он как нельзя лучше подходит для исследования данных методов. Данный набор можно загрузить в виде файла, а можно загрузить при помощи ```seaborn```. Воспользуемся этим способом

``` python
# Загружаем данные для анализа
iris = load_iris()
iris = sns.load_dataset('iris')
iris.head()
```

## Кластерный анализ. K-means
**k-means** — это алгоритм кластеризации, который позволяет разделить набор данных на k кластеров. Основная идея заключается в том, чтобы минимизировать внутрикластерное расстояние, то есть расстояние между точками внутри одного кластера, а также максимизировать расстояние между кластерами.

Вот основные шаги алгоритма **k-means**:

1. **Выбор количества кластеров (k):** Пользователь задает количество желаемых кластеров.
2. **Инициализация центров кластеров:** Случайным образом выбираются k точек из данных, которые служат начальными центрами кластеров.
3. **Сопоставление точек с кластерами:** Каждая точка данных присваивается ближайшему центру кластера, образуя кластеры.
4. **Обновление центров кластеров:** Центры кластеров пересчитываются как среднее значение всех точек, входящих в каждый кластер.
5. **Повторение:** Шаги 3 и 4 повторяются до тех пор, пока центры кластеров больше не изменяются или достигается заданное число итераций.

Алгоритм **k-means** прост в реализации и широко используется в задачах анализа данных, таких как сегментация рынка или организация информации. Однако он требует предварительного указания числа кластеров и может быть чувствителен к выбору начальных центров.

В данной работе рассматривается модификация алгоритма ```k-means++``` в которой применяется алгоритм начального распределения центроидов. Также мы попробуем исследовать поведение СКО от количества кластеров. На его основе можно сделать вывод об оптимальном количестве кластеров.

``` python
# Анализируем данные на предмет количества потенциальных кластеров
wcss = []
x = iris.iloc[:, [0, 1, 2, 3]].values

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
```


``` python
# Визуализация графика СКО от кол-ва кластеров
plt.plot(range(1, 11), wcss)
plt.xlabel('Кол-во кластеров')
plt.ylabel('СКО') #within cluster sum of squares
plt.show()
```


``` python
# Определяем центроиды для числа кластеров 3
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x)

# Визуализация кластеров для двумерного пространства признаков
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'purple', label = 'Iris-setosa')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'orange', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Iris-virginica')

# Выводим центроиды
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'red', label = 'Центроиды')

plt.legend()
```


``` python
# Задание 1.
# Разработайте свою реалзиацию метода k-means. Сравните результат её работы с результатом работы k-means++

# Задание 2.
# Подумайте над тем, как на оснвое графика СКО определить количество необходимых кластеров.
```

## Деревья принятия решения 
**Деревья принятия решений** - это модель машинного обучения, которая используется для классификации и регрессии. Они представляют собой иерархическую структуру, где каждый узел соответствует вопросу о значении признака, а ветки обозначают возможные ответы. Листовые узлы дерева содержат итоговые решения или классы.

Процесс построения дерева включает выбор признаков и порогов, которые наиболее эффективно разделяют данные на группы, оптимизируя критерий (например, информационную выгоду или энтропию). Деревья легко интерпретируются и визуализируются, что делает их популярными для задач, где важна понимание принятых решений. Однако они могут быть склонны к переобучению, если не применяются методы регуляризации.

``` python
# Разделим выборку на обучение и тест
X = iris.iloc[:, :-2]
y = iris.species
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                       test_size=0.33,
                                       random_state=42)

# Созадим дерево с минимальном лепестком - 4
treemodel = DecisionTreeClassifier(min_samples_leaf=4)

# Запустим обучение
treemodel.fit(X_train, y_train)
```


``` python
# Выведем дерево на экран
plt.figure(figsize=(15, 10))
tree.plot_tree(treemodel, filled=True)

# Построим прогноз на тестовой выборке
ypred = treemodel.predict(X_test)
score = accuracy_score(ypred, y_test)

# Выведем показатели точности
print(score)
print(classification_report(ypred, y_test))
```

## Метод опорных векторов
**SVM (Support Vector Machine)** — это метод машинного обучения, используемый для классификации и регрессии. В основе SVM лежит идея поиска гиперплоскости, которая наилучшим образом разделяет классы данных в многомерном пространстве.

Основные характеристики SVM:

Разделительные границы: SVM ищет гиперплоскость, которая максимизирует расстояние (или "зазор") между ближайшими точками различных классов, называемыми опорными векторами.

**Ядра:** Для сложных данных, которые не поддаются линейной классификации, SVM может использовать ядровые функции (например, полиномиальные или радиально-базисные функции), что позволяет эффективно преобразовать данные в более высокое измерение, где они могут быть линейно разделимы.

**Регуляризация:** SVM включает параметры, которые позволяют контролировать компромисс между максимизацией зазора и минимизацией ошибок классификации, что делает его устойчивым к переобучению.

SVM применяется в различных областях, включая распознавание образов, биоинформатику и текстовую классификацию.

``` python
# Готовим набор данных. Берем два столбца для удобства визуализации
# В последний столбец заносим индекс класса
X = iris.iloc[:, [0, 2]].values
y = iris.species
yy = np.unique(y, return_inverse=True)[1]
```

``` python
C = 1.0  # Задаём параметр регуляризации
models = (
    svm.SVC(kernel="linear", C=C),                          # Классическое линейное разделение
    svm.LinearSVC(C=C, max_iter=10000),                     # 
    svm.SVC(kernel="rbf", gamma=0.7, C=C),
    svm.SVC(kernel="poly", degree=3, gamma="auto", C=C),
)
models = (clf.fit(X, yy) for clf in models)
```


``` python
# Названия графиков
titles = (
    "SVC with linear kernel",
    "LinearSVC (linear kernel)",
    "SVC with RBF kernel",
    "SVC with polynomial (degree 3) kernel",
)


fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)


X0, X1 = X[:, 0], X[:, 1]

for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax
    )
    ax.scatter(X0, X1, c=yy, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()
```



