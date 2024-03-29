# Блок 1
## Подготовка данных
В данной работе предлагается обучить нейронную сеть находить дорожные ямы. Мы возьмём библиотеку Ultralytics и модель YOLO v8. 

В качестве набора данных будем использовать готовый набор данных от https://github.com/spmallick. Он доступен по ссылке: https://www.dropbox.com/s/qvglw8pqo16769f/pothole_dataset_v8.zip?dl=1. Его необходимо скачать и распаковать в рабочий каталог.

Для обучения сети Вы можете использовать собственные наборы данных. Посомтрите на формат разметки данных в предоставленном наборе. Для получения более-менее приемлемого результата необходима разметка минимум 10 изображений. При этом часть из них должна обязательно быть в валидации. 

# Блок 2
## Импорт необходимых библиотек
Если Вы еще не устанавливали Python и библиотеки, обратитесь к работе №0 для установки всего необходимого на Ваш ПК. 

``` python
import torch
from ultralytics import YOLO
import cv2
```

# Блок 3
## Подготовка файла настрек
Все настройки сети задаются в файлах yaml. Нам необходимо указать расположение набора данных для обучения, валидации и названия для классов. P.s. Класс в этом наборе данных один - дорожные ямы.
Запишите следющие данные в файл pothole_v8.yaml. Чтобы сделать это автоматически, можете использовать следующий код:

``` python
file_data = """
path: pothole_dataset_v8/
train: 'train/images'
val: 'valid/images'

# class names
names:
  0: 'pothole'

"""

filename = "pothole_v8.yaml"
myfile = open(filename, 'w')
myfile.write(file_data)
myfile.close()
```

# Блок 4
## Проверка поддержки GPU
Обучение без GPU может происходить несколько часов. Если у Вас есть видеокарта, проверьте, поддерживается ли она средствами PyTorch

``` python
import torch
print(torch.cuda.is_available())
```

# Блок 5
## Запуск обучения сети
Здесь автоматически скачается файл модели нейронной сети и запускается обучение. Оно займет около часа. Если хочется быстрее - уменьшите количество эпох, но тогда результат будет несколько хуже


``` python
# Выбираем нужную модель.
# https://docs.ultralytics.com/ru/models/yolov8/
# Мы используем самую новую YOLO v8 для локалзиации
# с минимальным количеством весовых коэффициентов (nano)
# также YOLO умеет: сегментировать и классифицировать изображения,
# определять позу человека и определять угол наклона объекта
model = YOLO('yolov8s.pt')

# Запуск обучения
results = model.train(
   data='pothole_v8.yaml',   # Файл расположением датасета
   imgsz=640,                # Размер входного изображения (все приводится к квадрату со стороной 640)
   epochs=10,                 # Количество эпох
   batch=16,                 # Количество данных в посылке
   name='yolov8n_custom')    # Имя сети (опционально)
```

# Блок 6
## Тестирование результата
Теперь можно запустить тесты. Выбираем картинку из валидации (а можете и свою загрузить) и запускаем predict.

``` python
# Тестируем результат на пользовательских картинках:
path = 'C:/Python/datasets/pothole_dataset_v8/valid/images/img-61_jpg.rf.a90cbd2da26633925c4bc0eb783ba58f.jpg'
results = model([path])

# Добавим немного визуализации
# Прочитаем изображение, для визуализации будем использовать OpenCV
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# В results[0] будут наши результаты работы
for r in results[0]:
    # Нас интерессуют все прямоугольники с объектами
    for box in r.boxes:
        # Получаем эти прямоугольники и рисуем на изображении
        b = box.xyxy[0]
        c = box.cls
        start_point = (int(b[0]), int(b[1]))
        end_point = (int(b[2]), int(b[3]))
        color = (255, 0, 0)
        thickness = 2

        image = cv2.rectangle(img, start_point, end_point, color, thickness)
        print(b)


cv2.imshow("frame", image)
```

# Блок 7
## В качестве заключения
Посмотрите на то, сколько слоёв имеет наша нейронная сеть и сколько в ней весовых коэффициентов

``` python
model.print()
```
