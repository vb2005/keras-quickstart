# Заргузка библиотеки
from ultralytics import YOLO
import cv2

# Выбираем нужную модель
model = YOLO('yolov8s.pt')

# Запуск обучения
results = model.train(
   data='pothole_v8.yaml',   # Файл расположением датасета
   imgsz=640,                # Размер входного изображения (все приводится к квадрату со стороной 640)
   epochs=10,                 # Количество эпох
   batch=16,                 # Количество данных в посылке
   name='yolov8n_custom')    # Имя сети (опционально)
   
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