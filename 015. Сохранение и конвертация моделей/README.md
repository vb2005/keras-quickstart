# Работа №15
### Сохранение и конвертация моделей

В данной работе рассматривается экспорт модели в формате ONNX с дальнейшим запуском под .NET
### Сведения о работе

Необходимые библиотеки: `tf2oxx, onnx`

Количество заданий для самостоятельного выполнения: ![](https://img.shields.io/badge/1%20задание-red)

Время на выполнение: ![](https://img.shields.io/badge/90-минут-blue)

### Введение
Конвертация моделей из Keras в ONNX (Open Neural Network Exchange) может быть полезной по нескольким причинам:
1. Большая совместимость. ONNX - открытый формат, который имеет поддержку различных платформ (PyTorch, TensorFlow, MXNet, Microsoft ML, Apple Core ML и др.)
2. ONNX имеет более удобные средства для запуска моделей на GPU, а также поддерживает более широкий их спектр
3. Расширенная интеграция. Например, ONNX-модели могут быть использованы в Microsoft Azure, Amazon Web Services (AWS) и других облачных платформах.

Для начала работы установите необходимую библиотеку при помощи команды:

```pip install tf2onnx```

## Разработка модели
### Импорт библиотек

``` python
import onnx
import tf2onnx
import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
```

### Разработка модели

Формируем некоторую модель, с которой будем вести работу. Пусть у неё будет один нейрон на входе и один на выходе.
** Важно! Для ONNX последний слой должен иметь имя output. Иначе он не сможет собрать модель **

``` python
def build_model():
  model = Sequential()
  model.add(Input(shape = (1,)))
  model.add(Dense(5, activation='sigmoid'))
  model.add(Dense(500, activation='sigmoid'))
  model.add(Dense(1, activation='sigmoid'))
  model.compile(optimizer=Adam(),  loss='mse')
  return model

model = build_model()
model.output_names = ['output']
model.summary()
```

Проверим работу модели на экспериментальных данных. Результат работы в Keras и ONNX должен совпадать

``` python
a = np.ones((1))
model.predict(a)
```

### Сохранение модели

``` python
# Указываем размерность, тип и имя входного слоя
input_signature = [tf.TensorSpec([1], tf.float32, name='input_1')]

# Собираем модель ONNX версии 13
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature, opset=13)

# Сохраняем модель в файл
onnx.save(onnx_model, "/content/model.onnx")
```


## Запуск на C#

Рассмотрим запуск модели на C# (.NET 6.0 И выше).

1. Создайте консольный проект .NET Core
2. Перейдите в диспетчер пакетов Nuget
3. Установите OnnxRuntime
4. В файл `Program.cs` добавьте следующий код:

``` cs
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

 float Predict(float f1) {
    // Загрузка модели в RAM. Если нейронка используется несколько раз за проект,
    // перенесите инициализацию в код первого вызова
    InferenceSession session = new InferenceSession(@"model.onnx");

    // Формироваие входного тензора
    Tensor<float> data = new DenseTensor<float>(new[] { 1 });
    data[0] = f1;

    // Указание названия для входного слоя
    var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("input_1", data) };
    var inputMeta = session.InputMetadata;

    // Получение результата работы
    var results = session.Run(inputs).ToList();

    // Формирование выходного тензора
    DenseTensor<float> outputs = (DenseTensor<float>)results[0].Value;
    var outputs_1d = outputs.ToArray();

    // Выходной массив всегда одномерный. Чтобы его преобразовать в нужную размерность
    // удобнее всего использовать BlockCopy с созданием нового массива
    float[,] outputs_result = new float[1, 1];
    Buffer.BlockCopy(outputs_1d, 0, outputs_result, 0, 4);

    return outputs_result[0, 0];
}

float x = 10;
float y = Predict(x);


Console.WriteLine(y);
```

# Самостоятельная работа
## Задание №1
![](https://img.shields.io/badge/Задача%201-red)
Преобразуйте любую сверточную модель в формат ONNX, а также запустите её на C# с поддержкой GPU CUDA или DirectML.
