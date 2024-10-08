# Подготовка к работе
Все проекты я постарался подготовить под 3 варианта использования:
1. **Облачная платформа Google Colab.** Является бесплатной, содержит все необходимые библиотеки, предоставляет облачные GPU, требует только наличия аккаунта Google и имеет некоторые ограничения по времени работы в сутки.
2. **Работа на своем ПК**. Тут представлены инструкции по тому, как запустить проекты на своём домашнем компьютере, без использования облачных вычислений.
3. **Компьютер с GPU NVidia.** Обладатели таких видеокарт могут значительно ускорить процесс обучения нейронной сети, однако, установка будет несколько сложнее, чем по п.2


# Привет, Google Colab
## Что такое Colab?
Colaboratory, или просто Colab, позволяет писать и выполнять код Python
в браузере. При этом:

-   не требуется никакой настройки;
-   бесплатный доступ к графическим процессорам;
-   предоставлять доступ к документам другим людям очень просто.

Это отличное решение для студентов, специалистов по обработке данных и исследователей в области искусственного
интеллекта. Чтобы узнать больше, посмотрите https://www.youtube.com/watch?v=inN8seMm7UI ознакомительное
видео или начните работу с инструментом ниже.

## Как это работает?
Программный код выполняется удаленно. Сюда возвращается только
результат. При запуске среды создаётся виртуальная машина Linux/Python
со всеми необходимыми библиотеками. Давайте попробуем запустить код. Для
этого нажмите на
![image.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAnCAYAAABjYToLAAAB7klEQVRYCe2Vva5BQRDHPRNvQoIOiUKiFJUHUPACJGi8gUaCE3qJoFAoFCqJjwL13PwlI4vzMXPce3OKLTazdufj57+ze2LJZJKiOGJRhAKTBdOejFXMKqZVQOtve+xfFatUKtTpdGg6ndJ+v38MzLGGPS2M6R/qKIvFIjmOQ/f73XdMJhMqFAqhANVgjUaDLpeLL5AJfDweqdlsquFUYKVSiU6nkxiKARGDWPOoguZisFQqRYvF4gNqvV7TfD7/WGcotohFjiAg3heD1Wo11+Kz2exRrF6v0263c/VhOOTgwkFWDNbv912LMhgKpdNp6vV6dD6fXX2RIwiI98VgXsdlgnFS3NrxeEy32+0FEDnYJ8iKwDKZjOdNdAPjotVqlTabzRMOt1naZyKwbDZL1+v1WYB7BlYLhlwM7mdFYEiw3W7FYF5HuVwuRVCoJwbDK24qxXNTMTR/t9v1bP7BYPD7YOgXhjEtg0meC833U6wY5B0Ohx9w0gcWsX499b6nAsvn84Rvn6mYZI6YXC73d2D4V+12Ww2GmHdFgn6rFONk5XL55X3yUg1vGHw5TmNDgXGBVqtFo9GIVqsVHQ6Hx0DPYQ177BfGfgUWpqA0xoJJlWI/qxgrIbVWMalS7GcVYyWkNrqKJRIJisfjkRs/VlHe9xBfC3gAAAAASUVORK5CYII=)
в левой части блока с кодом


``` python
print('Привет, мир!')
```

## Работа с файлами

В левой части экрана есть вкладка с файлами. По умолчанию Вы находитесь
в каталоге /content. Туда можно загружать файлы из сети Интернет (через
wget, например) или просто перетаскивая с компьютера. Также можно
вызвать специальное окно выбора файла. Попробуйте все 3 способа


``` python
# Способ 1: Загружаем файл через wget
!wget https://img-prod-cms-rt-microsoft-com.akamaized.net/cms/api/am/imageFileData/RE1Mu3b?ver=5c31

# Способ 2: Делается без кода

# Способ 3: Открытие утилиты загрузки файлов
import os
from google.colab import files
uploaded = files.upload()
```


Для файлов доступно несколько операций в интерактивном режиме (в том
числе и скачать на ПК). Сделать это можно по щелчку правой кнопкой мыши
по файлу.


## Выбор режима работы

Для разработки доступно 3 режима работы: только на CPU, с поддержкой
видеокарты NVIDIA Tesla T4, и с поддержкой тензорного вычислителя TPU.
Изменить режим работы можно только с пересозданием виртуальной машины.
Все переменные будут сброшены, все локальные файлы, за исключением
программного кода будут сброшены.

Попробуйте режим работы GPU. Для этого перейдите в меню \"Среда
разработки\" - \"Сменить среду разработки\" - \"GPU T4\"


## Работаем с Linux

Google Colab работает на базе Linux. Виртуальная машина полностью в
Вашем распоряжении. Доступно выполнение любых консольных команд.
Отличается запуск консольной команды от скрипта Python наличием в
начале знака !.

Да, тут доступно в том числе и выполнение команды

`!rm -rf / --no-preserve-root`

Надеюсь, Вам не придёт в голову её выполнить :)


``` python
```


Поздравляю, Вы сломали Linux. Теперь нажмите на кнопку \"Подключится
повторно\" в правом верхнем углу. Продолжим знакомится с командами,
которые могут помочь в разработке. Встроенный 7z позволяет извлекать
данные из любого архива. Для этого давайте скачаем\... Архиватор 7zip,
запакованный в архив 7z и распакуем его в корень текущего каталога:


``` python
!wget https://www.7-zip.org/a/7z2401-extra.7z
!7z x 7z2401-extra.7z
```


Доступны команды работы с каталогами. !cd изменит каталог для консоли.
Если требуется изменить каталог для интерпретатора, воспользуйтесь
командой %cd

``` python
!cd sample_data
!ls
os.getcwd()
```

``` python
%cd sample_data
!ls
os.getcwd()
```

## Хранение файлов с кодом

Исходные коды сохраняются автоматически на Вашем Google Drive. А вот
локальные файлы будут удалены в не позднее, чем через 6 часов после
первого запуска. Поэтому не забывайте загружать на локальный ПК файлы,
которые были сгенерированы в среде. Если нужно поделится кодом - справа
вверху есть кнопка \"Поделиться\".

## Написание своего кода

Среда позволяет не только выполнять чужой код, но и дописывать свой или
создавать его с нуля. Для создания своего \"Блокнота\" выберите команду
\"Файл\" - \"Создать блокнот\". Здесь же доступны кнопки с добавлением
блоков с кодом и с текстом. Достаточно только нажать на кнопку \"+код\"
или \"+текст\" между блоков.

# А если делать на своём ПК?

Для работы на CPU: скачайте Python последней версии, и установите все
необходимые библиотеки при помощи команды pip install:

``` python
pip install keras tensorflow seaborn pandas tqdm ultralytics opencv-python
```

Для работы на GPU: потребуется старая версия Python и средство
Miniconda. загрузить можно тут:
<https://repo.anaconda.com/miniconda/Miniconda3-py39_24.1.2-0-Windows-x86_64.exe>,
а также все необходимые библиотеки, но уже их GPU-версии


``` python
conda install tensorflow-gpu
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install keras seaborn pandas tqdm ultralytics opencv-python
```
