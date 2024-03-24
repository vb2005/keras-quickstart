# Super Resolution RAN
Перед Вами генеративно-состязательная сеть, которая позволяет повысить разрешение изображения в 16 раз. Для обучения такой сети достаточно иметь набор изображений и набор точно таких же изображений, только уменьшенных в 16 раз.
* Данная работа основана на проектах:# https://github.com/tensorlayer/SRGAN,  https://github.com/AvivSham/SRGAN-Keras-Implementation. Пожалуйста, при создании своих проектов оставляйте ссылки на авторов, и соблюдайте условия их лицензионных требований!*
Мы будем использовать набор данных DIV2K, состоящий из нескольких тысяч изображений, а для тестирования сети Вы можете выбрать любое изображение с низким разрешением.

Необходимо загрузить набор данных для обучения по ссылке, и распаковать его в рабочий каталог:
http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip

# Блок 1
## Подключение библиотек

``` python
# Импортируем необходимые библиотеки
import os
import cv2
import numpy as np
from numpy import array
from keras import Model
from keras.layers import *
from keras.applications import VGG19
``` 

# Блок 2
## Загрузка и чтение данных
Сеть работает с изображениями размером [100; 100; 3] и [25; 25; 3]
``` python
base_size = 100
crop = 25
```
``` python
# Метод для загрузки картинок в форматах (100; 100; 3) и (25;25;3)

# Формирование изображений в высоком разрешении
def hr_images(images):
    images_hr = array(images)
    return images_hr

# Формирование изображений низкого разрешения
def lr_images(images_real , downscale):
    images = []
    for img in  range(len(images_real)):
        images.append(cv2.resize(images_real[img], (crop,crop)))
    images_lr = array(images)
    return images_lr

# Перевод данных из диапазона 0..255 в диапазон -1..1
def normalize(input_data):
    return (input_data.astype(np.float32) - 127.5)/127.5

# Перевод данных из диапазона -1..1 в диапазон 0..255
def denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)

# Получаем список файлов
def load_path(path):
    directories = []
    if os.path.isdir(path):
        directories.append(path)
    for elem in os.listdir(path):
        if os.path.isdir(os.path.join(path,elem)):
            directories = directories + load_path(os.path.join(path,elem))
            directories.append(os.path.join(path,elem))
    return directories

# Читаем данные из каталога
def load_data_from_dirs(dirs, ext):
    files = []
    file_names = []
    count = 0
    for d in dirs:
        for f in os.listdir(d):
            if f.endswith(ext):
                image = cv2.resize(cv2.imread(os.path.join(d,f)), (base_size,base_size))
                if len(image.shape) > 2:
                    files.append(image)
                    file_names.append(os.path.join(d,f))
                count = count + 1
    return files

def load_data(directory, ext):
    files = load_data_from_dirs(load_path(directory), ext)
    return files

def load_training_data(directory, ext, number_of_images = 1000, train_test_ratio = 0.8):
    number_of_train_images = int(number_of_images * train_test_ratio)

    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        raise
    test_array = array(files)
    if len(test_array.shape) < 3:
        print("Images are of not same shape")
        print("Please provide same shape images")
        raise

    x_train = files[:number_of_train_images]
    x_test = files[number_of_train_images:number_of_images]

    x_train_hr = hr_images(x_train)
    x_train_hr = normalize(x_train_hr)

    x_train_lr = lr_images(x_train, 4)
    x_train_lr = normalize(x_train_lr)

    x_test_hr = hr_images(x_test)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(x_test, 4)
    x_test_lr = normalize(x_test_lr)

    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


def load_test_data_for_model(directory, ext, number_of_images = 100):
    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        raise

    x_test_hr = hr_images(files)
    x_test_hr = normalize(x_test_hr)

    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr, x_test_hr

def load_test_data(directory, ext, number_of_images = 100):
    files = load_data_from_dirs(load_path(directory), ext)

    if len(files) < number_of_images:
        print("Number of image files are less then you specified")
        print("Please reduce number of images to %d" % len(files))
        raise

    x_test_lr = lr_images(files, 4)
    x_test_lr = normalize(x_test_lr)

    return x_test_lr
```
``` python
# Грузим картинки
lr_ip = Input(shape=(25,25,3))
hr_ip = Input(shape=(100,100,3))
x_train_lr, x_train_hr, x_test_lr, x_test_hr = load_training_data('/content/DIV2K_train_HR/', '.png', 800, 0.8)
print(x_train_lr.shape, x_train_hr.shape)
```
``` python
# Посмотрим на набор данных
print(x_train_lr.shape)
```
# Блок 3
## Описание сети GAN
``` python
# Residual block
def res_block(ip):

    res_model = Conv2D(64, (3,3), padding = "same")(ip)
    res_model = BatchNormalization(momentum = 0.5)(res_model)
    res_model = PReLU(shared_axes = [1,2])(res_model)

    res_model = Conv2D(64, (3,3), padding = "same")(res_model)
    res_model = BatchNormalization(momentum = 0.5)(res_model)

    return add([ip,res_model])

# Upscale the image 2x
def upscale_block(ip):
    up_model = Conv2D(256, (3,3), padding="same")(ip)
    up_model = UpSampling2D( size = 2 )(up_model)
    up_model = PReLU(shared_axes=[1,2])(up_model)

    return up_model
num_res_block = 16

# Generator Model
def create_gen(gen_ip):
    layers = Conv2D(64, (9,9), padding="same")(gen_ip)
    layers = PReLU(shared_axes=[1,2])(layers)
    temp = layers
    for i in range(num_res_block):
        layers = res_block(layers)
    layers = Conv2D(64, (3,3), padding="same")(layers)
    layers = BatchNormalization(momentum=0.5)(layers)
    layers = add([layers,temp])
    layers = upscale_block(layers)
    layers = upscale_block(layers)
    op = Conv2D(3, (9,9), padding="same")(layers)
    return Model(inputs=gen_ip, outputs=op)
```
``` python
#Small block inside the discriminator
def discriminator_block(ip, filters, strides=1, bn=True):

    disc_model = Conv2D(filters, (3,3), strides, padding="same")(ip)
    disc_model = LeakyReLU( alpha=0.2 )(disc_model)
    if bn:
        disc_model = BatchNormalization( momentum=0.8 )(disc_model)
    return disc_model

# Discriminator Model
def create_disc(disc_ip):
    df = 64

    d1 = discriminator_block(disc_ip, df, bn=False)
    d2 = discriminator_block(d1, df, strides=2)
    d3 = discriminator_block(d2, df*2)
    d4 = discriminator_block(d3, df*2, strides=2)
    d5 = discriminator_block(d4, df*4)
    d6 = discriminator_block(d5, df*4, strides=2)
    d7 = discriminator_block(d6, df*8)
    d8 = discriminator_block(d7, df*8, strides=2)

    d8_5 = Flatten()(d8)
    d9 = Dense(df*16)(d8_5)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(disc_ip, validity)
```
``` python
# Attach the generator and discriminator
def create_comb(gen_model, disc_model, vgg, lr_ip, hr_ip):
    gen_img = gen_model(lr_ip)
    gen_features = vgg(gen_img)
    disc_model.trainable = False
    validity = disc_model(gen_img)
    return Model([lr_ip, hr_ip],[validity,gen_features])
```
``` python
def build_vgg():
  vgg = VGG19(weights="imagenet",input_shape =(base_size,base_size,3),include_top=False)
  outputs = [vgg.layers[9].output]
  return Model(vgg.input, outputs)

generator = create_gen(lr_ip)
discriminator = create_disc(hr_ip)
discriminator.compile(loss="binary_crossentropy", optimizer="adam",
  metrics=['accuracy'])
vgg = build_vgg()
vgg.trainable = False
gan_model = create_comb(generator, discriminator, vgg, lr_ip, hr_ip)
gan_model.compile(loss=["binary_crossentropy","mse"], loss_weights=
  [1e-3, 1], optimizer="adam")

```

# Блок 4
## Подготовка обучающей выборки и тестового изображения
``` python
# Разделение на мини-батчи
batch_size = 20
train_lr_batches = []
train_hr_batches = []
for it in range(int(x_train_hr.shape[0] / batch_size)):
    start_idx = it * batch_size
    end_idx = start_idx + batch_size
    train_hr_batches.append(x_train_hr[start_idx:end_idx])
    train_lr_batches.append(x_train_lr[start_idx:end_idx])
train_lr_batches = np.array(train_lr_batches)
train_hr_batches = np.array(train_hr_batches)
```

``` python
# Метод для тестирования сети.
# Не забудьте взять картинку для эксперимента и указать к ней путь
def test(e):
  b = cv2.imread('kremlin.png')
  a = np.zeros((b.shape[0]*4,b.shape[1]*4,3))
  b = normalize(b)

  for x in range(int(b.shape[0]/25)):
    for y in range(int(b.shape[1]/25)):
      l = [b[(x*25):(x*25+25),(y*25):(y*25+25), :]]
      l = np.array(l)
      #print(l.shape)
      img = generator.predict(l, verbose = 0)
      xs = x*100
      ys = y*100
      a[xs:(xs+100), ys:(ys+100)] = img[0]
  a = denormalize(a)
  cv2.imwrite(str(e) + ".jpg", a)
```

# Блок 5
## Запуск обучения

``` python
# Запуск обучения сети
epochs = 100
for e in range(epochs):
    gen_label = np.zeros((batch_size, 1))
    real_label = np.ones((batch_size,1))
    g_losses = []
    d_losses = []
    for b in range(len(train_hr_batches)):
        lr_imgs = train_lr_batches[b]
        hr_imgs = train_hr_batches[b]
        gen_imgs = generator.predict_on_batch(lr_imgs)      
        discriminator.trainable = True


        d_loss_gen = discriminator.train_on_batch(gen_imgs, gen_label )
        d_loss_real = discriminator.train_on_batch(hr_imgs, real_label)
        discriminator.trainable = False
        d_loss = 0.5 * np.add(d_loss_gen, d_loss_real)
        image_features = vgg.predict(hr_imgs, verbose = 0)

        g_loss, _, _ = gan_model.train_on_batch([lr_imgs, hr_imgs], [real_label, image_features])

        d_losses.append(d_loss)
        g_losses.append(g_loss)

    # каждые 4 эпохи формируем новый результат для нашей картинки    
    if (e%4 == 0):
      test(e)

    g_losses = np.array(g_losses)
    d_losses = np.array(d_losses)

    g_loss = np.sum(g_losses, axis=0) / len(g_losses)
    d_loss = np.sum(d_losses, axis=0) / len(d_losses)
    print("epoch:", e+1 ,"g_loss:", g_loss, "d_loss:", d_loss)
```

