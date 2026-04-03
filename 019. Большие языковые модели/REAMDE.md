# Обработка текста в задачах машинного обучения

В рамках данной работы рассмотрим высокоуровневые средства для использования и обучения моделей NLP.

Сама по себе обработка естественого языка (NLP) представляет собой область лингвистики и машинного обучения, которая изучает все, что связано с естественными языками. Главная цель NLP не просто понимать отдельные слова, но и иметь возможность понимать контекст, в котором эти слова находятся.

Список типичных NLP-задач с некоторыми примерами:

**Классификация предложений:** определить эмоциональную окраску отзыва, детектировать среди входящих писем спам, определить грамматическую правильность предложения или даже проверить, являются ли два предложения связанными между собой логически

**Классификация каждого слова в предложении:** вычленить грамматические составляющие предложения (существительное, глагол, прилагательное) или определить именованные сущности (персона, локация, организация)

**Генерация текста:** закончить предложение на основе некоторого запроса, заполнить пропуски в тексте, содержащем замаскированные слова

**Сформулировать ответ на вопрос:** получить ответ на заданный по тексту вопрос

**Сгенерировать новое предложение исходя из предложенного:** перевести текст с одного языка на другой, выполнить автоматическое реферирование текста




# Начало работы

Прежде, чем начать работу, поставим необходимые нагрузки.

Это всё надо на чём-то обучать. Делать это будем на PyTorch. Ставим версию под себя:
'pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu126'

Для реализации наших моделей будем использовать высокоуровневые средства для построения моделей-трансформеров и средства загрузки наборов данных с HuggingFace
'pip install datasets transformers[torch]'


Я работаю в Spyder, поэтому еще и:
'pip install spyder'

# Особенности запуска Torch и Spyder
Если нелегкая столкнёт Вас с этой проблемой, то путь её решения весьма специфичен:

1. Spyder использует PyQt, который, в свою очередь использует свой локальный MS VS Redist 2017 года. 
2. Torch использует более новые наборы (от 2020 года). Но, поскольку Spyder стартует со своим PyQt раньше, то и зависимости в RAM он заносит свои.
2. Поэтому, находим каталог \Lib\site-packages\PyQt5\Qt5\bin и заменяем в нём все 'msvcp*' и 'vcruntime*' на аналоги из C:\Windows\SystemWOW64

# Начало работы
Рассмотрим примеры работы с библиотекой transformers для  различных задач. Библиотека позволяет строить pipeline при помощи всего одной команды, выбирая корректную модель и входные данные. Рассмотрим несколько примеров:

## Задача классификации текста
Выберите модель для класификации на сайте:
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending

Запустите код, указав целевую модель. Проверьте качество работы модели
``` python
from transformers import pipeline
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
classifier("Beauiful day")
```


## Задача классификации текста
Выберите модель для класификации на сайте:
https://huggingface.co/models?pipeline_tag=text-classification&sort=trending

Запустите код, указав целевую модель. Проверьте качество работы модели
``` python
from transformers import pipeline
classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
classifier("Beauiful day")
```

## Классификация "на лету"

Существует ряд моделей, в которых возможно сделать классификацию "на лету" для оценки наличия некоторой оценки в тексте. Требуется указать целевой текст и классы

``` python 
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

## Генерация текста

Рассмотрим задачу генерации текста на основе последовательности. Воспользуемся стандартной моделью. Либо можно в явном виде указать целевую: https://huggingface.co/models?pipeline_tag=text-generation&sort=trending

``` python 
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

## Заполнение пропусков

Ряд моделей умеют заполнять пропуски в целевых предложениях. Можно вывести сразу несколько предложений

``` python 
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

## Авто-реферирование

Для решения задачи сжатия текста и формирования основного тезиса работы сущестует отдельная группа моделей

``` python 
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer(
    """
    America has changed dramatically during recent years. Not only has the number of 
    graduates in traditional engineering disciplines such as mechanical, civil, 
    electrical, chemical, and aeronautical engineering declined, but in most of 
    the premier American universities engineering curricula now concentrate on 
    and encourage largely the study of engineering science. As a result, there 
    are declining offerings in engineering subjects dealing with infrastructure, 
    the environment, and related issues, and greater concentration on high 
    technology subjects, largely supporting increasingly complex scientific 
    developments. While the latter is important, it should not be at the expense 
    of more traditional engineering.

    Rapidly developing economies such as China and India, as well as other 
    industrial countries in Europe and Asia, continue to encourage and advance 
    the teaching of engineering. Both China and India, respectively, graduate 
    six and eight times as many traditional engineers as does the United States. 
    Other industrial countries at minimum maintain their output, while America 
    suffers an increasingly serious decline in the number of engineering graduates 
    and a lack of well-educated engineers.
"""
)
```

# Этапы работы языковых моделей

## Выбор модели

## Токенизатор

Рассмотрим использование токенизатора BERT-подобной модели. 

``` python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer("Using a Transformer network is simple")
```

Начальный и конечный токен - служебные. Практически все остальные слова закодировались целыми токенами. Давайте проверим, как выполнилось это кодирование:

``` python
tokens = tokenizer.tokenize(sequence)
print(tokens)
```

Теперь все токены переведем в идентификаторы

``` python
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
```

И проверим, нормально при просходит декодирование

``` python
decoded_string = tokenizer.decode(ids)
print(decoded_string)
```

## Изучение Embedding-слоёв

Материал готовится...

## Fine-tune существующей модели

``` python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments
from transformers import pipeline
from datasets import load_dataset


# Выбор модели
model_name = "gpt2-medium"

# Загружаем токенизатор
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Добавляем 
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Загружаем модель для языкового моделирования (Causal Language Modeling)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Отправляем модель на GPU, если он доступен
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Используется устройство: {device}")



# Загрузка датасета для обучения модели
dataset =  load_dataset("Den4ikAI/russian_cleared_wikipedia")

# Формируем набор данных. Требуется из текста сделать наборы токенов
def tokenize_function(examples):
    # Токенизируем текст. padding='max_length' делает все последовательности одной длины.
    # truncation=True обрезает слишком длинные тексты.
    tokens = tokenizer(
        examples["sample"],
        padding="max_length",
        truncation=True,
        max_length=128  # Максимальная длина последовательности. Можно увеличить до 512 или 1024.
    )
    # Для обучения с учителем (next token prediction) нам нужны labels.
    # Они будут точно такими же, как input_ids.
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["sample"])
tokenized_dataset_train = tokenized_dataset['train']


#%%
training_args = TrainingArguments(
    output_dir="./my_gpt2_model",       # Папка, куда будет сохранена модель
    num_train_epochs=3,                 # Количество эпох (проходов по всему датасету) 
    per_device_train_batch_size=4,      # Размер батча (зависит от вашей видеокарты)
    save_steps=500,                     # Сохранять промежуточные чекпоинты каждые N шагов
    save_total_limit=2,                 # Хранить только последние 2 чекпоинта, чтобы не забить диск
    logging_steps=100,                  # Логировать результат каждые N шагов
    learning_rate=2e-5,                 # Скорость обучения (чем меньше, тем бережнее, но медленнее)
    weight_decay=0.01,                  # Регуляризация для предотвращения переобучения
)

# Создаем объект тренера
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_train,    
    #eval_dataset=tokenized_dataset_valid # Сюда можно подать отдельную выборку для валидации
)

trainer.train()

#%%
# Создаем пайплайн для генерации текста
generator = pipeline('text-generation', model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

# Задаем начальный текст
prompt = "Математика"

# Генерируем продолжение
output = generator(prompt, max_length=100, num_return_sequences=10, temperature=0.7, do_sample=True)
print(output[0]['generated_text'])
```

# Задание на лабораторную работу

1. Выполнить представленные кейсы (кроме последнего)
2. На сайте HuggingFace найти актуальные наборы данных для русского языка
3. Исследуйте различные модели для генерации текста. Сделайте выводы о способности их к решению задач до и после Fine-tune


# Ссылка на литературу

1. https://huggingface.co/learn/llm-course/ru/chapter0/1
