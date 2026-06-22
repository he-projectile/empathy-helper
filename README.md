# Emotion Assistant

**Emotion Assistant** — система автоматической классификации эмоций в коротких текстовых сообщениях.

Система помогает людям автоматически определять эмоцию, выраженную в коротком текстовом сообщении, и получать вероятности принадлежности ко всем эмоциональным классам.

Пример входных данных:

```json
{
  "text": "Сегодня у меня отличное настроение!"
}
```

Пример результата:

```json
{
  "label": "joy",
  "probs": {
    "joy": 0.82,
    "sadness": 0.04,
    "anger": 0.02,
    "neutral": 0.12
  }
}
```

---

# Быстрый старт

## 0. Клонирование модели

```bash
git clone https://github.com/he-projectile/empathy-helper .
```

## 1. Установка зависимостей

```bash
uv sync
```

## 2. Загрузка предобученной модели с Hugging Face

```bash
hf cp hf://he-projectile/empathy-helper-model/last.ckpt outputs/checkpoints/last.ckpt

hf cp hf://he-projectile/empathy-helper-model/metadata.pkl outputs/checkpoints/metadata.pkl
```


## 3. Использование модели

```bash
# Классификация текста
uv run python -m emotion_assistant.infer "I feel happy af"

# С явным указанием пути к модели
uv run python -m emotion_assistant.infer "I feel happy af" \
    --checkpoint outputs/checkpoints/last.ckpt \
    --metadata outputs/checkpoints/metadata.pkl
```

---

# Архитектура модели

Основная модель обучена на датасете **GoEmotions**:

* ~58 000 текстовых сообщений;
* 27 эмоциональных категорий;
* сообщения из Reddit.

Архитектура:

```
Текст → Токенизация → Embedding → Bidirectional LSTM → Классификатор → Вероятности эмоций
```


---

# Структура проекта

```
emotion_assistant/
  ├── data/              # Загрузка и подготовка данных
  ├── models/            # Архитектура моделей
  ├── inference/         # Инференс и экспорт
  ├── train.py           # Скрипт обучения
  └── infer.py           # Скрипт инференса

configs/                 # Конфигурации Hydra
plots/                   # Графики обучения
outputs/checkpoints/     # Сохраняемые модели
```

---

# Используемые технологии

* Python;
* PyTorch;
* PyTorch Lightning;
* Hydra;
* MLflow;
* uv.

---

# Расширенное использование

## Экспорт модели в ONNX

```bash
uv run python -m emotion_assistant.inference.export_onnx \
    --checkpoint outputs/checkpoints/last.ckpt \
    --metadata outputs/checkpoints/metadata.pkl \
    --output outputs/model.onnx
```

## Инференс из ONNX-модели

```bash
uv run python -m emotion_assistant.infer \
    "Сегодня отличный день" \
    --onnx outputs/model.onnx \
    --metadata outputs/checkpoints/metadata.pkl
```

---

# Обучение модели (опционально)

Для переобучения модели на своих данных:

## Подготовка данных

```bash
# Если используется GoEmotions
uv run python datasetDownload.py --output-dir data/raw
```

## Запуск обучения

```bash
# С параметрами по умолчанию
uv run python -m emotion_assistant.train

# С переопределением параметров
uv run python -m emotion_assistant.train trainer.max_epochs=20 data.batch_size=128
```

В процессе обучения:

1. Загружаются данные.
2. Строится словарь токенов.
3. Формируются обучающая, валидационная и тестовая выборки.
4. Обучается модель BiLSTM.
5. Рассчитываются метрики качества.
6. Логируются эксперименты в MLflow.
7. Сохраняются контрольные точки.

После завершения обучения:

```
outputs/checkpoints/last.ckpt
outputs/checkpoints/metadata.pkl
plots/train_loss.png
plots/val_loss.png
plots/macro_f1.png
```

## Логирование экспериментов

Метрики логируются в MLflow (адрес по умолчанию: `http://127.0.0.1:8080`):

* train loss;
* validation loss;
* accuracy;
* macro F1-score;
* гиперпараметры.

---

# Модель на Hugging Face

Предобученные модели доступны на Hugging Face:

**Репа:** [`he-projectile/empathy-helper-model`](https://huggingface.co/he-projectile/empathy-helper-model)

**Файлы модели:**
* `metadata.pkl` — метаданные (словарь токенов, индексы классов)
* `last.ckpt` — веса модели

Загрузка вручную:

```bash
# HF CLI
hf cp hf://he-projectile/empathy-helper-model/metadata.pkl outputs/checkpoints/metadata.pkl

hf cp hf://he-projectile/empathy-helper-model/last.ckpt outputs/checkpoints/last.ckpt
```
---

# Настройка разработки

## Установка зависимостей

```bash
uv sync
```

## Активация виртуального окружения

```bash
.venv\Scripts\activate
```

## Установка pre-commit хуков

```bash
pre-commit install
```

## Проверка качества кода

```bash
pre-commit run -a
```

## Запуск MLflow UI

Если вы хотите просматривать эксперименты и артефакты во время или после обучения, запустите MLflow UI локально:

```bash
# В консоли (порт можно изменить)
mlflow ui --host 127.0.0.1 --port 8080
```

По умолчанию MLflow сохранит треки в локальной директории `mlruns/`. Откройте в браузере:

```
http://127.0.0.1:8080
```

Если вы запускаете обучение через `uv run`, запустите MLflow UI в отдельном терминале перед или во время тренировки.

---

# Примечания

* **Инференс не требует обучения** — используйте предобученную модель с Hugging Face.
* **Обучение опционально** — переобучайте модель только при необходимости.
* **Быстрый старт** — загрузите модель и сразу начните использовать.

---
