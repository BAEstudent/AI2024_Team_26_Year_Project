# Checpoint 06. DL experiments
---
## 1. Выбор архитектуры
Мы  решили остановиться на выборе архитектуры EfficientNetV2 L (input 480x480), показавшей лучшее качество на предыдущем этапе работы над проектом.

Для экспериментов мы решили выбрать несколько способов улучшить качество модели:
1) Заморажитвать слои во время обучения;
2) Использовать различные learning rate schedulers;
3) Добавлять дополнительные слои в архитектуру модели.

Результаты экспериментов представлены в пункте 2.


## 2. Результаты экспериментов: сравнение DL-моделей

В таб. 1 представлены описания обученных моделей и их качество на тестовой выборке. Для всех моделей начальный learning rate был выбран равным 1e-5. Для обучения размер батча равен 4, обучение длилось максимально 15 эпох. Обучающий датасет был разбит на обучающую и валидационную выборки в пропорции 3:1. Порог раннего останова, early_stop_patience, равен 2 эпохам.

**Таб. 1**
| Model # | Additional layers | N-Freeze | Scheduler | Accuracy (%) | Precision (%) | Recall (%) |
|:---:|:---:|:--:|:---------:|:------------:|:-------------:|:----------:|
|  1  |  -  | 0  | ReduceLROnPlateau(patience=1, factor=0.5)  | 97.57   | 97.29  | 98.02  |
|  2  |  -  | 0  | CosineAnnealingLR  | 96.52  | 95.76  | 96.82  |
|  3  |  2 linear layers with dropout  | 0  | ReduceLROnPlateau(patience=1, factor=0.5)                          | 96.62  | 95.30  | 96.69  |
|  4  |  2 linear layers with dropout  | 0  |   CosineAnnealingLR   | 94.21  | 89.33  | 94.87  |
|  5  |  -  | 1  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 95.88  | 95.38  | 96.68  |
|  6  |  -  | 1  |   CosineAnnealingLR   | 97.27  | 95.81  | 97.51  |
|  7  |  2 linear layers with dropout  | 1  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 95.90  | 94.80  | 95.52  |
|  8  |  2 linear layers with dropout  | 1  |   CosineAnnealingLR   | 96.83  | 96.24  | 96.89  |
|  9  |  -  | 2  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 96.52  | 93.55  | 96.82  |
|  10  |  -  | 2  |   CosineAnnealingLR   | 96.43  | 95.04  | 96.98  |
|  11  |  2 linear layers with dropout  | 2  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 96.41  | 94.43 | 97.01  |
|  12  |  2 linear layers with dropout  | 2  |   CosineAnnealingLR   | 97.01  | 95.07  | 97.56  |
|  13  |  -  | 3  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 96.04  | 93.73 | 96.42  |
|  14  |  -  | 3  |   CosineAnnealingLR   | 96.44  | 94.86 | 97.24  |
|  15  |  2 linear layers with dropout  | 3  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 96.04  | 94.47 | 96.45  |
|  16  |  2 linear layers with dropout  | 3  |   CosineAnnealingLR   | 96.44  | 96.05 | 96.73  |
|  17  |  -  | 4  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 95.80  | 94.40 | 96.90  |
|  18  |  -  | 4  |   CosineAnnealingLR   | 97.01  | 96.42 | 97.46  |
|  19  |  2 linear layers with dropout  | 4  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 96.21  | 95.08 | 97.05  |
|  20  |  2 linear layers with dropout  | 4  |   CosineAnnealingLR   | 96.93  | 95.73 | 96.82  |
|  21  |  -  | 5  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 95.65  | 94.00 | 96.54  |
|  22  |  -  | 5  |   CosineAnnealingLR   | 96.81  | 95.29 | 96.88  |
|  23  |  2 linear layers with dropout  | 5  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 94.72  | 93.35 | 95.18  |
|  24  |  2 linear layers with dropout  | 5  |   CosineAnnealingLR   | 95.32  | 92.63 | 96.09  |
|  25  | -  | 6  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 93.75  | 91.20 | 94.74  |
|  26  | -  | 6  |   CosineAnnealingLR   | 92.87  | 89.30 | 94.79  |
|  27  | 2 linear layers with dropout  | 6  |   ReduceLROnPlateau(patience=1, factor=0.5)   | 92.97  | 89.68 | 93.69  |
|  28  | 2 linear layers with dropout  | 6  |   CosineAnnealingLR   | 94.63  | 93.12 | 95.60  |


**Описание:**
- *Architecture Details*: Описание дополнительных слоев, добавленных в оригинальную архитектуру;
- *N-Freeze*: Число замороженных на время обучения блоков слоев;
- *Scheduler*: Learning rate scheduler;
- *Accuracy, Precision, Recall*: Представлен лучший результат за цикл обучения, расчитанный на валидационной выборке.

## 3. Сравнение лучшей DL-модели с бейзлайном и лучшей ML-моделью

**Таб. 2**
| Model # | Accuracy (%) |
|:---:|:------------:|
| Бейзлайн | 84.7 |
| Лучшая ML-модель | 95.9 |
| Лучшая DL-модель | 97.6 |

## 4. Выводы

Лучшее качество показала модель #1. Удалось достигнуть более высокого качества, чем с использованием ML моделей.

Увеличение числа замороженных блоков слоев не дало увеличения качества моделей. Добавление дополнительных слоев так же не дало роста качества. CosineAnnealingLR в среднем давал лучшее качество на валидационной выборке.