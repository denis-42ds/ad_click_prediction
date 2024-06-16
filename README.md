# Название проекта: Ad click prediction

## Статус проекта: в работе

## Заказчик: natiscope

## Описание рабочих файлов и директорий:
- [ad_click_prediction.ipynb](https://github.com/denis-42ds/ad_click_prediction/blob/development/ad_click_prediction.ipynb) - рабочая тетрадь с исследованиями, визуализациями и текстовыми описаниями
- [requirements.txt](https://github.com/denis-42ds/ad_click_prediction/blob/development/requirements.txt) - список зависимостей, необходимых для работы проекта, а также их версии
- [research_functions.py](https://github.com/denis-42ds/ad_click_prediction/blob/development/research_functions.py) - скрипт с функциями для проведения исследования
- [run_jupyter.sh](https://github.com/denis-42ds/ad_click_prediction/blob/development/run_jupyter.sh) - скрипт для запуска тетрадки с исследованием
- [assets](https://github.com/denis-42ds/ad_click_prediction/tree/development/assets) - директория с сохранёнными артефактами
- [services](https://github.com/denis-42ds/ad_click_prediction/tree/development/services) - директория с приложением

## Установка зависимостей и просмотр исследования
```Bash
git clone https://github.com/denis-42ds/ad_click_prediction.git
cd ad_click_prediction
pip install -r requirements.txt
sh run_jupyter.sh
```

## Запуск FastAPI-микросервиса

```
git clone https://github.com/denis-42ds/ad_click_prediction.git
cd ad_click_prediction
docker compose up --build
```

Для просмотра документации API и совершения тестовых запросов пройти по ссылке: [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)
<br>Для остановки приложения: ```docker compose down```

## Описание проекта
<br>Требуется получить на выходе прогноз с высокой точностью вероятности клика по рекламному объявлению.
<br>На предикт ML модели выдается выборка объявлений доступных в текущий момент к показу, модель выдает по ним вероятность клика.

## Цель проекта
Построение модели для прогнозирования кликов по рекламному объявлению
	
## Ход исследования
- Знакомство с данными
- Исследовательский анализ данных
- Предобработка данных (при необходимости)
- Разработка дополнительных признаков (при необходимости)
- Построение baseline
- Обучение нескольких моделей
- Выбор лучшей модели
- Проверка важности признаков
- Проверка лучшей модели на отложенной выборке
- Заключение о проделанном исследовании

## Используемые инструменты
- python: pandas, seaborn, matplotlib, phik, category_encoders, LightGBM;
- mlflow;
- postgresql;
- bash

## Заключение:
- 
