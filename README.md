# Название проекта: Ad click prediction

## Статус проекта: завершён

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
cd ad_click_prediction/services
docker compose up --build
```

Для просмотра документации API и совершения тестовых запросов пройти по ссылке: [http://127.0.0.1:8081/docs](http://127.0.0.1:8081/docs)
<br>Доступ к экспозиции метрик `Prometheus`: [http://localhost:8081/metrics](http://localhost:8081/metrics)
<br>Доступ к веб-интерфейсу `Prometheus`: [http://localhost:9090](http://localhost:9090)
<br>Доступ к веб-интерфейсу `Grafana`: [http://localhost:3000](http://localhost:3000)
<br>Для остановки приложения: ```docker compose down``` или `Press CTRL+C to quit`

[Демонстрация работы приложения](https://drive.google.com/file/d/1EPmL2vz5csGTA4XTIGafRKn77hHohFZq/view?usp=sharing)

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
- bash;
- fastapi, grafana, prometheus

## Заключение:

Итоги проделанного исследования.

**Выбор целевых метрик**

- `ROC-AUC` (Receiver Operating Characteristic - Area Under the Curve) является метрикой,
  <br>которая оценивает качество бинарной классификации, учитывая полноту (True Positive Rate) и
  <br>специфичность (True Negative Rate) модели.
  <br>ROC-AUC измеряет способность модели различать между классами и представляет площадь под кривой ROC.
  <br>ROC-AUC особенно полезна, когда важно контролировать баланс между ложно-положительными и ложно-отрицательными предсказаниями.
  <br>Она также устойчива к несбалансированным классам.
- `F1-мера` является метрикой, которая оценивает точность и полноту модели для бинарной классификации.
  <br>F1-мера является гармоническим средним между точностью (precision) и полнотой (recall).
  <br>Она особенно полезна, когда важно достичь баланса между точностью и полнотой.
  <br>F1-мера хорошо работает, когда классы сбалансированы или когда важно минимизировать как ложно-положительные, так и ложно-отрицательные предсказания.

**Загрузка и ознакомление с данными**

- шаг выполнен полностью, трудностей не возникло;
- данные загружены;
- изучены типы данных;
- приняты решения о дальнейшей предобработке;

**Исследовательский анализ данных**

- шаг выполнен полностью, трудностей не возникло;
- исследованы зависимости и корреляции признаков;
- приняты решения об удалении неинформативных признаков;

**Построение базовой модели**

- в качестве базовой модели выбрана модель логистической регрессии;
- обусловлено это тем, что данная модель имеет небольшое количество гиперпараметров
  <br>и короткое время обучения;

**Генерация дополнительных признаков**

- шаг выполнен полностью, трудностей не возникло;
- разработаны несколько дополнительных признаков на основании уже существующих;

**Обучение моделей**

- шаг выполнен полностью, трудностей не возникло;
- обучены несколько различных моделей:
  - логистическая регрессия;
  - случайный лес;
  - градиентный бустинг;
  - градиентный бустинг с подбором гиперпараметров;
- наиболее высокое качество предсказания по выбранным метрикам показала модель градиентного бустинга.

**Итоговая оценка качества предсказания лучшей модели и анализ важности её признаков**

- шаг выполнен полностью, трудностей не возникло;
- на тестовой выборке модель показала неплохой результат ROC AUC = 0.82;
- модель верно предсказала 39092 отрицательных примеров и 34393 положительных примера;
- модель ошибочно предсказала 12124 отрицательных примеров как положительные и 13430 положительных примеров как отрицательные;
- наиболее важными для модели признаками определены:
  - процент видимости объявления;
  - значение ctr объявления;
  - ставка в рублях объявления;
  - тематика объявления;
  - значение таргетинга по поведению.

**Рекомендации**

- для получения более высокого качества предсказаний требуются дополнительные данные:
  - данные о пользователях;
  - временная последовательность;
  - возможно, ещё какая-то информация, которой обладает заказчик.

**Общий итог**

**Для решения поставленной задачи оптимальным образом подходит модель градиентного бустинга LightGBM**
