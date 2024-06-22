"""Класс FastApiHandler, который обрабатывает запрос и возвращает предсказание."""

import json
import joblib
import logging
import operator
import numpy as np
import pandas as pd
from constants import MODEL_PATH, REQUIRED_MODEL_PARAMS, ADDITIONAL_VARIABLES

logging.basicConfig(level=logging.INFO)

class FastApiHandler:

    def __init__(self):
        """Инициализация переменных класса."""

        # типы параметров запроса для проверки
        self.param_types = {
            'ad_id': str,
            'model_params': dict
        }

        self.load_model(model_path=MODEL_PATH)

    def load_model(self, model_path: str):
        """Загрузка обученной модели прогноза совершения клика.

        Args:
            model_path (str): Путь до модели.
        """
        try:
            self.model = joblib.load(model_path)
            logging.info('Model loaded successfully')
            return True
        except Exception as e:
            logging.error(f"Failed to load the model: {e}")
            return False

    def click_predict(self, model_params: dict) -> float:
        """Получение прогноза клика.

        Args:
            model_params (dict): Параметры для модели.

        Returns:
            float: вероятность клика.
        """
        # добавление расчётных параметров
        model_input = model_params
        model_input.update(ADDITIONAL_VARIABLES)
        
        model_input = pd.DataFrame([model_input])

        return self.model.predict_proba(model_input)[0][1]

    def check_required_query_params(self, query_params: dict) -> bool:
        """Проверяем параметры запроса на наличие обязательного набора.

        Args:
            query_params (dict): Параметры запроса.

        Returns:
                bool: True — если есть нужные параметры, False — в ином случае.
        """
        if 'ad_id' not in query_params or 'model_params' not in query_params:
            return False

        if not isinstance(query_params['ad_id'], self.param_types['ad_id']):
            return False

        if not isinstance(query_params['model_params'], self.param_types['model_params']):
            return False

        return True


    def check_required_model_params(self, model_params: dict) -> bool:
        """Проверяем параметры для получения предсказаний.

        Args:
            model_params (dict): Параметры для получения предсказаний моделью.

        Returns:
            bool: True — если есть нужные параметры, False — иначе
        """
        if set(model_params.keys()) == set(REQUIRED_MODEL_PARAMS):
            return True
        return False


    def validate_params(self, params: dict) -> bool:
        """Проверка корректности параметров запроса и параметров модели.

        Args:
            params (dict): Словарь параметров запроса.

        Returns:
             bool: True — если проверки пройдены, False — иначе
        """

        if self.check_required_query_params(params):
                print("All query params exist")
        else:
                print("Not all query params exist")
                return False

        if self.check_required_model_params(params['model_params']):
                print("All model params exist")
        else:
                print("Not all model params exist")
                return False
        return True


    def handle(self, params):
        """Функция для обработки запросов API.

        Args:
            params (dict): Словарь параметров запроса.

        Returns:
            dict: Словарь, содержащий результат выполнения запроса.
        """
        try:
            # валидация запроса к API
            if not self.validate_params(params):
                logging.error("Error while handling request")
                response = {"Error": "Problem with parameters"}
            else:
                model_params = params['model_params']
                ad_id = params['ad_id']
                logging.info(f"Predicting for ad_id: {ad_id} and model_params:\n{model_params}")
                # получение предсказания модели
                probability = self.click_predict(model_params)
                response = {
                        "ad_id": ad_id,
                        "click_probability": round(probability, 2),
                        "click_prediction": int(probability > 0.5)
                    }
                logging.info(response)
        except KeyError as e:
            logging.error(f"KeyError while handling request: {e}")
            return {"Error": "Missing key in request"}
        except Exception as e:
            logging.error(f"Error while handling request: {e}")
            return {"Error": "Problem with request"}
        else:
            return json.dumps(response)

if __name__ == "__main__":

    # создание тестового запроса
    test_params = {
        "ad_id": '99999999',
        "model_params": {
                    "region_id": 38.0,
                    "city_id": 590.0,
                    "tags_cont": 0.0,
                    "tags_bhv": 0.5,
                    "rubrica": 1.812,
                    "rate": 1.7,
                    "ctr_sort": 1.3072,
                    "rv_perc": 66.41,
                    "slider": 1.0
        }
    }

    # создание обработчика запросов для API
    handler = FastApiHandler()

    # осуществление тестового запроса
    response = handler.handle(test_params)
    print(f"Response: {response}")

