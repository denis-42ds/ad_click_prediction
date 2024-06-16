"""Класс FastApiHandler, который обрабатывает запрос и возвращает предсказание."""

import json
import joblib
import logging
import operator
import numpy as np
import pandas as pd
from constants import MODEL_PATH, REQUIRED_MODEL_PARAMS

logging.basicConfig(level=logging.INFO)

class FastApiHandler:

    def __init__(self):
        """Инициализация переменных класса."""

        # типы параметров запроса для проверки
        self.param_types = 'model_params': dict

        # список необходимых параметров модели
        self.required_model_params = REQUIRED_MODEL_PARAMS

        self.load_cost_model(model_path=MODEL_PATH)

    def load_model(self, model_path: str):
        """Загрузка обученной модели предсказания стоимости квартиры.

        Args:
            model_path (str): Путь до модели.
        """
        try:
            self.model = joblib.load(model_path)
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
        model_input = {}

        for feature_pair in FEATURE_OPERATIONS:
            feature1, feature2, operation = feature_pair
            if feature1 in model_params:
                if feature2 is None:
                    model_input[feature1] = model_params[feature1]
                elif feature2 in model_params:
                    if operation == operator.truediv:
                        model_input[f'{feature1}_{feature2}_ratio'] = operation(model_params[feature1], model_params[feature2])
                    elif operation == np.exp:
                        model_input[f'{feature1}*exp({feature2})'] = model_params[feature1] * operation(model_params[feature2])
                    else:
                        model_input[f'{feature1}*{feature2}'] = operation(model_params[feature1], model_params[feature2])

        model_input = pd.DataFrame([model_input])

        return self.model.predict(model_input)

    def check_required_query_params(self, query_params: dict) -> bool:
        """Проверяем параметры запроса на наличие обязательного набора.

        Args:
            query_params (dict): Параметры запроса.

        Returns:
                bool: True — если есть нужные параметры, False — в ином случае.
        """
        if 'flat_id' not in query_params or 'model_params' not in query_params:
            return False

        if not isinstance(query_params['flat_id'], self.param_types['flat_id']):
            return False

        if not isinstance(query_params['model_params'], self.param_types['model_params']):
            return False

        return True


    def check_required_model_params(self, model_params: dict) -> bool:
        """Проверяем параметры для получения предсказаний.

        Args:
            total_model_params (dict): Параметры для получения предсказаний моделью.

        Returns:
            bool: True — если есть нужные параметры, False — иначе
        """
        if set(model_params.keys()) == set(self.required_model_params):
            return True
        return False


    def validate_params(self, params: dict) -> bool:
        """Проверяем корректность параметров запроса и параметров модели.

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
                flat_id = params['flat_id']
                logging.info(f"Predicting for flat_id: {flat_id} and model_params:\n{model_params}")
                # получение предсказания модели
                predicted_apart_cost = self.apart_cost_predict(model_params).tolist()
                response = {
                        "flat_id": flat_id, 
                        "predicted_apart_cost": predicted_apart_cost
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
        "flat_id": '333990123',
        "model_params": {
                    "ceiling_height": 2.5,
                    "building_type_int": 4,
                    "age_of_building": 47,
                    "distance_to_center": 10,
                    "rooms": 2,
                    "floors_total": 12,
                    "living_area": 50,
                    "kitchen_area": 10,
                    "floor": 7,
                    "flats_count": 500
        }
    }

    # создание обработчика запросов для API
    handler = FastApiHandler()

    # осуществление тестового запроса
    response = handler.handle(test_params)
    print(f"Response: {response}")

