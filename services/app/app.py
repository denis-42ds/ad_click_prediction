import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from constants import ModelParams
from prometheus_client import Counter, Histogram
from app_fastapi_handler import FastApiHandler
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv(dotenv_path="../.env")
PORT = int(os.getenv("APP_PORT"))

app = FastAPI()
app.handler = FastApiHandler()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

positive_counter = Counter('positive_preds', 'positive class prediction counter')

main_app_predictions = Histogram(
    "main_app_predictions",
    "Histogram of predictions",
    buckets=(0.5, 0.63, 0.75, 0.88, 1)
)

@app.post("/api/app/") 
def get_prediction_for_item(ad_id: str, model_params: ModelParams):
    """Функция для получения прогноза клика по рекламному объявлению.

    Args:
        ad_id (str): Идентификатор объявления.
        model_params (ModelParams): Параметры объявления, которые нужно передать в модель.

    Returns:
        dict: Предсказание клика по объявлению с заданными параметрами.
    """
    all_params = {
        "ad_id": ad_id,
        "model_params": model_params.dict()
    }

    prediction = app.handler.click_predict(model_params.dict())
    main_app_predictions.observe(prediction)

    if prediction > 0.5:
        positive_counter.inc()

    return app.handler.handle(all_params)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
