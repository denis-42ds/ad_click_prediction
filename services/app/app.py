import os
import uvicorn
from fastapi import FastAPI
from dotenv import load_dotenv
from prometheus_client import Counter, Histogram
from constants import MIN_APART_COST, ModelParams
from apart_cost_fastapi_handler import FastApiHandler
from prometheus_fastapi_instrumentator import Instrumentator

load_dotenv(dotenv_path="../.env")
PORT = int(os.getenv("APP_PORT"))

app = FastAPI()
app.handler = FastApiHandler()

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

c = Counter('low_preds', 'forecast counter below 11,000,000')

main_app_predictions = Histogram(
    "main_app_predictions",
    "Histogram of predictions",
    buckets=(1e7, 1.5e7, 2e7, 2.5e7, 3e7)
)

@app.post("/api/cost/") 
def get_prediction_for_item(flat_id: str, model_params: ModelParams):
    """Функция для получения прогноза стоимости квартиры.

    Args:
        flat_id (str): Идентификатор квартиры.
        model_params (ModelParams): Параметры квартиры, которые нужно передать в модель.

    Returns:
        dict: Предсказание стоимости квартиры с заданными параметрами.
    """
    all_params = {
        "flat_id": flat_id,
        "model_params": model_params.dict()
    }

    prediction = app.handler.apart_cost_predict(model_params.dict())
    main_app_predictions.observe(prediction[0])

    if prediction[0] < MIN_APART_COST:
        c.inc()

    return app.handler.handle(all_params)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
