FROM python:3.11-slim

COPY ./app ./apart_cost_app
WORKDIR /apart_cost_app
COPY requirements.txt .

COPY ./models /models

RUN pip3 install -r requirements.txt

EXPOSE ${APP_PORT}

VOLUME /models

CMD uvicorn apart_cost_app:app --reload --port ${APP_PORT} --host 0.0.0.0
