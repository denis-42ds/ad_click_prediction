docker container run \
--publish 8081:8081 \
-d \
--volume=./models:/models \
--env-file .env \
apart_cost_pred:v1
