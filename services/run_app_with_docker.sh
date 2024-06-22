docker container run \
--publish 8081:8081 \
-d \
--volume=./models:/models \
--env-file .env \
app_click_pred:v1
