docker image build -t npd_core_api -f ./src/containers/npd_core_api/dockerfile ./src/containers/npd_core_api
docker image build -t npd_pred_api -f ./src/containers/npd_pred_api/dockerfile ./src/containers/npd_pred_api
docker image build -t npd_train_api -f ./src/containers/npd_train_api/dockerfile ./src/containers/npd_train_api
docker compose -f ./src/containers/docker-compose.yml -p npd_apis_grp up