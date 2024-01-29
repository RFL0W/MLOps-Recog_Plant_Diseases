docker image build -t rfl0w/npd_core_api -f ./npd_core_api/dockerfile ./npd_core_api
docker image build -t rfl0w/npd_pred_api -f ./npd_pred_api/dockerfile ./npd_pred_api
docker image build -t rfl0w/npd_train_api -f ./npd_train_api/dockerfile ./npd_train_api
docker login
docker push rfl0w/npd_core_api:latest
docker push rfl0w/npd_pred_api:latest
docker push rfl0w/npd_train_api:latest
docker image rm rfl0w/npd_core_api:latest
docker image rm rfl0w/npd_pred_api:latest
docker image rm rfl0w/npd_train_api:latest