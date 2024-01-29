# Création des répertoires et droits spécifiques à Airflow
mkdir -p ./src/airflow/dags ./src/airflow/logs ./src/airflow/plugins
echo -e "AIRFLOW_UID=$(id -u)\nAIRFLOW_GID=0" > ./src/airflow/.env

# Lancement d'airflow en utilisant le répertoire de travail ./src/airflow
docker compose -f ./src/airflow/docker-compose.yaml --project-directory ./src/airflow up airflow-init
docker compose -f ./src/airflow/docker-compose.yaml --project-directory ./src/airflow up -d

# Copie du fichier dag dans le nouveau répertoire dag
cp ./src/airflow/dag_new_picture_email.py ./src/airflow/dags

# Création de la variable Airflow path_fic
docker exec -it airflow-airflow-worker-1 airflow variables set path_fic /app/data/BDD/ 
# Création de la connection Airflow connection_data_BDD
docker exec -it airflow-airflow-worker-1 airflow connections add connection_data_BDD --conn-type "fs" --conn-description "Répertoire où se trouve les bases des nouvelles images et utilisateurs" --conn-extra '{"path":"/app/data/BDD/"}'
