from fastapi.testclient import TestClient
from api_core import app
from pathlib import Path
import httpx
import os
import pytest
from aioresponses import aioresponses

# Récupérer la valeur de la variable d'environnement data_Folder
valeur_data_Folder = os.environ.get('data_Folder')

##########
# TESTS DE VERIFICATION DE PRESENCE DES FICHIERS
##########

# test de vérification de la présence du fichier NPD.csv
def test_file_exists_NPD():
    thePath = './' + valeur_data_Folder + '/BDD/NPD.csv'
    file_path = Path(thePath)
    assert file_path.exists(), f"Le fichier {file_path} n'existe pas."

# test de vérification de la présence du fichier NPD_Buffer.csv
def test_file_exists_NPD_Buffer():
    thePath = './' + valeur_data_Folder + '/BDD/NPD_Buffer.csv'
    file_path = Path(thePath)
    assert file_path.exists(), f"Le fichier {file_path} n'existe pas."
    
# test de vérification de la présence du fichier Users.csv
def test_file_exists_Users():
    thePath = './' + valeur_data_Folder + '/BDD/Users.csv'
    file_path = Path(thePath)
    assert file_path.exists(), f"Le fichier {file_path} n'existe pas."

# test de vérification de la présence de l'image de test pour la prédiction PotatoHealthy2.jpeg
def test_file_exists_PotatoHealthy2():
    thePath = './pict_test/PotatoHealthy2.jpeg'
    file_path = Path(thePath)
    assert file_path.exists(), f"Le fichier {file_path} n'existe pas."
     
     
##########
# TESTS DE VERIFICATION DE FONCTIONNEMENT DES ROUTES DE L'API
##########
client = TestClient(app)
   
# test du fonctionnement de l'API sans authentification : route /apichk
def test_check_connection():
    response = client.get("/apichk")
    assert response.status_code == 200, f"Status code à {response.status_code} au lieu de 200 avec le message {response.json()} pour la route /apichk"
    assert response.json() == {
                                "code": 200,
                                "name": "API Check OK",
                                "message": "This API is working fine."
                            }, f"réponse renvoyée {response.json()} pour la route /apichk"
    
    
# tests du fonctionnement de l'API avec authentification : route /secapichk    
def test_check_connection_authent():
    # test avec authentification ok
    headers = {"Authorization": "Basic dGVzdF91c2VyOlB5dGhvblVzclRlc3QwMQ=="}
    response = client.get("/secapichk", headers=headers)
    response_json = response.json()
    assert response.status_code == 200, f"Status code à {response.status_code} au lieu de 200 avec le message {response.json()} pour la route /secapichk "
    assert response_json["name"] == "API Check OK",  f"name à {response_json['name']} au lieu de 'API Check OK' pour la route /secapichk "
    assert response_json["message"] == "This API is working fine.",  f"message à {response_json['message']} au lieu de 'This API is working fine.' pour la route /secapichk "

    # test avec erreur sur authentification
    headers = {"Authorization": "Basic dGVzdF91c2VyOlB5dGhvblVzclRlc3Qw"}
    response = client.get("/secapichk", headers=headers)
    response_json = response.json()
    assert response.status_code == 401, f"Status code à {response.status_code} au lieu de 401 avec le message {response.json()} pour la route /secapichk "
    assert response_json["name"] == "Authentication Error",  f"name à {response_json['name']} au lieu de 'Authentication Error' pour la route /secapichk "
    assert response_json["message"] == "This API is secured. Please provide proper username and password.",  f"message à {response_json['message']} au lieu de 'This API is secured. Please provide proper username and password.' pour la route /secapichk "
    
    #test sans authentification
    response = client.get("/secapichk")
    response_json = response.json()
    assert response_json["detail"] == "Not authenticated", f"detail à { response_json['detail']} au lieu de 'Not authenticated' pour la route /secapichk"
    

# test du fonctionnement de la prédicion : route /predict
# l'appel de la deuxième api npd_pred_api sera simulé par pytest-mock
async def test_predict_img():
     # Simulation de la réponse de l'API api_tf_pred
    expected_result = {
        "image_name": "PotatoHealthy2.jpeg",
        "image_path": "./pict_test/PotatoHealthy2.jpeg",
        "class_label": "Potato__healthy",
        "class_confidence": 99.999,
        "prediction_duration": 17.054523944854736
    }
    
    with aioresponses() as mock_responses:
        mock_responses.post('http://npd_pred_api:8088/predict/', status=200, payload=expected_result)

        # Appel de la fonction à tester
        with open("./pict_test/PotatoHealthy2.jpeg", "rb") as image_file:
            files = {"image": image_file}
            headers = {"Authorization": "Basic dGVzdF91c2VyOlB5dGhvblVzclRlc3QwMQ=="}
            response = client.post("/predict/", headers=headers, files=files)
            
        # Vérifier le comportement de la fonction
        assert response.status_code == 200
        assert response.json() == expected_result
    

## Tests de récupération de la log d'un utilisateur : route /gimmelogs
def test_send_user_logs():
    # test avec authentification ok
    headers = {"Authorization": "Basic dGVzdF91c2VyOlB5dGhvblVzclRlc3QwMQ=="}
    response = client.get("/gimmelogs", headers=headers)
    assert response.status_code == 200, f"Status code à {response.status_code} au lieu de 200 pour la route /gimmelogs avec le message {response.json()}"
    
    # test avec erreur sur authentification
    headers = {"Authorization": "Basic dGVzdF91c2VyOlB5dGhvblVzclRlc3Qw"}
    response = client.get("/gimmelogs", headers=headers)
    assert response.status_code == 401, f"Status code à {response.status_code} au lieu de 401 pour la route /gimmelogs avec le message {response.json()}"
    response_json = response.json()
    assert response_json["name"] == "Authentication Error",  f"name à {response_json['name']} au lieu de 'Authentication Error' pour la route /gimmelogs "
    assert response_json["message"] == "This API is secured. Please provide proper username and password.",  f"message à {response_json['message']} au lieu de 'This API is secured. Please provide proper username and password.' pour la route /gimmelogs "

    # test sans authentification
    response = client.get("/gimmelogs")
    response_json = response.json()
    assert response_json["detail"] == "Not authenticated", f"detail à { response_json['detail']} au lieu de 'Not authenticated' pour la route /gimmelogs "


## Tests de récupération de la log de tous les utilisateurs (uniquement user admin) : route /getalllogs
def test_send_all_logs():
    # test avec authentification ok pour un user admin
    headers = {"Authorization": "Basic dGVzdF9hZG1pbjpQeXRob25BZG1UZXN0MDE="}
    response = client.get("/getalllogs", headers=headers)
    assert response.status_code == 200, f"Status code à {response.status_code} au lieu de 200 pour la route /getalllogs avec le message {response.json()}"
    
    # test avec authentification ok pour un user non admin
    headers = {"Authorization": "Basic dGVzdF91c2VyOlB5dGhvblVzclRlc3QwMQ=="}
    response = client.get("/getalllogs", headers=headers)
    assert response.status_code == 403, f"Status code à {response.status_code} au lieu de 403 pour la route /getalllogs avec le message {response.json()}"
    response_json = response.json()
    assert response_json["name"] == "Access Denied",  f"name à {response_json['name']} au lieu de 'Access Denied' pour la route /getalllogs "
    assert response_json["message"] == "Only administrators can access all logs.",  f"message à {response_json['message']} au lieu de 'Only administrators can access all logs.' pour la route /getalllogs "

    # test avec erreur sur authentification
    headers = {"Authorization": "Basic dGVzdF91c2VyOlB5dGhvblVzclRlc3Qw"}
    response = client.get("/getalllogs", headers=headers)
    assert response.status_code == 401, f"Status code à {response.status_code} au lieu de 401 pour la route /getalllogs avec le message {response.json()}"
    response_json = response.json()
    assert response_json["name"] == "Authentication Error",  f"name à {response_json['name']} au lieu de 'Authentication Error' pour la route /getalllogs "
    assert response_json["message"] == "This API is secured. Please provide proper username and password.",  f"message à {response_json['message']} au lieu de 'This API is secured. Please provide proper username and password.' pour la route /getalllogs "

    # test sans authentification
    response = client.get("/getalllogs")
    response_json = response.json()
    assert response_json["detail"] == "Not authenticated", f"detail à { response_json['detail']} au lieu de 'Not authenticated' pour la route /getalllogs "    