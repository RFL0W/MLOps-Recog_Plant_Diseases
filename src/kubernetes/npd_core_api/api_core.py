from fastapi import Depends, FastAPI, Request, File, UploadFile, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from passlib.context import CryptContext
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import pandas as pd
import base64
import datetime
import asyncio
import aiofiles
import aiohttp
import json
import os


### ALL INIT ###

#Initialisation de l'application FastAPI
app = FastAPI(
    title="Recog Plant Diseases",
    description="Core API for Plant Disease Recog.",
    version="1.0.0"
)

#Initialisation de la sécurité HTTP Basic
security = HTTPBasic()

#Déclaration du contexte de hachage pour stocker et vérifier les mots de passe
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

#Classe d'exception personnalisée
class MyException(Exception):
    def __init__(self, code: int, name: str, message: str):
        self.code = code
        self.name = name
        self.message = message

#Gestionnaire d'exception personnalisé pour MyException
@app.exception_handler(MyException)
def MyExceptionHandler(request: Request, exception: MyException):
    return JSONResponse(
        status_code=exception.code,
        content={
            'code': exception.code,
            'url': str(request.url),
            'name': exception.name,
            'message': exception.message
        }
    )

#Récupération de la valeur de la variable d'environnement data_Folder
data_Folder_value = os.environ.get('data_Folder')

#Variables globales
npd_main_path = './' + data_Folder_value + '/BDD/NPD.csv'
npd_mini_path = './' + data_Folder_value + '/BDD/NPD_Mini.csv'
npd_buffer_path = './' + data_Folder_value + '/BDD/NPD_Buffer.csv'
users_init_path = './' + data_Folder_value + '/BDD/Users.csv'
log_path = './' + data_Folder_value + '/Logs/api_log.json'
tmp_models_path = './' + data_Folder_value + '/Models/Temp_Models/'
tmp_img_path = './' + data_Folder_value + '/Temp/'
core_api_url = '127.0.0.1:8080'
pred_api_url = '127.0.0.1:8088'
train_api_url = '127.0.0.1:8888'

#Chargement des bases de données nécessaires
npd_main = pd.read_csv(npd_main_path)
npd_mini = pd.read_csv(npd_mini_path)
users_init = pd.read_csv(users_init_path)

#Initialisation du dictionnaire des utilisateurs
users = {}
for _, row in users_init.iterrows():
    user_data = {
        "username": row["username"],
        "firstname": row["firstname"],
        "lastname": row["lastname"],
        "email": row["email"],
        "hashed_password": pwd_context.hash(row["password"]),
        "access_type": row["access_type"]
    }
    users[row["username"]] = user_data

#Chargement du ConfigMap
config.load_incluster_config()
v1 = client.CoreV1Api()
configmap_name = "npd-configmap"
namespace = "npd-space"

#Récupération des informations de connexion pour accéder aux APIs Tensorflow
tf_admin_pwd = os.environ.get('TF_ADMIN_PASSWORD')
tf_admin_pwd_encoded = base64.b64encode(tf_admin_pwd.encode('utf-8')).decode('utf-8')

#Message de réponse en cas de vérification réussie de l'API
chk_ok_message = {
    "code": 200,
    "name": "API Check OK",
    "message": "This API is working fine."
}

#Vérification de l'existence du fichier de log  au lancement du conteneur et création si absent
if not os.path.exists(log_path):
    #Si le fichier n'existe pas, on le créé avec les accolades dedans
    with open(log_path, 'w') as f:
        json.dump({}, f)

#Nettoyage du répertoire temporaire des fichiers prédits au lancement du conteneur
for file in os.listdir(tmp_img_path):
    file_path = os.path.join(tmp_img_path, file)
    
    #Confirmation que le chemin est un fichier (pas un sous-répertoire)
    if os.path.isfile(file_path):
        #Si OK, suppression du fichier
        os.remove(file_path)


### FUNCTIONS ###

#Fonction de chargement du ConfigMap pour la mise à jour de la liste des utilisateurs
async def update_npd_users_config_map():
    """
    Description:
    Met à jour la liste des utilisateurs dans le ConfigMap.

    La fonction récupère la version actuelle du ConfigMap, ajoute les informations des utilisateurs
    fournies dans le ConfigMap, et effectue le patch du ConfigMap pour mettre à jour la liste des utilisateurs.

    Raises:
    - ApiException: En cas d'erreur lors de la mise à jour du ConfigMap.
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            #Récupération de la version actuelle de la ConfigMap
            current_config_map = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
            current_version = current_config_map.metadata.resource_version

            #Ajout des informations sur les utilisateurs dans le ConfigMap
            users_json = json.dumps(users)

            #Préparation du corps de la requête avec la nouvelle version
            body = {
                "data": {
                    "npd-users": users_json
                },
                "metadata": {
                    "resourceVersion": current_version
                }
            }

            #Patch de la ConfigMap avec gestion de la version
            v1.patch_namespaced_config_map(name=configmap_name, namespace=namespace, body=body)
            print("npd-users list sucessfully updated in ConfigMap !")
            return

        except ApiException as e:
            if e.status == 409 and retry_count < max_retries - 1:
                #En cas de conflit, attendre un court instant avant de réessayer
                print("Conflict detected when updating the ConfigMap. We will try again in few seconds...")
                await asyncio.sleep(1)
                retry_count += 1
            else:
                print(f"Exception encountered when trying to update npd-users list in the ConfigMap: {e}")
                break

#Fonction de chargement du ConfigMap pour la mise à jour du token d'entraînement unique
async def update_train_token_config_map(new_value):
    """
    Description:
    Met à jour le token d'entraînement unique dans le ConfigMap.

    Cette fonction récupère la version actuelle du ConfigMap, remplace la valeur actuelle
    du token d'entraînement par la nouvelle valeur spécifiée, puis patche le ConfigMap
    pour mettre à jour le token d'entraînement.

    Args:
    - new_value (str): La nouvelle valeur du token d'entraînement à définir dans le ConfigMap.

    Raises:
    - ApiException: En cas d'erreur lors de la mise à jour du ConfigMap.
    """
    max_retries = 3
    retry_count = 0

    while retry_count < max_retries:
        try:
            #Récupération de la version actuelle du ConfigMap
            current_config_map = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
            current_version = current_config_map.metadata.resource_version

            #Prépare le corps de la requête avec la nouvelle valeur de train-token
            body = {
                "data": {
                    "train-token": new_value
                },
                "metadata": {
                    "resourceVersion": current_version
                }
            }

            #Patch du ConfigMap avec gestion de la version
            v1.patch_namespaced_config_map(name=configmap_name, namespace=namespace, body=body)
            print("train-token variable sucessfully updated in ConfigMap !")
            return

        except ApiException as e:
            if e.status == 409 and retry_count < max_retries - 1:
                #En cas de conflit, attendre un court instant avant de réessayer
                print("Conflict detected when updating the ConfigMap. We will try again in few seconds...")
                await asyncio.sleep(1)
                retry_count += 1
            else:
                print(f"Exception encountered when trying to update train-token variable in the ConfigMap: {e}")
                break

#Fonction pour obtenir l'utilisateur actuel à partir des informations d'identification HTTP Basic
async def get_current_user(credentials: HTTPBasicCredentials = Depends(security)):
    """
    Description:
    Cette fonction récupère le nom d'utilisateur à partir des informations d'identification HTTP Basic fournies.

    Args:
    - credentials (HTTPBasicCredentials, dépendance): Les informations d'identification HTTP Basic fournies par le client.

    Returns:
    - str: Le nom d'utilisateur extrait des informations d'identification.

    Raises:
    - MyException: Si les informations d'identification sont incorrectes.
    """
    global users
    global users_init

    #Rechargement des informations de la BDD "Users.csv" au cas où il y a eu une mise à jour
    users_init = pd.read_csv(users_init_path)

    #Rechargement du Dataframe users_init dans le dictionnaire users
    users = {}
    for _, row in users_init.iterrows():
        user_data = {
            "username": row["username"],
            "firstname": row["firstname"],
            "lastname": row["lastname"],
            "email": row["email"],
            "hashed_password": pwd_context.hash(row["password"]),
            "access_type": row["access_type"]
        }
        users[row["username"]] = user_data

    #Récupération de la liste des utilisateurs du Config Map
    await update_npd_users_config_map()
    updated_config_map = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
    users_conf_map = json.loads(updated_config_map.data["npd-users"])

    username = credentials.username
    if not(users_conf_map.get(username)) or not(pwd_context.verify(credentials.password, users_conf_map[username]['hashed_password'])):
        raise MyException(
            code=401,
            name='Authentication Error',
            message='This API is secured. Please provide proper username and password.'
        )
    return credentials.username

#Fonction pour vérifier l'état de l'API en vérifiant la présence du fichier "./BDD/NPD.csv"
async def api_check():
    """
    Description:
    Cette fonction vérifie si le fichier "./$data_Folder/BDD/NPD_Buffer.csv" existe.

    Args:
    Aucun argument n'est à fournir.

    Returns:
    - bool: True si le fichier existe, False sinon.

    Raises:
    Aucune exception n'est levée.
    """
    try:
        async with aiofiles.open(npd_main_path, 'r') as file:
            return True
    except FileNotFoundError:
        return False

#Fonction pour ajouter des éléments au fichier de logs
async def add_log(log_path, log_content):
    """
    Description:
    Cette fonction ajoute des données au fichier de logs JSON spécifié. Si le fichier existe, les nouvelles données sont ajoutées aux données existantes, sinon, un nouveau fichier est créé.

    Args:
    - log_path (str): Le chemin du fichier de logs JSON.
    - log_content (dict): Les données de log à ajouter au fichier.

    Returns:
    Aucune valeur de retour.

    Raises:
    Aucune exception n'est levée en cas de succès.
    """
    #Vérification de l'existence du fichier de log JSON
    if os.path.exists(log_path):
        #Si le fichier existe, charger les données existantes
        async with aiofiles.open(log_path, 'r') as json_file:
            data = json.loads(await json_file.read())

        #Ajout des nouvelles données à la structure JSON existante
        data.update(log_content)

        #Sauvegarde du fichier avec les données mises à jour
        async with aiofiles.open(log_path, 'w') as json_file:
            await json_file.write(json.dumps(data, indent=4))
    else:
        #Si le fichier n'existe pas, on le créé et ajoute les données
        async with aiofiles.open(log_path, 'a') as json_file:
            await json_file.write(json.dumps(log_content, indent=4))

#Fonction de nettoyage des "Best Weights" de l'entraînement précédent
async def clean_tmp_models():
    """
    Description:
    Cette fonction nettoie les "Best Weights" du modèle de l'entraînement précédent en supprimant les fichiers de poids temporaires.

    Args:
    Aucun argument n'est requis.

    Returns:
    Aucune valeur de retour.

    Raises:
    Aucune exception n'est levée en cas de succès.
    """
    #Récupération de la liste des fichiers
    for file in os.listdir(tmp_models_path):
            file_path = os.path.join(tmp_models_path, file)
            
            #Confirmation que le chemin est un fichier (pas un sous-répertoire)
            if os.path.isfile(file_path):
                #Si OK, suppression du fichier
                os.remove(file_path)

#Fonction de récupération de logs utilisateurs
async def get_user_logs(username):
    """
    Description:
    Cette fonction permet de récupérer les logs de l'utilisateur connecté en fonction de son nom d'utilisateur.

    Args:
    - username (str): Le nom d'utilisateur de l'utilisateur connecté.

    Returns:
    - list: Une liste contenant les logs de l'utilisateur connecté.

    Raises:
    Aucune exception n'est levée en cas de succès.
    """
    #Ouverture du fichier JSON de logs en mode lecture ('r')
    async with aiofiles.open(log_path, 'r') as json_file:
        content = await json_file.read()
        data = json.loads(content)

    #Initialisation d'une liste pour stocker les données de l'utilisateur connecté
    current_user_logs = []

    #Recherche des éléments liés à l'utilisateur dans le fichier JSON de logs
    for key, value in data.items():
        #Récupération des logs de l'utilisateur connecté uniquement
        if value["username"] == username:
            current_user_logs.append({key: value})

    #Envoi des données de logs de l'utilisateur connecté
    return current_user_logs

#Fonction pour obtenir toutes les logs si l'utilisateur est admin
async def get_all_logs(username):
    """
    Description:
    Cette fonction permet d'obtenir toutes les logs si l'utilisateur est un administrateur. Les logs sont renvoyées sous forme de liste au format JSON.

    Args:
    - username (str): Le nom d'utilisateur de l'utilisateur connecté.

    Returns:
    - list: Une liste contenant toutes les logs si l'utilisateur est un administrateur.

    Raises:
    - MyException: Avec un code 403 si l'utilisateur n'est pas un administrateur.
    """
    #Récupération de la liste des utilisateurs du Config Map
    await update_npd_users_config_map()
    updated_config_map = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
    users_conf_map = json.loads(updated_config_map.data["npd-users"])

    #Vérification du rôle de l'utilisateur
    user = users_conf_map.get(username)
    if user and user["access_type"] == "admin":
        #Ouverture du fichier JSON de logs en mode lecture ('r')
        async with aiofiles.open(log_path, 'r') as json_file:
            content = await json_file.read()
            data = json.loads(content)

        #Initialisation d'une liste pour stocker toutes les logs
        all_logs = []

        #Récupération de toutes les logs
        for key, value in data.items():
            all_logs.append({key: value})

        #Envoi de toutes les logs
        return all_logs
    else:
        raise MyException(
            code=403,
            name='Access Denied',
            message='Only administrators can access all logs.'
        )

#Fonction pour ajouter une nouvelle image dans le Dataset temporaire d'ajout
async def pict_to_buffer(plante: str, maladie: str, image: UploadFile):
    """
    Description:
    Cette fonction permet de téléverser une image de plante malade ou saine dans le répertoire de destination spécifié. Elle gère également la mise à jour du DataFrame NPD_Buffer avec les informations de l'image téléversée.

    Args:
    - plante (str): Le nom de la plante associée à l'image.
    - maladie (str): Le nom de la maladie associée à l'image.
    - image (UploadFile): L'objet représentant l'image téléchargée.

    Returns:
    - dict: Un dictionnaire contenant les informations de l'image téléversée, y compris le nom de l'image et le chemin complet de destination.

    Raises:
    Aucune exception n'est levée en cas de succès.
    """
    #Définition du répertoire de destination pour les images téléchargées
    upload_dir = './' + data_Folder_value + '/Datasets/NPD_Buffer/train/'

    #Chargement du fichier NPD_Buffer
    npd_buffer = pd.read_csv(npd_buffer_path)

    #Si la maladie est "healthy" on affecte 1 à "Saine"
    saine = 1 if maladie.lower() == 'healthy' else 0

    #Généreration d'un nom de fichier unique basé sur le timestamp
    timestamp = datetime.datetime.now().timestamp()
    categorie = plante + "__" + maladie
    folder_categorie = plante + "___" + maladie
    new_filename = f"{categorie}__{timestamp}.jpg"

    #Création du chemin complet de destination pour l'image
    destination_path = os.path.join(upload_dir, folder_categorie, new_filename)

    #Sauvegarde de l'image téléchargée dans le répertoire de destination
    async with aiofiles.open(destination_path, "wb") as image_file:
        content = await image.read()
        await image_file.write(content)

    #Ajout des informations pour le DataFrame NPD_Buffer
    new_data = {
        'Categorie': categorie,
        'Plante': plante,
        'Maladie': maladie,
        'Saine': saine,
        'Set': 'train',
        'DirPath': os.path.join(upload_dir, folder_categorie),
        'FileName': new_filename,
        'FilePath': destination_path,
        'Status': 0
    }

    #Création d'un DataFrame temporaire avec les nouvelles données
    tmp_df = pd.DataFrame([new_data])

    #Concaténation du DataFrame temporaire avec le DataFrame NPD_Buffer
    npd_buffer = pd.concat([npd_buffer, tmp_df], ignore_index=True)

    #Enregistrement du DataFrame mis à jour dans le fichier NPD_Buffer.csv
    npd_buffer.to_csv(npd_buffer_path, index=False)

    return {
        "image_name": str(image.filename),
        "image_path": str(image.file.name)
    }

#Fonction pour ajouter un nouvel utilisateur  de type user dans le Dataset users
async def user_to_dataset(ident: str, prenom: str, nom: str, email: str, motdepasse: str):
    """
    Description:
    Cette route permet d'ajouter un utilisateur de type user au Dataset users. 
    Pas besoin d'être autentifié pour créer un utilisateur de type user.
    
   Args:
    - ident (str): L'identifiant du nouvel utilisateur
    - prenom (str): Le prénom du nouvel utilisateur
    - nom (str): Le nom du nouvel utilisateur
    - email (str): L'adresse email du nouvel utilisateur
    - motdepasse (str) : Le mot de passe du nouvel utilisateur

    Returns:
    - dict: Un dictionnaire contenant les informations sur l'utilisateur créé

    Raises:
    Aucune exception n'est levée en cas de succès.
    """
    #Utilisation des variables globales users & users_init
    global users
    global users_init
    
    #Récupération de la liste des utilisateurs du Config Map
    await update_npd_users_config_map()
    updated_config_map = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
    users_conf_map = json.loads(updated_config_map.data["npd-users"])

    #Vérification que l'identifiant n'est pas déjà utilisé pour un autre utilisateur
    if ident in users_conf_map:
        #Levée d'une exception car l'identifiant est déjà présent dans la base Users
        raise MyException(
            code=405,
            name='New User Creation Issue',
            message='Username already exists. Please choose another username.'
            )
    
    #Ajout des informations de l'utilisateur pour le DataFrame users_init
    new_user = {
        "username": ident,
        "firstname": prenom,
        "lastname": nom,
        "email": email,
        "password": motdepasse,
        "access_type": "user"
    }
    
    #Création d'un DataFrame temporaire avec les nouvelles données
    tmp_df = pd.DataFrame([new_user])
    
    #Concaténation du DataFrame temporaire avec le DataFrame users_init
    users_init = pd.concat([users_init, tmp_df], ignore_index=True)
    
    #Enregistrement du DataFrame mis à jour dans le fichier Users.csv
    users_init.to_csv(users_init_path, index=False)
    
    #Rechargement du Dataframe users_init dans le dictionnaire users
    users = {}
    for _, row in users_init.iterrows():
        user_data = {
            "username": row["username"],
            "firstname": row["firstname"],
            "lastname": row["lastname"],
            "email": row["email"],
            "hashed_password": pwd_context.hash(row["password"]),
            "access_type": row["access_type"]
        }
        users[row["username"]] = user_data

    #Enregistrement de la réponse dans un fichier de log au format JSON
    log_time = datetime.datetime.now()
    curl_url = 'http://{url}/adduser/'.format(url=core_api_url)
    log_train = {
        str(int(log_time.timestamp())): {
            "logtime": str(log_time),
            "username": ident,
            "endpoint": curl_url,
            "endpoint_params": "None",
            "response_code": 200,
            "image_name": "N/A",
            "image_path": "N/A",
            "prediction": "N/A",
            "accuracy": "N/A",
            "pred_duration": "N/A",
            "satis_pred": "N/A",
            "satis_pred_duration": "N/A",
            "train_duration": "N/A"
        }
    }
    await add_log(log_path, log_train)
        
    return {
        "code": 200,
        "name": "New User Added Successfully",
        "message": f"The new user with username {ident} was created successfully."
    }

#Fonction de vérification du statut de l'entraînement
async def check_training_status():
    try:
        # Récupère le ConfigMap pour obtenir la valeur actuelle de train-token
        current_config_map = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
        train_token = current_config_map.data.get("train-token", "False")

        if train_token.lower() == "true":
            raise MyException(
                code=423,
                name='API Busy',
                message='A training is already in progress. Please try again later.'
            )
    except ApiException as e:
        print(f"Exception lors de la vérification du statut de l'entraînement : {e}")


### ENDPOINTS ###

#Route pour vérifier l'état de l'API
@app.get("/apichk")
async def check_connection():
    """
    Description:
    Cette route vérifie l'état de l'API en vérifiant la présence du fichier "./$dataFolder/BDD/NPD.csv". Si le fichier est présent, l'API renvoie un message indiquant que tout fonctionne correctement. Sinon, une exception est levée pour signaler un problème.

    Args:
    Aucun argument n'est à fournir.

    Returns:
    - JSON contenant un message indiquant que l'API fonctionne correctement.

    Raises:
    - MyException: Avec un code 503 si le fichier "./$data_Folder/BDD/NPD_Buffer.csv" est introuvable, indiquant que l'API ne fonctionne pas correctement.
    """
    if await api_check():
        return JSONResponse(content=chk_ok_message)
    else:
        raise MyException(
            code=503,
            name='API Issued',
            message='This API is not working fine. Please contact your administrators.'
        )

#Route pour vérifier l'état sécurisé de l'API
@app.get("/secapichk")
async def check_connection_secured(username: str = Depends(get_current_user)):
    """
    Description:
    Cette route vérifie l'état sécurisé de l'API en vérifiant la présence du fichier "./$dataFolder/BDD/NPD_Buffer.csv" et l'authentification de l'utilisateur. Si le fichier est présent et que l'utilisateur est authentifié, l'API renvoie un message indiquant que tout fonctionne correctement et que l'API est sécurisée. Si l'utilisateur n'est pas authentifié, une exception est levée pour signaler un problème d'authentification. Sinon, une exception est levée pour signaler un problème général avec l'API.

    Args:
    - username (str, dépendance): Le nom d'utilisateur récupéré à partir de la dépendance `get_current_user`.

    Returns:
    - JSON contenant un message indiquant que l'API fonctionne correctement et est sécurisée.

    Raises:
    - MyException: Avec un code 401 si l'utilisateur n'est pas authentifié, indiquant que l'API est sécurisée et que des informations d'identification valides sont requises.
    - MyException: Avec un code 503 si le fichier "./$data_Folder/BDD/NPD_Buffer.csv" est introuvable, indiquant que l'API ne fonctionne pas correctement.
    """
    if await api_check() and username:
        return JSONResponse(content=chk_ok_message)
    elif await api_check() and not username:
        raise MyException(
            code=401,
            name='Authentication Error',
            message='This API is secured. Please provide proper username and password.'
        )
    else:
        raise MyException(
            code=503,
            name='API Issued',
            message='This API is not working fine. Please contact your administrators.'
        )

#Route pour lancer une prédiction
@app.post("/predict/")
async def predict_img(image: UploadFile = File(...), username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet de prédire la classe d'une image en utilisant le modèle TensorFlow. L'utilisateur doit fournir une image au format JPEG. L'image est sauvegardée sur le serveur, puis une requête est envoyée au modèle TensorFlow pour effectuer la prédiction.

    Args:
    - image (UploadFile): L'image à prédire, au format JPEG.
    - username (str, dépendance): Le nom d'utilisateur récupéré à partir de la dépendance `get_current_user`.

    Returns:
    - JSON contenant la prédiction de la classe, la confiance de la prédiction et le temps de prédiction.

    Raises:
    - MyException: Avec un code 400 si l'utilisateur n'a pas fourni d'image.
    - MyException: Avec un code 400 si l'extension de l'image n'est pas autorisée (seules les images JPEG sont autorisées).
    - MyException: Avec un code 500 en cas d'erreur lors de la prédiction de l'image.
    """
    #Vérification qu'un fichier image a bien été fourni
    if not image:
        raise MyException(
            code=400,
            name="Bad Request",
            message="No given image as argument."
        )

    #Vérification de l'extension du fichier qui doit être JPG
    if image.filename.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        try:
            #Création du nom de l'image
            timestamp = datetime.datetime.now().timestamp()
            image_name = f"image_to_predict-{timestamp}.jpg"
            image_path = tmp_img_path + image_name

            #Enregistrement de l'image téléchargée sur le serveur
            async with aiofiles.open(image_path, "wb") as img_file:
                await img_file.write(await image.read())

            #Exécution de la requête curl pour prédire l'image
            async with aiohttp.ClientSession() as session:
                curl_url = 'http://{url}/predict/'.format(url=pred_api_url)
                headers = {'Authorization': f'Basic {tf_admin_pwd_encoded}'}
                data = aiohttp.FormData()
                data.add_field('image_name', image_name)
                async with session.post(curl_url, headers=headers, data=data) as response:
                    prediction_result = await response.json()

                    #Enregistrement de la réponse dans un fichier de log au format JSON
                    log_time = datetime.datetime.now()
                    log_predict = {
                        str(int(log_time.timestamp())): {
                            "logtime": str(log_time),
                            "username": username,
                            "endpoint": curl_url,
                            "endpoint_params": "None",
                            "response_code": int(response.status),
                            "image_name": prediction_result["image_name"],
                            "image_path": prediction_result["image_path"],
                            "prediction": prediction_result["class_label"],
                            "accuracy": prediction_result["class_confidence"],
                            "pred_duration": prediction_result["prediction_duration"],
                            "satis_pred": "Unknown",
                            "satis_pred_duration": "Unknown",
                            "train_duration": "N/A"
                        }
                    }

                    await add_log(log_path, log_predict)

                    return prediction_result

        except Exception as e:
            raise MyException(
                code=500,
                name='Image Prediction Error',
                message='An error occurred while predicting the image class.'
            )
    else:
        raise MyException(
            code=400,
            name='Invalid Image Format',
            message='Only JPG images are allowed: .jpg .jpeg .JPG .JPEG !'
        )

#Route pour lancer un entraînement
@app.post("/train/{training_type}/")
async def launch_train(training_type: str, username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet de lancer l'entraînement d'un modèle TensorFlow. L'utilisateur doit spécifier le type d'entraînement en fournissant "full" pour un entraînement complet ou "mini" pour un entraînement miniaturisé. Le répertoire de sauvegarde des "Best Weights" est nettoyé avant de lancer l'entraînement. Une requête est ensuite envoyée au modèle TensorFlow pour démarrer l'entraînement.

    Args:
    - training_type (str): Le type d'entraînement ("full" ou "mini").
    - username (str, dépendance): Le nom d'utilisateur récupéré à partir de la dépendance `get_current_user`.

    Returns:
    - JSON contenant un message indiquant que l'entraînement a été lancé avec succès.

    Raises:
    - MyException: Avec un code 400 si le type d'entraînement n'est pas valide.
    - MyException: Avec un code 401 si l'utilisateur n'est pas authentifié.
    - MyException: Avec un code 403 si l'utilisateur n'est pas un administrateur.
    """
    #Récupération de la liste des utilisateurs du Config Map
    await update_npd_users_config_map()
    updated_config_map = v1.read_namespaced_config_map(name=configmap_name, namespace=namespace)
    users_conf_map = json.loads(updated_config_map.data["npd-users"])

    #Vérification du type d'accès de l'utilisateur
    user = users_conf_map.get(username)
    if user['access_type'] != "admin":
        raise MyException(
            code=403,
            name='Access Denied',
            message='Only administrators are allowed to launch a new Model Training.'
        )
    
    #Vérification du statut de l'entraînement pour envoi d'un message s'il est déjà en cours
    await check_training_status()

    async with aiohttp.ClientSession() as session:
        #Vérification du type d'entraînement voulu
        if training_type == "full":
            #Mise à True de la variable de suivi pour indiquer que l'entraînement est en cours
            await update_train_token_config_map("True")
            #Nettoyage du répertoire de sauvegarde des "Best Weights" pendant l'entraînement
            await clean_tmp_models()
            #Exécution de la requête curl pour lancer l'entraînement complet
            curl_url = 'http://{url}/train/full/'.format(url=train_api_url)
            headers = {'Authorization': f'Basic {tf_admin_pwd_encoded}'}
            async with session.get(curl_url, headers=headers) as response:
                result = await response.json()

                #Enregistrement de la réponse dans un fichier de log au format JSON
                log_time = datetime.datetime.now()
                log_train = {
                    str(int(log_time.timestamp())): {
                        "logtime": str(log_time),
                        "username": username,
                        "endpoint": curl_url,
                        "endpoint_params": training_type,
                        "response_code": int(response.status),
                        "image_name": "N/A",
                        "image_path": "N/A",
                        "prediction": "N/A",
                        "accuracy": "N/A",
                        "pred_duration": "N/A",
                        "satis_pred": "N/A",
                        "satis_pred_duration": "N/A",
                        "train_duration": result["duration"]
                    }
                }

                await add_log(log_path, log_train)

                #Mise à False de la variable de suivi pour indiquer que l'entraînement est terminé
                await update_train_token_config_map("False")

        elif training_type == "mini":
            #Mise à True de la variable de suivi pour indiquer que l'entraînement est en cours
            await update_train_token_config_map("True")
            #Nettoyage du répertoire de sauvegarde des "Best Weights" pendant l'entraînement
            await clean_tmp_models()
            #Exécution de la requête curl pour lancer l'entraînement miniaturisé
            curl_url = 'http://{url}/train/mini/'.format(url=train_api_url)
            headers = {'Authorization': f'Basic {tf_admin_pwd_encoded}'}
            async with session.get(curl_url, headers=headers) as response:
                result = await response.json()

                #Enregistrement de la réponse dans un fichier de log au format JSON
                log_time = datetime.datetime.now()
                log_train = {
                    str(int(log_time.timestamp())): {
                        "logtime": str(log_time),
                        "username": username,
                        "endpoint": curl_url,
                        "endpoint_params": training_type,
                        "response_code": int(response.status),
                        "image_name": "N/A",
                        "image_path": "N/A",
                        "prediction": "N/A",
                        "accuracy": "N/A",
                        "pred_duration": "N/A",
                        "satis_pred": "N/A",
                        "satis_pred_duration": "N/A",
                        "train_duration": result["duration"]
                    }
                }

                await add_log(log_path, log_train)

                #Mise à False de la variable de suivi pour indiquer que l'entraînement est terminé
                await update_train_token_config_map("False")

        else:
            raise MyException(
                code=400,
                name='Invalid Training Type Given',
                message='Please choose "full" or "mini" as training type.'
            )

        return result

#Route pour récupérer les logs utilisateur
@app.get("/gimmelogs")
async def send_user_logs(username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet à un utilisateur de récupérer ses propres logs. Les logs sont stockés au format JSON. Les logs de l'utilisateur actuel sont renvoyés au client.

    Args:
    - username (str, dépendance): Le nom d'utilisateur récupéré à partir de la dépendance `get_current_user`.

    Returns:
    - JSON contenant les logs de l'utilisateur connecté.

    Raises:
    - MyException: Avec un code 401 si l'utilisateur n'est pas authentifié.
    - MyException: Avec un code 503 en cas de problème lors de la récupération des logs.
    """
    if username:
        response = await get_user_logs(username)
        return json.dumps(response)
    elif not username:
        raise MyException(
            code=401,
            name='Authentication Error',
            message='This API is secured. Please provide proper username and password.'
        )
    else:
        raise MyException(
            code=503,
            name='Logs Retrieval Issue',
            message='There was a problem trying to retrieve your logs. Please contact your administrators.'
        )
    
#Route pour récupérer toutes les logs si l'utilisateur est admin
@app.get("/getalllogs")
async def send_all_logs(username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet à un administrateur de récupérer toutes les logs. Les logs sont stockés au format JSON. Toutes les logs sont renvoyées au client.

    Args:
    - username (str, dépendance): Le nom d'utilisateur récupéré à partir de la dépendance `get_current_user`.

    Returns:
    - JSON contenant toutes les logs si l'utilisateur est un administrateur.

    Raises:
    - MyException: Avec un code 403 si l'utilisateur n'est pas un administrateur.
    """  
    if username:
        response = await get_all_logs(username)
        return json.dumps(response)
    elif not username:
        raise MyException(
            code=401,
            name='Authentication Error',
            message='This API is secured. Please provide proper username and password.'
        )
    else:
        raise MyException(
            code=503,
            name='Logs Extraction Issue',
            message='There was a problem when extracting all logs.'
        )

#Fonction pour traiter l'envoi d'une image avec les informations fournies
@app.post("/addpict/")
async def add_new_image(plante: str = Form(...), maladie: str = Form(...), image: UploadFile = File(...), username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet d'ajouter des images au Dataset tampon avec des informations telles que le nom de la plante et le nom de la maladie associés à l'image.

    Args:
    - plante (str): Le nom de la plante associée à l'image.
    - maladie (str): Le nom de la maladie associée à l'image.
    - image (UploadFile): Le fichier image à téléverser.
    - username (str, dépendance): Le nom d'utilisateur de l'expéditeur.

    Returns:
    - dict: Un dictionnaire de réponse indiquant le succès de l'opération.

    Raises:
    - MyException: Avec un code 503 en cas de problème lors de l'ajout de la nouvelle image.
    """
    #Vérification qu'un fichier image a bien été fourni
    if not image:
        raise MyException(
            code=400,
            name="Bad Request",
            message="No given image as argument."
        )

    #Vérification de l'extension du fichier qui doit être JPG
    if image.filename.endswith(('.jpg', '.jpeg', '.JPG', '.JPEG')):
        try:
            #Lancement de la fonction pour ajouter une image dans le Dataset tampon
            response = await pict_to_buffer(plante, maladie, image)

            #Enregistrement de la réponse dans un fichier de log au format JSON
            log_time = datetime.datetime.now()
            curl_url = 'http://{url}/addpict/'.format(url=core_api_url)
            log_train = {
                str(int(log_time.timestamp())): {
                    "logtime": str(log_time),
                    "username": username,
                    "endpoint": curl_url,
                    "endpoint_params": "None",
                    "response_code": 200,
                    "image_name": response["image_name"],
                    "image_path": response["image_path"],
                    "prediction": "N/A",
                    "accuracy": "N/A",
                    "pred_duration": "N/A",
                    "satis_pred": "N/A",
                    "satis_pred_duration": "N/A",
                    "train_duration": "N/A"
                }
            }

            await add_log(log_path, log_train)

            return {
                "code": 200,
                "name": "New Image Added Successfully",
                "message": "The given picture was recorded and will be added to scope after verification."
                }

        except Exception as e:
            raise MyException(
                code=503,
                name='New Image Uploading Issue',
                message='There was a problem trying to add the new picture. Please contact your administrators.'
            )
    else:
        raise MyException(
            code=400,
            name='Invalid Image Format',
            message='Only JPG images are allowed: .jpg .jpeg .JPG .JPEG !'
        )

#Fonction pour traiter l'ajout d'un utilisateur dans la base utilisateurs avec les informations fournies
@app.post("/adduser/")
async def add_new_user(ident: str = Form(...), prenom: str = Form(...), nom: str = Form(...), email: str = Form(...), motdepasse: str = Form(...)):
    """
    Description:
    Cette route permet d'ajouter un utilisateur au Dataset users de type user avec des informations telles que l'identifiant du nouvel utilisateur, ainsi que son prénom, nom, adresse email et mot de passe.

    Args:
    - ident (str): L'identifiant du nouvel utilisateur
    - prenom (str): Le prénom du nouvel utilisateur
    - nom (str): Le nom du nouvel utilisateur
    - email (str): L'adresse email du nouvel utilisateur
    - motdepasse (str) : Le mot de passe du nouvel utilisateur

    Returns:
    - dict: Un dictionnaire de réponse indiquant le succès de l'opération.

    Raises:
    - MyException: Avec un code 503 en cas de problème lors de l'ajout du nouvel utilisateur.
    """
    global users
    global users_init

    #Rechargement des informations de la BDD "Users.csv" au cas où il y a eu une mise à jour
    users_init = pd.read_csv(users_init_path)

    #Rechargement du Dataframe users_init dans le dictionnaire users
    users = {}
    for _, row in users_init.iterrows():
        user_data = {
            "username": row["username"],
            "firstname": row["firstname"],
            "lastname": row["lastname"],
            "email": row["email"],
            "hashed_password": pwd_context.hash(row["password"]),
            "access_type": row["access_type"]
        }
        users[row["username"]] = user_data

    #Lancement de la fonction pour ajouter un utilisateur dans le Dataset tampon
    response = await user_to_dataset(ident, prenom, nom, email, motdepasse)
    await update_npd_users_config_map()

    return response

@app.post("/opinion/")
async def add_opinion(image: str = Form(...), opinionPrediction: str = Form(...), opinionDuree: str = Form(...), username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet d'ajouter dans le fichier de log l'opinion d'un utilisateur suite à sa demande de prédiction avec des informations telles que l'identifiant de l'utilisateur, ainsi que le nom de l'image prédite, son sentiment sur la prédiction ainsi que son opinion sur le temps de prédiction.

    Args:
    - username (str, dépendance): Le nom d'utilisateur récupéré à partir de la dépendance `get_current_user`.ident (str): L'identifiant du nouvel utilisateur
    - image (str): Le nom de l'image prédite
    - opinionPrediction (str): Le sentiment de l'utilisateur sur la prédiction de l'image
    - opinionDuree (str): Le sentiment de l'utilisateur sur la durée du temps de prédiction de l'image
    
    Returns:
    - dict: Un dictionnaire de réponse indiquant le succès de l'opération.

    Raises:
    - MyException: Avec un code 503 en cas de problème lors de l'ajout de la log du sentiment.
    """
    try:
        #Enregistrement du sentiment dans un fichier de log au format JSON
        log_time = datetime.datetime.now()
        curl_url = 'http://{url}/opinion/'.format(url=core_api_url)
        log_train = {
            str(int(log_time.timestamp())): {
                "logtime": str(log_time),
                "username": username,
                "endpoint": curl_url,
                "endpoint_params": "None",
                "response_code": 200,
                "image_name": image,
                "image_path": "N/A",
                "prediction": "N/A",
                "accuracy": "N/A",
                "pred_duration": "N/A",
                "satis_pred": opinionPrediction,
                "satis_pred_duration": opinionDuree,
                "train_duration": "N/A"
            }
        }

        await add_log(log_path, log_train)

        return {
            "code": 200,
            "name": "New Opinion Added Successfully",
            "message": "The given opinion was recorded. Thanks."
            }

    except Exception as e:
        raise MyException(
            code=503,
            name='New Opinion Adding Issue',
            message='There was a problem trying to add the opinion. Please contact your administrators.'
        )
