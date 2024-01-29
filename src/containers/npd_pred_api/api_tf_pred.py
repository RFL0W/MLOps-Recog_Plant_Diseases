import pandas as pd
import numpy as np
import asyncio
import time
import os

from fastapi import Depends, FastAPI, Request, Form
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from passlib.context import CryptContext

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input


### ALL INIT ###

#Initialisation de l'application FastAPI
app = FastAPI(
    title="Prediction API for NPD",
    description="Tensorflow based API for Prediction on NPD",
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
model_path = './' + data_Folder_value + '/Models/Current_Model/eNetV2S_NPD_Full.h5'
tmp_img_path = './' + data_Folder_value + '/Temp/'

#Chargement des bases de données nécessaires
npd_main = pd.read_csv(npd_main_path)
npd_mini = pd.read_csv(npd_mini_path)
npd_buffer = pd.read_csv(npd_buffer_path)

#Dictionnaire contenant uniquement l'admin de l'API
users = {
    "admin" : {
        "username" :  "admin",
        "firstname" : "Elliot",
        "lastname": "Alderson",
        "hashed_password" : pwd_context.hash('4dm1N'),
    }
}


### FUNCTIONS ###

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
    username = credentials.username
    if not(users.get(username)) or not(pwd_context.verify(credentials.password, users[username]['hashed_password'])):
        raise MyException(
            code=401,
            name='Authentication Error',
            message='This API is secured. Please provide proper username and password.'
        )
    return credentials.username

#Fonction asynchrone de chargement du modèle
async def load_model_async(model_path):
    loop = asyncio.get_running_loop()
    #Exécution de load_model dans un pool de threads pour libérer le thread principal
    return await loop.run_in_executor(None, load_model, model_path)

#Fonction de chargement de l'image avec preprocessing utilisée par la fonction asynchrone load_image
def load_and_preprocess_image(image_path):
    test_image = image.load_img(image_path, target_size=(224, 224))
    test_image = image.img_to_array(test_image) / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    return preprocess_input(test_image)

#Fonction asynchrone de chargement de l'image avec preprocessing
async def load_image_async(image_path):
    loop = asyncio.get_running_loop()
    # Traitement de l'image dans un pool de threads pour libérer le thread principal
    return await loop.run_in_executor(None, load_and_preprocess_image, image_path)

#Fonction de prédiction
async def predict(image_name):
    """
    Description:
    Cette fonction permet de prédire la classe d'une image en utilisant le modèle TensorFlow. L'image à prédire est stockée dans le fichier temporaire "./$dataFolder/Temp/image_to_predict.jpg". L'image est ensuite chargée, prétraitée et soumise au modèle pour prédire sa classe.

    Args:
    - image_name (str): Le nom de l'image à prédire.

    Returns:
    - dict: Un dictionnaire contenant la prédiction de la classe, la confiance de la prédiction et le temps de prédiction.

    Raises:
    - MyException: Avec un code 500 en cas d'erreur lors de la prédiction de l'image.
    """
    start_time = time.time()

    image_path = tmp_img_path + image_name

    #Chargement du modèle
    classifier = await load_model_async(model_path)

    #Dictionnaire de mapping des classes du modèle
    sickness_class_mapping = {'Apple__Apple_scab': 0,
                              'Apple__Black_rot': 1,
                              'Apple__Cedar_apple_rust': 2,
                              'Apple__healthy': 3,
                              'Blueberry__healthy': 4,
                              'Cherry_(including_sour)__Powdery_mildew': 5,
                              'Cherry_(including_sour)__healthy': 6,
                              'Corn_(maize)__Cercospora_leaf_spot Gray_leaf_spot': 7,
                              'Corn_(maize)__Common_rust_': 8,
                              'Corn_(maize)__Northern_Leaf_Blight': 9,
                              'Corn_(maize)__healthy': 10,
                              'Grape__Black_rot': 11,
                              'Grape__Esca_(Black_Measles)': 12,
                              'Grape__Leaf_blight_(Isariopsis_Leaf_Spot)': 13,
                              'Grape__healthy': 14,
                              'Orange__Haunglongbing_(Citrus_greening)': 15,
                              'Peach__Bacterial_spot': 16,
                              'Peach__healthy': 17,
                              'Pepper,_bell__Bacterial_spot': 18,
                              'Pepper,_bell__healthy': 19,
                              'Potato__Early_blight': 20,
                              'Potato__Late_blight': 21,
                              'Potato__healthy': 22,
                              'Raspberry__healthy': 23,
                              'Soybean__healthy': 24,
                              'Squash__Powdery_mildew': 25,
                              'Strawberry__Leaf_scorch': 26,
                              'Strawberry__healthy': 27,
                              'Tomato__Bacterial_spot': 28,
                              'Tomato__Early_blight': 29,
                              'Tomato__Late_blight': 30,
                              'Tomato__Leaf_Mold': 31,
                              'Tomato__Septoria_leaf_spot': 32,
                              'Tomato__Spider_mites Two-spotted_spider_mite': 33,
                              'Tomato__Target_Spot': 34,
                              'Tomato__Tomato_Yellow_Leaf_Curl_Virus': 35,
                              'Tomato__Tomato_mosaic_virus': 36,
                              'Tomato__healthy': 37}

    #Inversion de la correspondance pour obtenir les noms de classe
    plant_class_names = {v: k for k, v in sickness_class_mapping.items()}

    #Chargement et prétraitement de l'image
    processed_test_img = await load_image_async(image_path)

    #Prédiction de la classe
    prediction_plant = classifier.predict(processed_test_img, verbose=0)
    predicted_plant_class_index = np.argmax(prediction_plant)
    predicted_plant_class = plant_class_names[predicted_plant_class_index]
    confidence_percentage = round(100 * prediction_plant[0][predicted_plant_class_index], 3)
    prediction = predicted_plant_class

    end_time = time.time()

    #Calcul du temps de prédiction
    prediction_duration = end_time - start_time

    return {
        "image_name": str(image_name),
        "image_path": str(image_path),
        "class_label": str(prediction),
        "class_confidence": float(confidence_percentage),
        "prediction_duration": float(prediction_duration)
    }


### ENDPOINTS ###

#Définition de la route "predict"
@app.post("/predict/")
async def predict_img(image_name: str = Form(...), username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet de prédire la classe d'une image en utilisant le modèle TensorFlow. L'image à prédire est stockée dans le fichier temporaire "./$dataFolder/Temp/image_to_predict.jpg". L'image est ensuite chargée, prétraitée et soumise au modèle pour prédire sa classe.

    Args:
    Aucun argument n'est à fournir.

    Returns:
    - JSON contenant la prédiction de la classe, la confiance de la prédiction et le temps de prédiction.

    Raises:
    - MyException: Avec un code 500 en cas d'erreur lors de la prédiction de l'image.
    """
    try:
        #Prédiction de la classe de l'image
        result = await predict(image_name)

        #Renvoi du résultat de la prédiction
        return result
        
    except Exception as e:
        raise MyException(
            code=500,
            name='Image Processing Error',
            message='An error occurred while predicting the image class.'
        )
