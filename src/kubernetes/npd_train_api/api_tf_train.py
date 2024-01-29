import pandas as pd
import datetime
from datetime import date
from timeit import default_timer as timer
import time
import aiofiles
import os

from fastapi import Depends, FastAPI, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import JSONResponse
from passlib.context import CryptContext
from kubernetes import client, config
from kubernetes.client.rest import ApiException

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


### ALL INIT ###

#Initialisation de l'application FastAPI
app = FastAPI(
    title="Training API for NPD",
    description="Tensorflow based API for Training on NPD",
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

#Chargement du ConfigMap
config.load_incluster_config()
v1 = client.CoreV1Api()
configmap_name = "npd-configmap"
namespace = "npd-space"

#Message de réponse en cas de vérification réussie de l'API
chk_ok_message = {
    "code": 200,
    "name": "API Check OK",
    "message": "This API is working fine."
}


### FUNCTIONS ###

#Fonction de chargement du ConfigMap pour la mise à jour du token d'entraînement unique (non-asynchrone)
def update_train_token_config_map(new_value):
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
                time.sleep(1)
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
    username = credentials.username
    if not(users.get(username)) or not(pwd_context.verify(credentials.password, users[username]['hashed_password'])):
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

#Fonction d'entraînement
async def new_train(training_type):
    """
    Description:
    Cette fonction permet de lancer l'entraînement d'un modèle TensorFlow. L'utilisateur doit spécifier le type d'entraînement en fournissant "full" pour un entraînement complet ou "mini" pour un entraînement miniaturisé. Un nouveau modèle sera entraîné en utilisant les données appropriées et les poids seront sauvegardés. Le chemin du modèle entraîné sera renvoyé.

    Args:
    - training_type (str): Le type d'entraînement ("full" ou "mini").

    Returns:
    - dict: Un dictionnaire contenant un message indiquant que l'entraînement a été lancé avec succès et le chemin du modèle entraîné.

    Raises:
    - MyException: Avec un code 500 en cas d'erreur lors de l'entraînement du modèle.
    """
    #Mise à True de la variable de suivi pour indiquer que l'entraînement est en cours
    update_train_token_config_map("True")

    #Récupération de l'heure de début d'entraînement
    start_time = time.time()

    ###DATAFRAME SPLIT
    print("INFO:     Currently charging Dataframe")

    #Récupération des données des set train et valid dans 2 nouveaux dataframes
    if training_type == "full":
        train_data_sickness = npd_main[npd_main["Set"] == "train"]
        valid_data_sickness = npd_main[npd_main["Set"] == "valid"]
        batch_size = 64

    elif training_type == "mini":
        train_data_sickness = npd_mini[npd_mini["Set"] == "train"]
        valid_data_sickness = npd_mini[npd_mini["Set"] == "valid"]
        batch_size = 32

    ###DATAGEN
    print("INFO:     Currently generating Datasets")

    #Création du générateur d'image pour Train avec beaucoup de nouvelles modifications bien que les images aient été
    #   déjà augmentées. Cela afin d'éviter un sur-apprentissage sur des images trop similaires.
    train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rescale=1./255,
                                    shear_range=0.2, brightness_range=(0.4,1.7),
                                    zoom_range=0.5, rotation_range=40,
                                    width_shift_range=0.3, height_shift_range=0.3,
                                    horizontal_flip=True, vertical_flip=True,
                                    fill_mode='nearest')

    #Création du générateur d'image pour Valid avec seulement des modifications cohérentes avec ce qu'on pourrait
    #   trouver dans la nature.
    valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                    rescale=1./255, 
                                    brightness_range=(0.6,1.2),
                                    zoom_range=0.5, rotation_range=40,
                                    horizontal_flip=True, vertical_flip=True)

    #Création du dataset d'entraînement
    training_set_sickness = train_datagen.flow_from_dataframe(
        dataframe=train_data_sickness,
        directory=None,   #None car nous utilisons "FilePath" du dataframe pour charger les images
        x_col='FilePath',   #Colonne contenant les chemins des fichiers
        y_col='Categorie',   #Colonne contenant les étiquettes
        target_size=(224, 224),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=True,
        seed=111
    )

    #Création du dataset de validation
    valid_set_sickness = valid_datagen.flow_from_dataframe(
        dataframe=valid_data_sickness,
        directory=None,   #None car nous utilisons "FilePath" du dataframe pour charger les images
        x_col='FilePath',   #Colonne contenant les chemins des fichiers
        y_col='Categorie',   #Colonne contenant les étiquettes
        target_size=(224, 224),
        class_mode="categorical",
        batch_size=batch_size,
        shuffle=False
    )

    ###MODEL
    print("INFO:     Currently creating Model")

    #Chargement du modèle EfficientNetV2S pré-entraîné
    eNetV2S_model = EfficientNetV2S(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=(224, 224, 3),
            pooling=None,
            classes=2,
            classifier_activation="softmax",
            include_preprocessing=False
            )

    #Freeze des poids des couches de base du modèle pré-entraîné
    for layer in eNetV2S_model.layers:
        layer.trainable = False

    #Unfreeze des poids des 48 dernières couches de base du modèle pré-entraîné (mais pas les BatchNormalization)
    #Si le temps le permet, tester aussi un Unfreeze des 63 dernières couches.
    for layer in eNetV2S_model.layers[-48:]:
            if not isinstance(layer, layers.BatchNormalization):
                layer.trainable = True

    #Création du modèle basé sur EfficientNetV2S
    classifier = Sequential()
    classifier.add(eNetV2S_model)
    classifier.add(GlobalAveragePooling2D())
    classifier.add(Dense(256, activation='relu', kernel_initializer="glorot_uniform"))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(128, activation='relu', kernel_initializer="glorot_uniform"))
    classifier.add(Dropout(rate=0.2))
    classifier.add(Dense(38, activation='softmax', kernel_initializer="glorot_uniform"))

    ###CALLBACKS
    print("INFO:     Currently creating Callbacks")

    #Création d'un Callback de logging
    class TimingCallback(Callback):
        def __init__(self, logs={}):
            self.logs=[]
        def on_epoch_begin(self, epoch, logs={}):
            self.starttime = timer()
        def on_epoch_end(self, epoch, logs={}):
            self.logs.append(timer()-self.starttime)


    #Définition du chemin et du nom de la sauvegarde des meilleurs poids
    curr_date = datetime.datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
    savepath = './' + data_Folder_value + '/Models/Temp_Models/eNetV2S_NPD-Best_Weights-ep_{epoch:02d}-vac_{val_accuracy:.2f}-date_' + curr_date + '.h5'

    #Initialisation des différents Callback pour le modèle
    time_callback = TimingCallback()

    early_stopping = EarlyStopping(patience=5, min_delta = 0.01, mode = 'min', 
                                monitor='val_loss', verbose=1)

    reduce_learning_rate = ReduceLROnPlateau(monitor="val_loss", patience=3, min_delta= 0.01, 
                                            factor=0.1, cooldown = 4, verbose=1)

    checkpoint = ModelCheckpoint(filepath = savepath, monitor = 'val_loss', save_best_only = True,
                                save_weights_only = True, mode = 'min', save_freq = 'epoch', verbose=1)

    ###COMPILE
    print("INFO:     Currently compiling Model")

    #Utilisation des paramètres ayant donnés les meilleurs résultats pendant les tests
    epochs = 20
    learning_rate = 1e-3

    #Compilation du modèle
    classifier.compile(optimizer=Adam(learning_rate=learning_rate), 
                    loss='categorical_crossentropy', metrics=['accuracy'])

    ###TRAINING
    print("INFO:     Currently starting Training")

    classifier.fit(x=training_set_sickness, epochs = epochs, 
                            steps_per_epoch = training_set_sickness.samples // batch_size, 
                            validation_data=valid_set_sickness, 
                            validation_steps=valid_set_sickness.samples // batch_size,
                            callbacks = [checkpoint,
                                        time_callback,
                                        reduce_learning_rate,
                                        early_stopping],
                            workers = -1)

    ###MODEL BACKUP
    print("INFO:     Currently backuping Model")

    today = date.today()
    today = today.strftime("%Y-%m-%d")

    #Sauvegarde du modèle complet
    f_savename = './' + data_Folder_value + '/Models/New_Models/eNetV2S_NPD-Full_Model-' + curr_date + '.h5'
    classifier.save(f_savename)

    #Récupération de l'heure de fin d'entraînement
    end_time = time.time()

    #Calcul du temps d'entrainement
    training_duration = end_time - start_time

    #Mise à False de la variable de suivi pour indiquer que l'entraînement est terminé
    update_train_token_config_map("False")

    return {
        "code": 200,
        "name": "Training Ended OK",
        "duration": float(training_duration),
        "message": "The New Model can be found here : " + f_savename
    }


### INIT 2ND ###

#Remise à "False" du token pour un entraînement unique en cas de redémarrage du conteneur npd-train-api
update_train_token_config_map("False")


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

#Définition de la route "train"
@app.get("/train/{training_type}/")
async def launch_train(training_type: str, username: str = Depends(get_current_user)):
    """
    Description:
    Cette route permet de lancer l'entraînement d'un modèle TensorFlow. L'utilisateur doit spécifier le type d'entraînement en fournissant "full" pour un entraînement complet ou "mini" pour un entraînement miniaturisé. Un nouveau modèle sera entraîné en utilisant les données appropriées et les poids seront sauvegardés. Le chemin du modèle entraîné sera renvoyé.

    Args:
    - training_type (str): Le type d'entraînement ("full" ou "mini").

    Returns:
    - JSON contenant un message indiquant que l'entraînement a été lancé avec succès et le chemin du modèle entraîné.

    Raises:
    - MyException: Avec un code 500 en cas d'erreur lors de l'entraînement du modèle.
    """
    try:
        #Lancement de l'entraînement complet
        if training_type == "full":
            result = await new_train("full")
        
        #Lancement de l'entraînement miniaturisé
        elif training_type == "mini":
            result = await new_train("mini")

        #Message de fin d'entraînement
        return result
        
    except Exception as e:
        raise MyException(
            code=500,
            name='Training Processing Error',
            message='An error occurred while training the model.'
        )
