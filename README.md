Project "Plants and Diseases Recognition"
=========================================

This project aims to create a solution that will help amateur and/or professional gardeners to easily identify the plants in their garden or fields, as well as to diagnose their possible diseases.


Project Organization
--------------------

    ├── LICENSE
    │
    ├── README.md                <- The top-level README for developers using this project
    │
    ├── .github\workflows        <- Source code for all workflows for GitHub Actions
    │   |
    │   └── python-app.yml       <- workflow file use to test the code by Github Actions     
    │
    ├── notebooks                <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                               the creator's initials, and a short `-` delimited description,
    │                               e.g. `1.0-jqp-initial-data-exploration`
    │
    ├── references               <- Data dictionaries, manuals, and all other explanatory materials
    │   |
    │   ├── Projet Reco Maladie - MLOps - Cahier des charges - V1.0.docx   <- Project Specifications
    │   │                                                                     Document
    │   └── Projet Reco Maladie - MLOps - Soutenance - V2.2.pptx           <- Document related to the
    │                                                                         Final Presentation
    ├── docker-setup.sh          <- Shell Script which can be used to create the Docker images and
    │                               launch the containers
    │
    ├── docker-setup.ps1         <- PowerShell Script which can be used to create the Docker images and
    │                               launch the containers on Windows environment
    │
    ├── docker-airflow-setup.sh  <- Shell Script used to launch the airflow specific containers
    │                               
    ├── k8s-setup.sh             <- Shell Script used to deploy the containers whithin a kubernetes
    │                               environment
    │
    ├── src                      <- Source code for use in this project.
    │   |
    │   ├── pyproject.toml             <- Python configuration file
    │   |
    │   ├── airflow                        <- Folder used to store all data needed to create the Docker
    │   |   │                                 images related to Airflow
    │   |   │ 
    │   │   ├── dag_new_picture_email.py   <- Python script use to define the DAG dag_new_picture_email 
    │   │   │                       
    │   │   └── docker-compose.yaml        <- Docker-compose file use to launch the Airflow's Docker images
    │   | 
    │   ├── containers                 <- Folder used to store all data needed to create the Docker images
    │   │   │ 
    │   │   ├── docker-compose.yml     <- Docker-compose file use to launch the Docker images        
    │   │   │                       
    │   │   ├── npd_core_api           <- Folder which contains necessary data to create the Core API
    │   │   │   │
    │   │   │   |── api_core.py        <- Python script used to manage Core API
    │   │   │   │
    │   │   │   |── dockerfile         <- dockerfile used to create the image related to the Core API
    │   │   │   │
    │   │   │   |── requirements.txt   <- file which contains the list of required librairies
    │   │   │   |                         for the Core API
    |   |   |   |
    |   │   |   └── test_api_core.py   <- Python script used to test api_core.py with GitHub Actions
    │   │   │                       
    │   │   ├── npd_pred_api           <- Folder which contains necessary data to create the API
    │   │   │   │                         dedicated to predictions
    │   │   │   |── api_tf_pred.py     <- Python script used to manage Tensorflow API dedicated
    │   │   │   │                         to predictions
    │   │   │   |── dockerfile         <- dockerfile used to create the image related to the API
    │   │   │   │                         dedicated to predictions
    │   │   │   └── requirements.txt   <- file which contains the list of required librairies
    │   │   │                             for the API dedicated to predictions
    │   │   │                       
    │   │   └── npd_train_api          <- Folder which contains necessary data to create the API
    │   │       │                         dedicated to trainings
    │   │       |── api_tf_train.py    <- Python script used to manage Tensorflow API dedicated
    │   │       │                         to trainings
    │   │       |── dockerfile         <- dockerfile used to create the image related to the API
    │   │       │                         dedicated to trainings
    │   │       |── requirements.txt   <- file which contains the list of required librairies
    │   │       │                         for the API dedicated to trainings
    │   │       └── efficientnetv2-s_notop.h5   <- EfficientNet V2S based pre-trained model used
    │   │                                          to perform new trainings
    │   │
    │   ├── data                    <- Main folder which contains all "real" necessary data
    │   │   │                          (Entire "data" folder is listed in .gitignore)
    │   │   ├── BDD                 <- Folder dedicated to store the Databases
    │   │   │   │    
    │   │   │   |── NPD.csv         <- CSV Database used to manage the full "New Plant Diseases"
    │   │   │   │                      Dataset
    │   │   │   |── NPD_Mini.csv    <- CSV Database used to manage the demonstration specific
    │   │   │   │                      "New Plant Diseases" Dataset
    │   │   │   |── NPD_Buffer.csv  <- CSV Database used to manage the Dataset used to store the
    │   │   │   │                      newly added pictures before verification
    │   │   │   └── Users.csv       <- CSV Database which contains the list of known users
    │   │   │
    │   │   ├── Datasets            <- Folder dedicated to store the Databases
    │   │   │   │    
    │   │   │   ├── NPD             <- Folder which contains the full "New Plant Diseases" Dataset
    │   │   │   │   |── train          (training and validation data)
    │   │   │   │   └── valid
    │   │   │   │
    │   │   │   ├── NPD_Mini        <- Folder which contains the demonstration specific
    │   │   │   │   |── train          "New Plant Diseases" Dataset (training and validation data)
    │   │   │   │   └── valid    
    │   │   │   │
    │   │   │   └── NPD_Buffer      <- Folder which contains the Dataset used to store the newly
    │   │   │       └── train          added pictures before verification (training data only)
    │   │   │ 
    │   │   ├── Logs                <- Folder dedicated to store the API related Logs
    │   │   │   └── api_log.json    <- JSON Database used to store all Logs dedicated API
    │   │   │
    │   │   ├── Models                       <- Folder dedicated to store the Models
    │   │   │   │                               (Listed in .gitignore as models are often over 100MB)
    │   │   │   ├── Current_Model            <- Folder which contains the currently used model for
    │   │   │   │   │                           the prediction
    │   │   │   │   └── eNetV2S_NPD_Full.h5  <- Model currently used to predict the classes
    │   │   │   │
    │   │   │   ├── Initial_Model            <- Folder which contains the very initial model for
    │   │   │   │   │                           recording purpose
    │   │   │   │   └── eNetV2S_NPD_Full.h5  <- Initial model used to predict the classes
    │   │   │   │
    │   │   │   ├── New_Models               <- Folder which contains the full backup of a new model
    │   │   │   │   │                           after a training was launched and finished
    │   │   │   │   └── eNetV2S_NPD-Full_Model-%DATE%.h5   <- Full model backup after new training
    │   │   │   │
    │   │   │   └── Temp_Models              <- Folder which contains model's Best Weights created during
    │   │   │       │                           a training at each epoch (in case the training crashes)
    │   │   │       └── eNetV2S_NPD-Best_Weights-%EPOCH%-%ACC%-%DATE%.h5   <- Temporary model weights
    │   │   │                                                                 backed up during training
    │   │   └── Temp                         <- Folder dedicated to temporary store the image for which
    │   │       │                               we will predict the class    
    │   │       └── image_to_predict.jpg     <- Temporary copy of the file given by the user for class
    │   │                                       prediction
    │   │    
    │   ├── data_test               <- Folder which contains all data for testing purposes
    │   │   │    
    │   │   ├── BDD                 <- Folder dedicated to store the Databases
    │   │   │   │    
    │   │   │   |── NPD.csv         <- Minimal version of the CSV Database used to manage the full 
    │   │   │   │                      "New Plant Diseases" Dataset for testing purposes only
    │   │   │   |── NPD_Mini.csv    <- CSV Database used to manage the demonstration specific
    │   │   │   │                      "New Plant Diseases" Dataset    
    │   │   │   |── NPD_Buffer.csv  <- CSV Database used to manage the Dataset used to store the
    │   │   │   │                      newly added picture before verification
    │   │   │   └── Users.csv       <- Minimal version of the Users related CSV Database which only  
    │   │   │                          contains the "test users"
    │   │   ├── Datasets            <- Folder dedicated to store the Databases
    │   │   │   │    
    │   │   │   ├── NPD             <- Folder which contains a minimal version of the full 
    │   │   │   │   |── train          "New Plant Diseases" Dataset for testing purposes only
    │   │   │   │   └── valid          (training and validation data)
    │   │   │   │
    │   │   │   ├── NPD_Mini        <- Folder which contains the demonstration specific
    │   │   │   │   |── train          "New Plant Diseases" Dataset (training and validation data)
    │   │   │   │   └── valid
    │   │   │   │
    │   │   │   └── NPD_Buffer      <- Folder which contains the Dataset used to store the newly
    │   │   │       └── train          added pictures before verification (training data only)
    │   │   │ 
    │   │   ├── Logs                <- Folder dedicated to store the API related Logs
    │   │   │   └── api_log.json    <- JSON Database used to store all Logs dedicated API
    │   │   │
    │   │   ├── Models                       <- Folder dedicated to store the Models
    │   │   │   │                               (Listed in .gitignore as models are often over 100MB)
    │   │   │   ├── Current_Model            <- Folder which contains the currently used model for
    │   │   │   │   │                           the prediction
    │   │   │   │   └── eNetV2S_NPD_Full.h5  <- Model currently used to predict the classes
    │   │   │   │                               (only 0 Kb file in "data_set" folder for tests)
    │   │   │   ├── Initial_Model            <- Folder which contains the very initial model for
    │   │   │   │   │                           recording purpose
    │   │   │   │   └── eNetV2S_NPD_Full.h5  <- Initial model used to predict the classes
    │   │   │   │                               (only 0 Kb file in "data_set" folder for tests)
    │   │   │   ├── New_Models               <- Folder which contains the full backup of a new model
    │   │   │   │   │                           after a training was launched and finished
    │   │   │   │   └── eNetV2S_NPD-Full_Model-%DATE%.h5   <- Full model backup after new training
    │   │   │   │
    │   │   │   └── Temp_Models              <- Folder which contains model's Best Weights created during
    │   │   │       │                           a training at each epoch (in case the training crashes)
    │   │   │       └── eNetV2S_NPD-Best_Weights-%EPOCH%-%ACC%-%DATE%.h5   <- Temporary model weights
    │   │   │                                                                 backed up during training
    │   │   └── Temp                         <- Folder dedicated to temporary store the image for which
    │   │       │                               we will predict the class    
    │   │       └── image_to_predict.jpg     <- Temporary copy of the file given by the user for class
    │   │                                       prediction
    │   │
    │   ├── kubernetes                 <- Folder used to store all data needed to create the Docker images
    │   │   │ 
    │   │   ├── docker-hub-img-builder.sh    <- Shell script used to create (and send) the Docker Hub
    │   │   │                                   specific images that will be used to deploy the kubernetes
    │   │   │                                   environment
    │   │   ├── docker-hub-img-builder.ps1   <- PowerShell script used to create (and send) the Docker Hub
    │   │   │                                   specificimages that will be used to deploy the kubernetes
    │   │   │                                   environment
    │   │   ├── k8s-configmap.yml            <- Configuration file used to create the required kubernetes
    │   │   │                                   variables global to all pods
    │   │   ├── k8s-deployement.yml          <- Configuration file used to deploy the kubernetes pods and
    │   │   │                                   the related containers
    │   │   ├── k8s-ingress.yml              <- Configuration file used to create the ingress related to
    │   │   │                                   the kubernetes deployment
    │   │   ├── k8s-namespace.yml            <- Configuration file used to create the namespace "npd-space"
    │   │   │                                   in which the pods will be deployed
    │   │   ├── k8s-role.yml                 <- Configuration file used to create a role that grants the
    │   │   │                                   permissions to modify the variables defined in the
    │   │   │                                   kubernetes ConfigMap
    │   │   ├── k8s-rolebinding.yml          <- Configuration file used to link the created role to the
    │   │   │                                   default kubernetes service account
    │   │   ├── k8s-secret.yml               <- Configuration file used to create the kubernetes secret
    │   │   │                                   storing the password of the account connecting to the
    │   │   │                                   Tensorflow APIs
    │   │   ├── k8s-service.yml              <- Configuration file used to create the Node Port that will
    │   │   │                                   enable the access to the pods from the outside
    │   │   │
    │   │   ├── npd_core_api           <- Folder which contains necessary data to create the Core API
    │   │   │   │
    │   │   │   |── api_core.py        <- Python script used to manage Core API
    │   │   │   │                         (specific to kubernetes deployment)
    │   │   │   |── dockerfile         <- dockerfile used to create the image related to the Core API
    │   │   │   │                         (specific to kubernetes deployment)
    │   │   │   └── requirements.txt   <- file which contains the list of required librairies
    │   │   │                             for the Core API (specific to kubernetes deployment)
    │   │   │                       
    │   │   ├── npd_pred_api           <- Folder which contains necessary data to create the API
    │   │   │   │                         dedicated to predictions
    │   │   │   |── api_tf_pred.py     <- Python script used to manage Tensorflow API dedicated
    │   │   │   │                         to predictions (specific to kubernetes deployment)
    │   │   │   |── dockerfile         <- dockerfile used to create the image related to the API
    │   │   │   │                         dedicated to predictions (specific to kubernetes deployment)
    │   │   │   └── requirements.txt   <- file which contains the list of required librairies for the
    │   │   │                             API dedicated to predictions (specific to kubernetes deployment)
    │   │   │                       
    │   │   └── npd_train_api          <- Folder which contains necessary data to create the API
    │   │       │                         dedicated to trainings
    │   │       |── api_tf_train.py    <- Python script used to manage Tensorflow API dedicated
    │   │       │                         to trainings (specific to kubernetes deployment)
    │   │       |── dockerfile         <- dockerfile used to create the image related to the API
    │   │       │                         dedicated to trainings (specific to kubernetes deployment)
    │   │       |── requirements.txt   <- file which contains the list of required librairies for the
    │   │       │                         API dedicated to trainings (specific to kubernetes deployment)
    │   │       └── efficientnetv2-s_notop.h5   <- EfficientNet V2S based pre-trained model used
    │   │                                          to perform new trainings
    │   |
    │   └── pict_test                  <- Folder used to store pictures for testing or development
    │                                     purposes
    ┴ 


Endpoints Description
---------------------

1. **Endpoint to check API Status without authentication**
    - **Endpoint:**  
      http://127.0.0.1:8080/apichk (GET)
    - **Parameters:**
      - None
    - **Output:**
      - API Status confirmation message
    - **Example:**
      ```
      curl.exe -X GET http://127.0.0.1:8080/apichk
      ```

2. **Endpoint to check API Status with authentication**
    - **Endpoint:**
      http://127.0.0.1:8080/secapichk (GET)
    - **Parameters:**
      - Base64 encoded user login name and password
    - **Output:**
      - API Status confirmation message
    - **Example:**
      ```
      curl.exe -X GET -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/secapichk
      ```

3. **Endpoint for image class prediction**
    - **Endpoint:**
      http://127.0.0.1:8080/predict/ (POST)
    - **Parameters:**
      - Base64 encoded user login name and password
      - Picture filepath
    - **Output:**
      - Prediction
      - Prediction accuracy
      - Prediction duration
    - **Example:**
      ```
      curl.exe -X POST -F "image=@image_folder/image_to_predict.jpeg" -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/predict/
      ```

4. **Endpoint for initiating a new training (accessible to administrators only)**
    - **Endpoint:**
      - For "full" training: http://127.0.0.1:8080/train/full/ (POST)
      - For "demo" training: http://127.0.0.1:8080/train/mini/ (POST)
    - **Parameters:**
      - Base64 encoded user login name and password
    - **Output:**
      - Training ending message with new folder filepath confirmation
      - Error message if the user is not an "admin"
    - **Example:**
      - Full training:
      ```
      curl.exe -X POST -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/train/full/
      ```
      - Demo training:
      ```
      curl.exe -X POST -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/train/mini/
      ```

5. **Endpoint for adding a new image**
    - **Endpoint:**
      http://127.0.0.1:8080/addpict/ (POST)
    - **Parameters:**
      - Base64 encoded user login name and password
      - Picture filepath
      - Plant type
      - Disease type
    - **Output:**
      - Confirmation message that the newly added picture was added and will be pushed for verification
    - **Example:**
      ```
      curl.exe -X POST -F "plante=Apple" -F "maladie=healthy" -F "image=@image_folder/image_to_add.jpeg" -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/addpict/
      ```

6. **Endpoint for a user to access his log history**
    - **Endpoint:**
      http://127.0.0.1:8080/gimmelogs (GET)
    - **Parameters:**
      - Base64 encoded user login name and password
    - **Output:**
      - Reply of Log elements related to the connected user in JSON format
    - **Example:**
      ```
      curl.exe -X GET -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/gimmelogs
      ```

7. **Endpoint to view the entirety of logs (accessible to administrators only)**
    - **Endpoint:**
      http://127.0.0.1:8080/getalllogs (GET)
    - **Parameters:**
      - Base64 encoded user login name and password
    - **Output:**
      - Reply of all the existing Logs in JSON format
      - Error message if the user is not an "admin"
    - **Example:**
      ```
      curl.exe -X GET -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/getalllogs
      ```

8. **Endpoint to add a new user**
    - **Endpoint:**
      http://127.0.0.1:8080/adduser (POST)
    - **Parameters:**
      - Username of the new user
      - Firstname of the new user
      - Lastname of the new user
      - Email of the new user
      - Password of the new user
    - **Output:**
      - Confirmation message that the newly added user was added 
      - Error message if the username is already existed
    - **Example:**
      ```
      curl.exe -X POST -F "ident=identUser" -F "prenom=firstNameUser" -F "nom=nameUser" -F "email=emailUser" -F "motdepasse=passwordUser" http://127.0.0.1:8080/adduser/
      ```

9. **Endpoint to add a new opinion**
    - **Endpoint:**
      http://127.0.0.1:8080/opinion (POST)
    - **Parameters:**
      - Base64 encoded user login name and password
      - Name of the predicted picture
      - Opinion about the prediction
      - Opinion about the length of time of the prediction
    - **Output:**
      - Confirmation message that the newly added opinion was added
    - **Example:**
      ```
      curl.exe -X POST -F "image=imageName.jpeg" -F "opinionPrediction=5" -F "opinionDuree=5" -H "Authorization: Basic ABCDEF123456" http://127.0.0.1:8080/opinion/
      ```
----------
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>