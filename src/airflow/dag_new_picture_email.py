from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.python_operator import BranchPythonOperator
from airflow.operators.email_operator import EmailOperator
from airflow.models import Variable
import pandas as pd

# Récupération de la variable path_fic
path_fic = Variable.get(key="path_fic")
    
my_dag = DAG(
    dag_id='dag_new_picture_email',
    doc_md="""## Ce DAG est pour le projet reconnaissance de plantes
    Il permet d'envoyer un mail aux administrateurs si de nouvelles images ont été ajoutées dans le fichier NPD_Buffer.csv
    Ce DAG comporte plusieurs tâches :
    * tâche 1 : sensor_NPD_Buffer : permet de vérifier l'existance du fichier NPD_Buffer.csv
    * tâche 2 : sensor_Users - permet de vérifier l'existance du fichier Users.csv
    * tâche 3 : task_Check_New_Picture - permet de vérifier s'il y a de nouvelles images à valider - tâche conditionnelle : renvoie le nom de la tâche suivante
    * tâche 4 : task_send_email - permet d'envoyer un message
    * tâche 5 : task_no_email - ne fait rien
    Voici l'enchainement des tâches :
    sensor_NPD_Buffer >> sensor_Users
    sensor_Users >> task_Check_New_Picture
    task_Check_New_Picture >> task_send_email
    task_Check_New_Picture >> task_no_email
    """,
    tags=['projet reconnaissance plantes', 'datascientest'],
    #commence maintenant sans reprise
    start_date=days_ago(2),
    #Exécution toutes les minutes
    schedule_interval='0 1 * * *',
    catchup=False
)

def GetRecipients():
    """
    Fonction permettant de renvoyer la liste des adresses emails des administrateurs
    """
    #chargement de la base des utilisateurs
    path_npd_Users = path_fic + "Users.csv"
    npd_Users = pd.read_csv(path_npd_Users,header=0)
    
    # Sélection des adminsitrateurs uniquement
    npd_Admin = npd_Users[npd_Users["access_type"] == "admin"]
    lstEmail = []
    for unEmail in npd_Admin["email"]:
        lstEmail.append(unEmail)
    print("liste des destinataires : ", lstEmail)
    return lstEmail


def CheckNewPicture(task_instance):
    """
    Fonction permettant de vérifier s'il y a de nouvelles images à valider, c'est à dire les images ayant le code status à 0 dans le fichier
    NPD_Buffer.csv
    Renvoie le nom de la tâche à exécuter à la suite :
    - "task_sendemail" : s'il y a de nouvelles images à valider
    - " task_noemail" : s'il n'y a pas de nouvelles images à valider
    """
    #chargement de la base NPD_Buffer
    path_npd_Buffer = path_fic + "NPD_Buffer.csv"
    npd_Buffer = pd.read_csv(path_npd_Buffer,header=0)
    
    # initialisation du compteur de nouvelles images
    nbNewPicture = 0
    for theStatus in npd_Buffer["Status"]:
        if theStatus == 0:
            nbNewPicture += 1
    
    if nbNewPicture >= 1:
        print("Nouvelles images : ", nbNewPicture)
        return "task_sendemail"    
    else:
        print("Pas de nouvelles images")
        return "task_noemail"

def NoEmail():
    """
    Fonction qui ne fait rien et qui est nécessaire pour la tâche task_noemail
    """
    print ("Pas d'envoi d'email")
    
"""
Définition des tâches
"""
#Recherche de la liste des destinataires du message
recipients = GetRecipients()

sensor_NPD_Buffer=FileSensor(
    task_id="check_file_NPD_Buffer",
    fs_conn_id="connection_data_BDD",
    filepath="NPD_Buffer.csv",
    poke_interval=10,
    dag=my_dag,
    timeout=3 * 10,
    mode='reschedule'
)

sensor_Users=FileSensor(
    task_id="check_file_Users",
    fs_conn_id="connection_data_BDD",
    filepath="Users.csv",
    poke_interval=10,
    dag=my_dag,
    timeout=3 * 10,
    mode='reschedule'
)

task_Check_New_Picture = BranchPythonOperator(
    task_id='task_CheckNewPicture',
    python_callable=CheckNewPicture,
    doc_md=""" # Task task_Check_New_Picture
    Tâche permettant la vérification de la présence de nouvelles images à valider dans la base de données temporaire""",
    trigger_rule="all_done",
    dag=my_dag
)


task_send_email = EmailOperator(
    task_id='task_sendemail',
    to=recipients,
    subject='Nouvelles images à valider',
    html_content='<p>Bonjour, vous avez de nouvelles images à valider</p>',
    dag=my_dag,
)

task_no_email = PythonOperator(
    task_id='task_noemail',
    python_callable=NoEmail,
    doc_md=""" # Task task_no_email
    Tâche vide pour le cas où il n'y a pas de nouvelles images à valider""",
    dag=my_dag
)

sensor_NPD_Buffer >> sensor_Users
sensor_Users >> task_Check_New_Picture
task_Check_New_Picture >> task_send_email
task_Check_New_Picture >> task_no_email