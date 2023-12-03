from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import Regression4
import YB_function
import regression_optimise
from airflow.models import Variable


api_key = "6a1f139e7d8df9dce68bb9f44a9a9f1a"
cities = ['seville', 'rabat', 'alger','damas']
#repertoire_stockage = 'app/raw_files'
#repertoire_data = 'app/clean_data'

repertoire_stockage = './raw_files' # '/home/ubuntu/raw_files'
repertoire_data = './clean_data' #'/home/ubuntu/clean_data'


my_dag = DAG(
    dag_id='Examen_DashBoard_version',
    description='Le Dag s exécute toutes les minutes',
    tags=['ExamenDag_2'],
    schedule_interval='* * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
)

Task1 = PythonOperator(
    task_id='collectData',
    dag=my_dag,
    python_callable=YB_function.collect_data,
    #retry_delay=datetime.timedelta(seconds=61)
)


Task2 = PythonOperator(
    task_id='ConcatDataALL',
    dag=my_dag,
    python_callable=YB_function.transform_data_into_csv,
    #retry_delay=datetime.timedelta(seconds=61)
)

Task3 = PythonOperator(
    task_id='ConcatData20',
    dag=my_dag,
    python_callable=YB_function.transform_data_into_csv_20,
    #retry_delay=datetime.timedelta(seconds=61)
)


Task4 = PythonOperator(
    task_id='LinearRegression',
    dag=my_dag,
    python_callable=regression_optimise.compute_model_score_LinearRegression,
    #retry_delay=datetime.timedelta(seconds=61)
    provide_context=True,
    trigger_rule='all_success'
)

Task44 = PythonOperator(
    task_id='RandomForestRegressor',
    dag=my_dag,
    python_callable=regression_optimise.compute_model_score_RandomForestRegressor,
    provide_context=True,
    trigger_rule='all_success'
    #retry_delay=datetime.timedelta(seconds=61)
)

Task444 = PythonOperator(
    task_id='DecisionTreeRegressor',
    dag=my_dag,
    python_callable=regression_optimise.compute_model_score_DecisionTreeRegressor,
    provide_context=True,
    trigger_rule='all_success'
    #retry_delay=datetime.timedelta(seconds=61)
)

Task5 = PythonOperator(
    task_id='MeilleurScore',
    dag=my_dag,
    provide_context=True,
    python_callable=regression_optimise.choisir_le_meilleur_score,
    trigger_rule='all_success',

    #retry_delay=datetime.timedelta(seconds=61)"""
)
Task1 >> Task2
Task1 >> Task3
Task2 >> [Task4,Task444,Task44]>> Task5
