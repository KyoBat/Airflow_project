import datetime
import logging
import os
import requests
import json
import pandas as pd


api_key = "6a1f139e7d8df9dce68bb9f44a9a9f1a"
cities = ['seville', 'rabat', 'alger','damas']
#cities = ['damas']
repertoire_stockage = './raw_files' # '/home/ubuntu/raw_files'
repertoire_data = './clean_data' #'/home/ubuntu/clean_data'

def collect_data():
    logging.info("Collecting data...")
    date_formattee = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')
    json_list =[]
    for city in cities :
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
        response = requests.get(url)
        
    # Vérifiez si la requête a réussi
        if response.status_code == 200:
            data = response.json()
            json_list.append(data)
    
    if not os.path.exists(repertoire_stockage):
        os.makedirs(repertoire_stockage)
        # Chemin complet du fichier dans le répertoire de destination
    chemin_fichier = os.path.join(repertoire_stockage, f"{date_formattee}.json")

        # Écrivez les données dans le fichier en mode append
    with open(chemin_fichier, 'w') as fichier:
        json.dump(json_list, fichier)
            
    return json_list

#collect_data()

def transform_data_into_csv(n_files=None, filename='fulldata.csv'):
    logging.info("Transforming FULL data into CSV...")
    #repertoire_stockage = '/home/ubuntu/raw_files'
    files = sorted(os.listdir(repertoire_stockage), reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []

    for f in files:
        print(f)
        with open(os.path.join(repertoire_stockage, f), 'r') as file:
            try:
                data_temp = json.load(file)
                if isinstance(data_temp, list):
                    for data_city in data_temp:
                        if isinstance(data_city, dict):
                            dfs.append(
                                {
                                    'temperature': data_city['main']['temp'],
                                    'city': data_city['name'],
                                    'pression': data_city['main']['pressure'],
                                    'date': f.split('.')[0]
                                }
                            )
                        else:
                            print("Les données dans le fichier ne sont pas au format JSON valide.")
                else:
                    print("Le fichier ne contient pas de liste JSON valide.")
            except json.JSONDecodeError:
                print("Le fichier n'est pas au format JSON valide.")

    df = pd.DataFrame(dfs)

    print('\n', df.head(10))

    if not os.path.exists(repertoire_data):
        os.makedirs(repertoire_data)

    df.to_csv(os.path.join(repertoire_data, filename), index=False)

    #return df


def transform_data_into_csv_20(n_files=20, filename='data.csv'):
    logging.info("Transforming 20 data into CSV...")
    #repertoire_stockage = '/home/ubuntu/raw_files'
    files = sorted(os.listdir(repertoire_stockage), reverse=True)
    if n_files:
        files = files[:n_files]

    dfs = []

    for f in files:
        print(f)
        with open(os.path.join(repertoire_stockage, f), 'r') as file:
            try:
                data_temp = json.load(file)
                if isinstance(data_temp, list):
                    for data_city in data_temp:
                        if isinstance(data_city, dict):
                            dfs.append(
                                {
                                    'temperature': data_city['main']['temp'],
                                    'city': data_city['name'],
                                    'pression': data_city['main']['pressure'],
                                    'date': f.split('.')[0]
                                }
                            )
                        else:
                            print("Les données dans le fichier ne sont pas au format JSON valide.")
                else:
                    print("Le fichier ne contient pas de liste JSON valide.")
            except json.JSONDecodeError:
                print("Le fichier n'est pas au format JSON valide.")

    df = pd.DataFrame(dfs)

    print('\n', df.head(10))

    if not os.path.exists(repertoire_data):
        os.makedirs(repertoire_data)

    df.to_csv(os.path.join(repertoire_data, filename), index=False)

    #return df

#transform_data_into_csv(5, 'data.csv')
#transform_data_into_csv(None, 'fulldata.csv')

