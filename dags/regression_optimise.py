import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from joblib import dump


def compute_model_score(model, X, y):
    # computing cross val
    cross_validation = cross_val_score(
        model,
        X,
        y,
        cv=3,
        scoring='neg_mean_squared_error')

    model_score = cross_validation.mean()

    return model_score


def prepare_data(path_to_data='./clean_data/fulldata.csv'):

    # reading data
    df = pd.read_csv(path_to_data)
    # ordering data according to city and date
    df = df.sort_values(['city', 'date'], ascending=True)

    dfs = []

    for c in df['city'].unique():
        df_temp = df[df['city'] == c]

        # creating target
        df_temp.loc[:, 'target'] = df_temp['temperature'].shift(1)

        # creating features
        for i in range(1, 10):
            df_temp.loc[:, 'temp_m-{}'.format(i)
                        ] = df_temp['temperature'].shift(-i)

        # deleting null values
        df_temp = df_temp.dropna()

        dfs.append(df_temp)

    # concatenating datasets
    df_final = pd.concat(
        dfs,
        axis=0,
        ignore_index=False
    )

    # deleting date variable
    df_final = df_final.drop(['date'], axis=1)

    # creating dummies for city variable
    df_final = pd.get_dummies(df_final)

    features = df_final.drop(['target'], axis=1)
    target = df_final['target']

    """ kwargs['ti'].xcom_push(
        key="my_xcom_target",
        value=features.to_json(orient='split')
    )

    kwargs['ti'].xcom_push(
        key="my_xcom_features",
        value=features.to_json(orient='split')
    )"""
    
    return features, target

def compute_model_score_LinearRegression(task_instance,**kwargs):
    """ti = kwargs['ti']
    X = ti.xcom_pull(task_ids='task_pushing_id', key='my_xcom_target')
    y = ti.xcom_pull(task_ids='task_pushing_id', key='my_xcom_features')"""

    X, y = prepare_data('./clean_data/fulldata.csv')

    score_lr = compute_model_score(LinearRegression(), X, y)
    task_instance.xcom_push(
        key="my_xcom_score_lr",
        value=score_lr
    )

    kwargs['ti'].xcom_push(
        key="my_xcom_score_lr_tableau",
        value=score_lr
    )
    return score_lr

def compute_model_score_DecisionTreeRegressor(task_instance,**kwargs):
    """ti = kwargs['ti']
    X = ti.xcom_pull(task_ids='task_pushing_id', key='my_xcom_target')
    y = ti.xcom_pull(task_ids='task_pushing_id', key='my_xcom_features')"""
    X, y = prepare_data('./clean_data/fulldata.csv')

    score_dt = compute_model_score(DecisionTreeRegressor(), X, y)
    task_instance.xcom_push(
        key="my_xcom_score_dt",
        value=score_dt
    )
    kwargs['ti'].xcom_push(
        key="my_xcom_score_dt_tableau",
        value=score_dt
    )
    return score_dt


def compute_model_score_RandomForestRegressor(task_instance,**kwargs):
    """X =task_instance.xcom_pull(key="my_xcom_score_Forest")
    score_dt =task_instance.xcom_pull(key="my_xcom_score_dt")"""
    X, y = prepare_data('./clean_data/fulldata.csv')

    score_Forest= compute_model_score(RandomForestRegressor(), X, y)
    task_instance.xcom_push(
        key="my_xcom_score_Forest",
        value=score_Forest
    )
    kwargs['ti'].xcom_push(
        key="my_xcom_score_Forest_tableau",
        value=score_Forest
    )
    return score_Forest

def train_and_save_model(model, X, y, path_to_model='./app/model.pckl'):
    # training the model
    model.fit(X, y)
    # saving model
    print(str(model), 'saved at ', path_to_model)
    dump(model, path_to_model)

def choisir_le_meilleur_score(task_instance,**kwargs):
    score_forest =task_instance.xcom_pull(key="my_xcom_score_Forest")
    score_dt =task_instance.xcom_pull(key="my_xcom_score_dt")
    score_lr =task_instance.xcom_pull(key="my_xcom_score_lr")
    X, y = prepare_data('./clean_data/fulldata.csv')
    ti = kwargs['ti']
    score_Forest_tabl = ti.xcom_pull(key='my_xcom_score_Forest_tableau')
    score_dt_tabl = ti.xcom_pull(key='my_xcom_score_dt_tableau')
    score_lr_tabl = ti.xcom_pull(key='my_xcom_score_lr_tableau')
    
    print("Je PREMIERRRRRRRR",score_forest,score_dt )
    print("Je suis la", score_lr_tabl,score_dt_tabl,score_Forest_tabl)
    liste_de_doubles = [score for score in [score_forest,score_dt,score_lr] if score is not None]
    valeur_maximale = None

    if liste_de_doubles:
        valeur_maximale = max(liste_de_doubles)
        print("La valeur maximale dans la liste est :", valeur_maximale)
    else:
        print("Tous les scores sont None.")

    if (valeur_maximale is not None) and(valeur_maximale == score_forest ):
        train_and_save_model(
            RandomForestRegressor(),
            X,
            y,
            './clean_data/best_model.pickle'
        )
    if (valeur_maximale is not None)and(valeur_maximale == score_dt ):
        train_and_save_model(
            DecisionTreeRegressor(),
            X,
            y,
            './clean_data/best_model.pickle'
        )
    if (valeur_maximale is not None)and(valeur_maximale == score_lr ):
        train_and_save_model(
            LinearRegression(),
            X,
            y,
            './clean_data/best_model.pickle'
        )
    return valeur_maximale


    