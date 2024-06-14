import os
import mlflow
import pandas as pd
import seaborn as sns
import psycopg2 as psycopg
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

ASSETS_DIR = 'assets'
RANDOM_STATE = 42
EXPERIMENT_NAME = 'click_pred'

load_dotenv()
connection = {"sslmode": "require",
              "target_session_attrs": "read-write"}
postgres_credentials = {"host": os.getenv("DB_HOST"),
                        "port": os.getenv("DB_PORT"),
                        "dbname": os.getenv("DB_MLFLOW_NAME"),
                        "user": os.getenv("DB_USERNAME"),
                        "password": os.getenv("DB_PASSWORD")}
TRACKING_SERVER_PORT = os.getenv("TRACKING_SERVER_PORT")
TRACKING_SERVER_HOST = os.getenv("TRACKING_SERVER_HOST")

pd.options.display.max_columns = 100
pd.options.display.max_rows = 64
pd.options.mode.copy_on_write = True

sns.set_theme(style='white', palette='husl')

def data_review(dataset):
    '''
    на вход принимает датафрейм,
    выводит общую обзорную информацию
    '''
    print('Общая информация о наборе данных:')
    dataset.info()
    print()
    print('Первые пять строк набора данных:')
    display(dataset.head())
    print()
    print(f"количество полных дубликатов строк: {dataset.duplicated().sum()}")
    print()
    print(f"""количество пропущенных значений: 
    {dataset.isna().sum()}""")
    print()
    print('Вывод количества уникальных записей в каждом числовом признаке:')
    for column in dataset.select_dtypes(include=['int', 'float']).columns:
        unique_values = dataset[column].nunique()
        print(f"Количество уникальных записей в признаке '{column}': {unique_values}")

def data_preprocessing(test_size=0.1, dataset=None, features=None, target=None, add_features=False):
    X_train, X_test, y_train, y_test = train_test_split(dataset[features],
                                                        dataset[target],
                                                        test_size=test_size,
                                                        random_state=RANDOM_STATE,
                                                        stratify=dataset[target])
    scaler = StandardScaler()
    X_train_scl = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scl = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
    if add_features:
        transformations = transformations
        afc = AutoFeatClassifier(categorical_cols=categorical_features, feateng_steps=1, max_gb=2, transformations=transformations, n_jobs=-1)
        X_train_afc = afc.fit_transform(X_train, y_train)
        X_test_afc = afc.transform(X_test)
        print(f'размерности выборок: {X_train_afc.shape, X_test_afc.shape, y_train.shape, y_test.shape}')
        return X_train_afc, X_test_afc, y_train, y_test
    else:
        print(f'размерности выборок: {X_train_scl.shape, X_test_scl.shape, y_train.shape, y_test.shape}')
        return X_train_scl, X_test_scl, y_train, y_test

def model_fitting(model_name=None, features_train=None, target_train=None, n_splits=None, params=None):
    if model_name == 'Baseline' or model_name == 'Logistic Regression':
        model = LogisticRegression(**params)
    elif model_name == 'Random Forest':
        model = RandomForestClassifier(**params)
    model.fit(features_train.values, target_train.values)
    cv_strategy = StratifiedKFold(n_splits=n_splits)
    cv_res = cross_validate(model,
                            features_train,
                            target_train,
                            cv=cv_strategy,
                            n_jobs=-1,
                            scoring=['roc_auc', 'f1_micro', 'f1', 'f1_weighted', 'f1_macro'])
    for key, value in cv_res.items():
        cv_res[key] = round(value.mean(), 3)
    print(f"результаты кросс-вадидации: {cv_res}")
    y_pred = model.predict(features_train.values)
    y_pred_proba = model.predict_proba(features_train.values)[:, 1]
				
    roc_auc_value = roc_auc_score(target_train, y_pred_proba)
    f1_value = f1_score(target_train, y_pred)

    fpr, tpr, thresholds = roc_curve(target_train, y_pred_proba)
    plt.plot(fpr, tpr, linewidth=1.5, label='ROC-AUC (area = %0.2f)' % roc_auc_value)
    plt.plot([0, 1], [0, 1], linestyle='--', linewidth=1.5, label='random_classifier')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title(f'{model_name} Receiver Operating Characteristic', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(os.path.join(ASSETS_DIR, f'{model_name} Receiver Operating Characteristic.png'))
    plt.show()

    return cv_res['test_f1'], cv_res['test_roc_auc'], model

def model_logging(signature=None, input_example=None, metadata=None, metrics=None, model=None, params=None, run_name=None, reg_model_name=None):
    mlflow.set_tracking_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")
    mlflow.set_registry_uri(f"http://{TRACKING_SERVER_HOST}:{TRACKING_SERVER_PORT}")

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    experiment_id = mlflow.set_experiment(EXPERIMENT_NAME).experiment_id

    pip_requirements = 'requirements.txt'
    code_paths = ['ad_click_prediction.ipynb']

    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
        run_id = run.info.run_id
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path='artifacts',
            pip_requirements=pip_requirements,
            signature=signature,
            input_example=input_example,
            metadata=metadata,
            code_paths=code_paths,
            registered_model_name=reg_model_name,
            await_registration_for=60
        )

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    run = mlflow.get_run(run_id)
    assert run.info.status == "FINISHED"

def models_comparison(save_figure=False):
    connection.update(postgres_credentials)
    with psycopg.connect(**connection) as conn:
        with conn.cursor() as cur:
            cur.execute('''
                    SELECT
                      experiments.name AS experiment_name,
                      runs.name AS run_name,
                      model_versions.name AS model_name,
                      model_versions.version AS model_version,
                      MAX(CASE WHEN metrics.key = 'f1_score' THEN metrics.value END) AS f1_score,
                      MAX(CASE WHEN metrics.key = 'roc_auc_score' THEN metrics.value END) AS roc_auc_score
                    FROM experiments
                      LEFT JOIN runs USING (experiment_id)
                      LEFT JOIN metrics USING (run_uuid)
                      LEFT JOIN model_versions ON model_versions.run_id=runs.run_uuid
                    WHERE
                      experiments.name = %s
                      AND model_versions.name IS NOT Null
                    GROUP BY
                      experiments.name,
                      runs.name,
                      model_versions.name,
                      model_versions.version
                    ORDER BY roc_auc_score
                    ''', (EXPERIMENT_NAME,))
            table_data = cur.fetchall()
            table_columns = [desc[0] for desc in cur.description]
            print('Models and their metrics:')
            models_data = pd.DataFrame(table_data, columns=table_columns)
            display(models_data)
    plt.figure(figsize=(12, 6))
    metrics = ['f1_score', 'roc_auc_score']

    for i, metric in enumerate(metrics):
        plt.subplot(1, 3, i+1)
        sns.barplot(x='model_name', y=metric, data=models_data)
        plt.title(f'Comparison of {metric.upper()}')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    if save_figure:
        plt.savefig(os.path.join(ASSETS_DIR, f'Comparison of model metrics.png'))
    plt.show()