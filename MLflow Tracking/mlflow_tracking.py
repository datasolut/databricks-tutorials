# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://datasolut.com/wp-content/uploads/2020/01/logo-horizontal.png" alt="Databricks Learning" style="width: 300px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## <img src="https://yt3.googleusercontent.com/ytc/AL5GRJWSDkfSdYxhsFknPhzJUWjYkHMEdYJHRO_AuCzlfQ=s900-c-k-c0x00ffffff-no-rj" alt="ds Logo Tiny" width="100" height="100"/> MLflow Tracking
# MAGIC 
# MAGIC ### In diesem Video werden wir MLflow nutzen, um
# MAGIC - Parameter, Metriken, Modelle des Trainingsprozesses zu tracken
# MAGIC - bereits durchgeführte und getrackte Trainingsprozesse aufrufen
# MAGIC 
# MAGIC Schlüsselwörter: MLflow, Spark, Tracking, pandas, sklearn, MLOps
# MAGIC 
# MAGIC Dokumentation MLflow - https://mlflow.org/docs/latest/index.html

# COMMAND ----------

# MAGIC %md
# MAGIC ### Runs und Experiments
# MAGIC 
# MAGIC MLflow basiert auf dem Prinzip von Runs. 
# MAGIC 
# MAGIC Ein Run besteht aus einer Ausführung des Codes. 
# MAGIC 
# MAGIC Mehrere Runs werden dann in einem Experiment zusammengefasst. Ein MLflow-Server kann mehrere Runs hosten.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lade Datensatz Airbnb - San Francisco und trainiere ein ML-Modell
# MAGIC http://insideairbnb.com/get-the-data/

# COMMAND ----------

import pandas as pd

url = "http://data.insideairbnb.com/united-states/ca/san-francisco/2022-12-04/data/listings.csv.gz"

df = pd.read_csv(url, compression='gzip', sep=",")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verwenden nur einige Spalten
# MAGIC Zur Vereinfachung filtern wir die Spalten
# MAGIC - **`host_response_rate`**
# MAGIC - **`host_acceptance_rate`**
# MAGIC - **`bedrooms`**
# MAGIC - **`beds`**
# MAGIC - **`reviews_per_month`**
# MAGIC - **`review_scores_rating`**
# MAGIC - **`review_scores_accuracy`**
# MAGIC - **`review_scores_cleanliness`**
# MAGIC - **`review_scores_checkin`**
# MAGIC - **`review_scores_communication`**
# MAGIC - **`review_scores_location`**
# MAGIC - **`review_scores_value`**
# MAGIC - **`latitude`**
# MAGIC - **`longitude`**
# MAGIC 
# MAGIC um **`price`** vorherzusagen.

# COMMAND ----------

# für die Modellierung beschränken wir uns auf eine Auswahl von Spalten
column_names = ['host_response_rate', 'host_acceptance_rate',
 "bedrooms", "review_scores_rating",
 "review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin",
 "review_scores_communication", "review_scores_location", "review_scores_value",
 "latitude", "longitude", "price"]

df_modelling = df[column_names]

# mit dem Schema können wir sehen, welche Datentypen angepasst werden müssen
display(df_modelling)

df_modelling.info()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Datenbearbeitung für das Modell-Training
# MAGIC - Fasse einige Review Scores zu einem Gesamt-Score zusammen
# MAGIC - Konvertiere **`price`** von string zu float
# MAGIC - Entferne '%' in den beiden Spalten mit Raten und konvertiere zu float
# MAGIC - Zeilen, die *null*-values enthalten, werden entfernt

# COMMAND ----------

# einzelne review scores werden zu einem gesamten score summiert
df_modelling.loc[:, "review_scores_sum"] = (
    df_modelling["review_scores_accuracy"] + 
    df_modelling["review_scores_cleanliness"]+
    df_modelling["review_scores_checkin"] + 
    df_modelling["review_scores_communication"] + 
    df_modelling["review_scores_location"] + 
    df_modelling["review_scores_value"]
)

df_modelling = df_modelling.drop(["review_scores_accuracy",
                                  "review_scores_cleanliness",
                                  "review_scores_checkin",
                                  "review_scores_communication",
                                  "review_scores_location",
                                  "review_scores_value"], axis=1
                                )

# COMMAND ----------

# Konvertierung von longitude, latitude und price zu numerischen Wert
df_modelling.loc[:, 'price'] = df_modelling['price'].replace({'\$|,': ''}, regex=True).astype(float)
                   

# Antwort- und Akzeptanzrate müssen in ein numerisches Format konvertiert werden
# Antwort- und Akzeptanzrate müssen in ein numerisches Format konvertiert werden
df_modelling.loc[:, 'host_response_rate'] = df_modelling['host_response_rate'].str.strip('%').astype(float) / 100
df_modelling.loc[:, 'host_acceptance_rate'] = df_modelling['host_acceptance_rate'].str.strip('%').astype(float) / 100

display(df_modelling)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Entferne Zeilen mit *null*-values aus dem Datensatz

# COMMAND ----------

# Zeilen, die null values enthalten werden entfernt
df_modelling = df_modelling.dropna()
display(df_modelling)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC Nun ist der Datensatz befreit von *null*-Werten.<br>
# MAGIC Als nächstes teilen wir die Daten in einen Trainings- und Testdatensatz auf.<br>

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_modelling.drop(['price'], axis=1), df_modelling['price'], test_size=0.1, random_state=42)

print("Größe von X_train:", len(X_train))
print("Größe von X_test:", len(X_test))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Training eines Random Forests und erstes Tracking mithilfe von MLflow

# COMMAND ----------

import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

with mlflow.start_run(run_name="Erster Run") as run:
    # Erstelle Modell und trainiere dieses
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)

    # Log Modell
    mlflow.sklearn.log_model(rf, "random_forest_model")

    # Log Metriken
    mse = mean_squared_error(y_test, predictions)
    mlflow.log_metric("mse", mse)

    run_id = run.info.run_id
    experiment_id = run.info.experiment_id

    print(f"Run ID: `{run_id}` und Experiment ID: `{experiment_id}`")

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### Fortgeschritteneres Tracking mit MLflow
# MAGIC 
# MAGIC Wir schreiben uns nun eine Funktion, welche verschiedene Argumente erhält und damit das Tracking durchführt.
# MAGIC 
# MAGIC Zusätzlich tracken wir noch
# MAGIC - eine Signatur (Schema des Trainingsdatensatzes)
# MAGIC - Input Examples (Einige Zeilen aus dem Trainingsdatensatz um schnell auszutesten, ob das Modell korrekt ausgeführt werden kann)

# COMMAND ----------

# Erstelle Speicherort für Artifacts
dbutils.fs.mkdirs("dbfs:/mnt/mlflow-rf-feature-importances/")

# COMMAND ----------

import matplotlib.pyplot as plt
from mlflow.models.signature import infer_signature
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def log_rf(experiment_id, run_name, params, X_train, X_test, y_train, y_test):
  
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as run:
        # Erstelle und trainiere Modell
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)
        predictions = rf.predict(X_test)

        signature = infer_signature(X_train, pd.DataFrame(y_train))
        input_example = X_train.head(3)

        # Log model
        mlflow.sklearn.log_model(rf, "random_forest_model", signature=signature, input_example=input_example)

        # Log params
        mlflow.log_params(params)

        # Log metrics
        mlflow.log_metrics({
            "mse": mean_squared_error(y_test, predictions), 
            "mae": mean_absolute_error(y_test, predictions), 
            "r2": r2_score(y_test, predictions)
        })

        # Log feature importance
        importance = (pd.DataFrame(list(zip(X_train.columns, rf.feature_importances_)), columns=["Feature", "Importance"])
                      .sort_values("Importance", ascending=False))
        importance = importance.set_index('Feature') 
        
        importance_path = "/dbfs/mnt/mlflow-rf-feature-importances/features.csv"
        importance.to_csv(importance_path, index=True) # index wird zu csv hinzugefuegt
        mlflow.log_artifact(importance_path, "feature-importances.csv")

        # Log plot
        fig, ax = plt.subplots()
        importance.plot.barh(ax=ax) 
        plt.title("Feature Importances")
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        mlflow.log_figure(fig, "feature_importances.png")

        return run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC #### Tracking eines zweiten Runs

# COMMAND ----------

params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42
}

log_rf(experiment_id, "Zweiter Run", params, X_train, X_test, y_train, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Informationen zu vergangenen Runs erhalten
# MAGIC 
# MAGIC Informationen zu bereits getrackten Runs lassen sich über MLflowClient verwalten.
# MAGIC 
# MAGIC Dazu müssen wir zunächst ein Client-Objekt erstellen.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Zeige alle Metainformationen zu vergangene Runs eines Experiments

# COMMAND ----------

runs = spark.read.format("mlflow-experiment").load(experiment_id)
display(runs)

# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Artificats eines Runs anzeigen

# COMMAND ----------

run_rf = runs.orderBy("start_time", ascending=False).first()

client.list_artifacts(run_rf.run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Metriken eines Runs anzeigen

# COMMAND ----------

client.get_run(run_rf.run_id).data.metrics

# COMMAND ----------

# MAGIC %md
# MAGIC #### Modell eines Runs laden

# COMMAND ----------

# Lade bereits getracktes Modell
model = mlflow.sklearn.load_model(f"runs:/{run_rf.run_id}/random_forest_model")
model.feature_importances_

# Zeige die Feature Importance als Dataframe an
feature_importances = model.feature_importances_
feature_names = X_train.columns

print(len(feature_names))
print(len(model.feature_importances_))

importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

display(importance_df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Kurzer Clean-Up
# MAGIC 
# MAGIC - Lösche die gespeicherten csv-Dateien der Feature Importance im dbfs

# COMMAND ----------

display(dbutils.fs.ls("/mnt/"))
dbutils.fs.rm("/mnt/mlflow-rf-feature-importances/features.csv")
dbutils.fs.rm("/mnt/mlflow-rf-feature-importances/")
display(dbutils.fs.ls("/mnt/"))
