# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://datasolut.com/wp-content/uploads/2020/01/logo-horizontal.png" alt="Databricks Learning" style="width: 300px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## <img src="https://yt3.googleusercontent.com/ytc/AL5GRJWSDkfSdYxhsFknPhzJUWjYkHMEdYJHRO_AuCzlfQ=s900-c-k-c0x00ffffff-no-rj" alt="ds Logo Tiny" width="100" height="100"/> MLflow Registry
# MAGIC 
# MAGIC ### In diesem Video werden wir MLflow nutzen, um
# MAGIC - Modelle zu registrieren
# MAGIC - Übergänge zwischen Modellphasen durchzuführen
# MAGIC - Modelle zu versionieren
# MAGIC - Modelle aus der Registry zu löschen
# MAGIC 
# MAGIC Schlüsselwörter: MLflow, Spark, Registry
# MAGIC 
# MAGIC Dokumentation MLflow - https://mlflow.org/docs/latest/index.html
# MAGIC 
# MAGIC Besuchen Sie auch gerne die [datasolut-Website](https://datasolut.com/) und unseren [Youtube-Kanal](https://www.youtube.com/@datasolut6213)!

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC ### Lade Datensatz Airbnb - San Francisco
# MAGIC http://insideairbnb.com/get-the-data/

# COMMAND ----------

import pandas as pd

url = "http://data.insideairbnb.com/united-states/ca/san-francisco/2022-12-04/data/listings.csv.gz"

df = pd.read_csv(url, compression='gzip', sep=",")

display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bereite den Datensatz für die Modellierung vor
# MAGIC Bevor ein Modell registriert werden kann, sollte dieses zunächst trainiert werden. Dazu müssen vorab Daten vorbereitet werden.<br>
# MAGIC In den nächsten Zellen werden folgende Verarbeitungsschritte durchgeführt:
# MAGIC - auswählen von bestimmten Features
# MAGIC - konvertieren der Features in numerische Datentypen
# MAGIC - erstellen von neuen Features aus bestehen Feautres
# MAGIC - entfernen von Zeilen mit *null values*

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


# COMMAND ----------

# konvertiere den Datentyp bestimmter Spalten
df_modelling.loc[:, 'price'] = df_modelling['price'].replace({'\$|,': ''}, regex=True).astype(float)
display(df_modelling)

# COMMAND ----------

# ein wenig feature-engineering
# einzelne review scores werden zu einem gesamten score summiert
df_modelling["review_scores_sum"] = (
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
                                  "review_scores_value"], axis=1)

# Antwort- und Akzeptanzrate müssen in ein numerisches Format konvertiert werden
df_modelling['host_response_rate'] = df_modelling['host_response_rate'].str.strip('%').astype(float) / 100
df_modelling['host_acceptance_rate'] = df_modelling['host_acceptance_rate'].str.strip('%').astype(float) / 100

display(df_modelling)


# COMMAND ----------

# Zeilen, die null values enthalten werden entfernt
df_modelling = df_modelling.dropna()
display(df_modelling)
dbutils.data.summarize(df_modelling)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Registrieren von Modellen<br>
# MAGIC Die folgenden Zellen zeigen, wie ein Modell mit Python über Code registriert werden kann. Prinzipiell ist auch eine Registrierung über die UI möglich.<br>
# MAGIC Bevor wir das Modell registrieren können, muss es in MLFlow getrackt werden.<br>
# MAGIC Für die Registrierung ist die Zuweisung eines **eindeutigen** Namens notwendig.

# COMMAND ----------

# erstelle ein Modell, welches in der modelregistry registriert werden soll
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Split in Trainings- und Testdatensatz
X_train, X_test, y_train, y_test = train_test_split(df_modelling.drop(["price"], axis=1), df_modelling["price"], random_state=42)

# Hyperparameter des Random Forest Modells
params =  {"n_estimators": 200,
          "max_depth": 10,
          "max_features": 1,
          "min_samples_split": 5,
          "min_samples_leaf": 5,
          "max_leaf_nodes": None,
}

# random state zur Reproduzierbarkeit des Ergebnisses
random_state = 42

# initialisiere und trainiere Modell
rf_model = RandomForestRegressor(**params, random_state = random_state)
rf_model.fit(X_train, y_train)


# COMMAND ----------

# erstelle Metriken, Signatur, ... , die mit dem Model getrackt werden sollen
from sklearn.metrics import r2_score
from mlflow.models.signature import infer_signature

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Fehlermetrik auf Trainings- und Testdaten
r2_score_test = r2_score(y_test, y_test_pred)
r2_score_train = r2_score(y_train, y_train_pred)

# Input Beispiel und signature
input_example = X_train.head(3)
signature = infer_signature(X_train, pd.DataFrame(y_train))


# COMMAND ----------

import mlflow
import mlflow.sklearn

# logge Modell in MLFlow
with mlflow.start_run(run_name="RF Model") as run:
    mlflow.sklearn.log_model(rf_model, "rf_model", input_example=input_example, signature=signature)
    mlflow.log_metric("r2_train", r2_score_train)
    mlflow.log_metric("r2_test", r2_score_test)
    mlflow.log_params(params)
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC Hier legen wir den eindeutigen Namen des Modells fest

# COMMAND ----------

# erstelle eindeutige Namen für die Registrierung
number_id = "01"
model_name = f"example-rf-model_{number_id}"

# registriere Modell
model_uri = f"runs:/{run_id}/rf_model"
model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Das registrierte Modell lässt sich nun auf der Seite **Models** in der UI finden.

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Client
# MAGIC Mithilfe des **MLflow Clients** können wir auf Details zu den registrierten Modellen zugreifen.<br>
# MAGIC Außerdem lassen sich Modelle z.B durch das Hinzufügen von Beschreibungen updaten.<br>
# MAGIC Später werden außerdem übergänge zwischen den Modellstages mithilfe des MLFlow Clients durchführen.

# COMMAND ----------

from mlflow.tracking.client import MlflowClient

# initialisiere client
client = MlflowClient()

# erhalte Details der Model Version
model_version_details = client.get_model_version(name=model_name, version=1)

model_version_details.status

# COMMAND ----------

# füge eine Beschreibung zum registrierten Modell hinzu
client.update_registered_model(
    name=model_details.name,
    description="Dieses Random Forest Modell schätzt den Preis von Airbnb Mietwohnungen basierend auf verschiedenen Featuregrößen."
)

# COMMAND ----------

# füge eine Beschreibung zur Modellversion hinzu
client.update_model_version(
    name=model_details.name,
    version=model_details.version,
    description="Dies ist die erste Version des Modells. Es wurde mit der Bibliothek scikit-learn erstellt."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Phasenübergänge zwischen Modell Stages
# MAGIC Ein registriertes Modell in MLflow kann folgende Stages einnehmen:
# MAGIC - **None**
# MAGIC - **Staging**
# MAGIC - **Production**
# MAGIC - **Archived**
# MAGIC 
# MAGIC Dabei entspricht jede dieser Phasen einer anderen Rolle im Modelllebenzzyklus.<br>
# MAGIC Mit dem MLFlow Client können Übergänge zwischen diesen Phasen durchgeführt werden.<br>
# MAGIC In der folgenden Zelle wird das registrierte Modell in die Phase **Production** übergeben.<br>

# COMMAND ----------

# ändere die Stage der Modellversion zu production
client.transition_model_version_stage(
    name=model_details.name,
    version=model_details.version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Mithilfe des Mlflow Clients lässt sich der Status der aktuellen Stage einsehen.

# COMMAND ----------

# Zugriff auf die Modelldetails
model_version_details = client.get_model_version(
    name=model_details.name,
    version=model_details.version,
  )
print(f"Das Modell befindet sich im Stage: '{model_version_details.current_stage}'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Versionierung von Modellen
# MAGIC Für ein registriertes Modell können weitere Versionen erstellt werden.<br>
# MAGIC Dies kann beispielsweise nötig sein, wenn das Modell auf neuen Daten trainiert wurde oder andere Parameter erhält.<br>
# MAGIC Wir loggen in der folgenden Zelle ein neu trainertes Modell und fügen es als neue Version zum bisher registrierten Modell hinzu.<br>
# MAGIC Wir ändern dabei nur die Hyperparameter des Modells. Um eine neue Version zum registrierten Modell hinzuzufügen müssen wir das Argument<br> **registered_model_name** in der Funktion **mlflow.sklearn.log_model()** setzen

# COMMAND ----------

# für die neue Modellversion wählen wir neue Hyperparameter
params_new = {"n_estimators": 100,
              "max_depth": 15,
              "max_features": 0.8,
              "min_samples_split": 4,
              "min_samples_leaf": 4,
              "max_leaf_nodes": None,
}

# initialisiere und trainiere Modell
rf_new = RandomForestRegressor(**params_new)
rf_new.fit(X_train, y_train)

# neuer run
with mlflow.start_run(run_name="RF Model") as run:
    mlflow.sklearn.log_model(
        sk_model=rf_new,
        artifact_path="rf-sklearn-model",
        registered_model_name=model_name,
        input_example=input_example,
        signature=signature
    )
    mlflow.log_metric("r2_test", r2_score(y_test, rf_new.predict(X_test)))
    mlflow.log_metric("r2_train", r2_score(y_train, rf_new.predict(X_train)))
    mlflow.log_params(params_new)
    run_id = run.info.run_id

# COMMAND ----------

# MAGIC %md
# MAGIC Wir setzen die neue Version nun in die Stage Production. <br>
# MAGIC Dabei ist es direkt möglich, das bereits im Production Stage stehende Modell zu archivieren. 

# COMMAND ----------

model_version_details = client.get_model_version(name=model_name, version=2)

# füge Beschreibung zu neuer Modellversion hinzu
client.update_model_version(
    name=model_name,
    version=model_version_details.version,
    description="Dies ist eine neue Version des Random Forest Modells"
)

# archiviere alte Version
client.transition_model_version_stage(
    name=model_name,
    version=model_version_details.version,
    stage="Production",
    archive_existing_versions=True 
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Laden aus der MLflow Registry
# MAGIC Mithilfe von **pyfunc** lassen sich Modellversionen aus der Modelregistry laden um beispielsweise Vorhersagen für Daten zu erstellen.

# COMMAND ----------

import mlflow.pyfunc

# Speicherort der Modellversion
model_version_2_uri = f"models:/{model_name}/2"

print(f"Lade registriertes Modell von URI: '{model_version_2_uri}'")
model_version_2 = mlflow.pyfunc.load_model(model_version_2_uri)

# wende Modell auf Testdaten an
model_version_2.predict(X_test)


# COMMAND ----------

# MAGIC %md
# MAGIC Dies ist auch für ältere Modellversionen möglich.

# COMMAND ----------

# Speicherort der Modellversion
model_version_1_uri = f"models:/{model_name}/1"

print(f"Lade registriertes Modell von URI: '{model_version_1_uri}'")
model_version_1 = mlflow.pyfunc.load_model(model_version_1_uri)

# wende geladenes Modell auf Testdaten an
model_version_1.predict(X_test)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Wie lassen sich registrierte Modelle löschen?
# MAGIC Das Löschen eines registrierten Modells ist ebenfalls mithilfe des MLflow Clients möglich.<br>
# MAGIC Wichtig hierbei ist, dass nur Modellversionen gelöscht werden können, die sich im **Archived** oder **None** Stage befinden.

# COMMAND ----------

# wir erhalten eine Fehlermeldung, da sich die Modellversion nicht im Archived Stage befindet
client.delete_model_version(
    name=model_name,
    version=2
)

# COMMAND ----------

# ändere die Stage zum Stage archived
client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Archived"
)

# COMMAND ----------

# im Archived Stage lässt sich das Modell aus der Registry löschen
client.delete_model_version(
    name=model_name,
    version=2
)

# COMMAND ----------

# löschen des registrierten Modells
client.delete_registered_model(model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC Auf der Seite **Models** können wir nun auch sehen, dass das registrierte Modell entfernt wurde.
