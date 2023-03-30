# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://datasolut.com/wp-content/uploads/2020/01/logo-horizontal.png" alt="Databricks Learning" style="width: 300px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # <img src="https://yt3.googleusercontent.com/ytc/AL5GRJWSDkfSdYxhsFknPhzJUWjYkHMEdYJHRO_AuCzlfQ=s900-c-k-c0x00ffffff-no-rj" alt="ds Logo Tiny" width="100" height="100"/> AutoML
# MAGIC 
# MAGIC ## In diesem Video werden wir
# MAGIC - AutoML nutzen, um den typischen ML-Prozess zu automatisieren
# MAGIC - die von AutoML trainierten Modelle und Notebooks untersuchen
# MAGIC 
# MAGIC 
# MAGIC Schlüsselwörter: AutoML, Spark, Decision Trees, Random Forests, XGBoost

# COMMAND ----------

# MAGIC %md
# MAGIC ### Funktionsweise von AutoML
# MAGIC AutoML trainiert verschiedene Modelle, optimiert (mithilfe von HyperOpt) und evaluiert anschließend die trainierten Modelle.
# MAGIC 
# MAGIC Sowohl über UI als auch programmatisch nutzbar.
# MAGIC 
# MAGIC Erstellt automatisch Notebooks für die verschiedenen Modelle, welche trainiert wurden.
# MAGIC 
# MAGIC Dokumenatation: https://docs.databricks.com/machine-learning/automl/index.html

# COMMAND ----------

# MAGIC %md
# MAGIC  
# MAGIC ### Lade Datensatz Airbnb - San Francisco
# MAGIC http://insideairbnb.com/get-the-data/

# COMMAND ----------

from pyspark import SparkFiles

url = "http://data.insideairbnb.com/united-states/ca/san-francisco/2022-12-04/data/listings.csv.gz"
sc.addFile(url)

path  = SparkFiles.get('listings.csv.gz')

df = (spark.read.option("header", "true").option("multiLine", "true")
                .option("escape", "\"")
                .option("inferSchema", "true")
                .csv("file://" + path, sep = ",")
)

display(df)

print(f"Anzahl der Zeilen: {df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Vereinfachung des Datensatzes
# MAGIC In diesem Fall wollen wir nur die Spalten:
# MAGIC - **`bedrooms`**
# MAGIC - **`beds`**
# MAGIC - **`latitude`**
# MAGIC - **`longitude`**
# MAGIC - **`host_time_response`** (kategorisches Feature!)
# MAGIC 
# MAGIC nutzen, um **`price`** vorherzusagen.

# COMMAND ----------

column_names = ["bedrooms", "beds", "latitude", "longitude", "host_response_time", "price"]

# Erstelle Teildatensatz, der für die Modellierung verwendet wird
df_modelling = df.select(column_names)

display(df_modelling)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verändere den Datentyp der Spalte *price* zu double
# MAGIC Da wir ein Regressions-Modell aufstellen werden, müssen wir die Spalte **`price`** zu einem numerischen Wert konvertieren.
# MAGIC 
# MAGIC Diese würden wir gerne zu einem numerischen Datentypen wie z.B. double umwandeln.
# MAGIC 
# MAGIC Dazu nutzen wir die Funktion **`cast()`**.

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace

# konvertiere string Spalte price in einen numerischen Wert vom Typ double, entferne $-Zeichen aus dem String
df_modelling = df_modelling.withColumn("price", regexp_replace(col("price"), "\\$", "").cast("double"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC Als nächstes teilen wir die Daten in einen Trainings- und Testdatensatz auf.<br>

# COMMAND ----------

df_train, df_test = df_modelling.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md <i18n value="1b5c8a94-3ac2-4977-bfe4-51a97d83ebd9"/>
# MAGIC 
# MAGIC Um den Preis vorherzusagen, stellen wir nun ein Regressions-Modell auf. Um in diesem Fall AutoML zu nutzen, benötigen wie automl.regress().
# MAGIC 
# MAGIC Benötige Parameter für automl.regress():
# MAGIC * **`dataset`** - Spark oder pandas DataFrame, welches Trainingsdaten und Zielvariable enthält. Falls ein Spark DataFrame genutzt wird, es zu einem pandas DataFrame umgewandelt.
# MAGIC * **`target_col`** - Spaltenname der Zielvariable.
# MAGIC 
# MAGIC Optionale Parameter für automl.regress():
# MAGIC * **`primary_metric`** - Primäre Metrik um das beste Modell auszuwählen. In jedem Trainingsversuch werden verschiedene Metriken berechnet, aber nur die primäre Metrik verwendet, um das beste Modell zu berstimmen. **`r2`** (default, R squared), **`mse`** (mean squared error), **`rmse`** (root mean squared error), **`mae`** (mean absolute error) für Regressionsprobleme.
# MAGIC * **`timeout_minutes`** - Die maximale Zeit, welche AutoML nutzt, um einen Trainignsversuch durchzuführen. Wenn **`timeout_minutes=None`** wird den Trainingsversuch ohne Zeitbeschränkung laufen lassen.

# COMMAND ----------

from databricks import automl

summary = automl.regress(df_train, target_col="price", primary_metric="rmse", timeout_minutes=5)

# COMMAND ----------

# MAGIC %md <i18n value="57d884c6-2099-4f34-b840-a4e873308ffe"/>
# MAGIC 
# MAGIC 
# MAGIC Nachdem die letzte Zeile ausgeführt worden ist, werden **mehrere** Notebooks und **ein** MLflow Experiment generiert:
# MAGIC 
# MAGIC Die Notebooks:
# MAGIC * **`Data exploration notebook`** - Hier werden diverse Statistiken zu den Trainingsdaten gezeigt und visualisiert.
# MAGIC * **`Trial notebooks`** - Zeigt den Code für den Trainingsdurchlauf mit dem besten Score, um das Modelltraining zu reproduzieren.
# MAGIC 
# MAGIC Das MLflow Experiment:
# MAGIC * **`MLflow experiment`** - enthält einige Informationen, wie z.B. den root artifact Speicherort, Experiment ID und Experiment tags. Die Liste der 
# MAGIC Durchgänge enthält Informationen zu jedem einzelnen Durchlauf, sowie ein entsprechendes Notebook und den Speicherort, Trainingsparameter und verschiedene Metriken.

# COMMAND ----------

# zeige Informationen zum besten Durchlauf
print(summary.best_trial)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lade das beste Modell unter den getrackten Modellen
# MAGIC Wir laden nun das beste Modell aus, welches in diesem Run von AutoML erstellt worden ist. Damit führen wir noch eine prediction auf den Testdatensatz durch und evaluieren diese.<br>

# COMMAND ----------

# Lade das beste Modell unter den getrackten Modellen
import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = df_test.withColumn("prediction", predict(*df_test.drop("price").columns))
display(pred_df)
