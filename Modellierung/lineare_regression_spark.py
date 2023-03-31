# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://datasolut.com/wp-content/uploads/2020/01/logo-horizontal.png" alt="Databricks Learning" style="width: 300px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC # <img src="https://yt3.googleusercontent.com/ytc/AL5GRJWSDkfSdYxhsFknPhzJUWjYkHMEdYJHRO_AuCzlfQ=s900-c-k-c0x00ffffff-no-rj" alt="ds Logo Tiny" width="100" height="100"/> Erstellen eines Machine-Learning Modells
# MAGIC 
# MAGIC ## In diesem Video werden wir
# MAGIC - einen Datensatz aufbereiten
# MAGIC - ein einfaches lineares Regressionsmodell mithilfe von SparkML trainieren
# MAGIC 
# MAGIC 
# MAGIC Schlüsselwörter: Spark, Lineare Regression, Imputer, Transformer, Vector Assembler, Evaluator<br>
# MAGIC <br>
# MAGIC Dokumentation: https://spark.apache.org/docs/latest/ml-guide
# MAGIC 
# MAGIC Besuchen Sie auch gerne die [datasolut-Website](https://datasolut.com/) und unseren [Youtube-Kanal](https://www.youtube.com/@datasolut6213)!

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
# MAGIC 
# MAGIC ### Selektiere relevante Spalten
# MAGIC Zur Vereinfachung filtern wir die Spalten
# MAGIC - **`bedrooms`**
# MAGIC - **`price`**
# MAGIC 
# MAGIC Mithilfe der Spalte *bedrooms* werden wir versuchen die Spalte **`price`** vorherzusagen.

# COMMAND ----------

# Erstelle Teildatensatz, der für die Modellierung verwendet wird
df_modelling = df.select("bedrooms", "price")

display(df_modelling)

# COMMAND ----------

display(df_modelling.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Weiterverarbeitung der Daten
# MAGIC Die Spalte *price* ist im Datentyp String gegeben. Diese müssen wir in einen numerischen Typen umwandeln.

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace

# konvertiere string Spalte price in einen numerischen Wert vom Typ double
df_modelling = df_modelling.withColumn("price",
 regexp_replace(col("price"), "\\$", "").cast("double"))

# COMMAND ----------

display(df_modelling.summary())

# COMMAND ----------

# alternative Methode zum Betrachten einiger Statistiken
dbutils.data.summarize(df_modelling)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### Fülle fehlende Werten mithilfe des Imputers
# MAGIC Sowohl die Spalte *bedrooms*, als auch die Spalte *price* enthält fehlende Werte (*null values*).
# MAGIC Prinzipiell gibt es verschiedene Möglichkeiten mit *null values* umzugehen. <br>
# MAGIC Wir benutzen einen sogenannten **Imputer** um *null values* mit dem jeweiligen Mittelwert der Spalte zu ersetzen.<br>
# MAGIC Die Verwendung des Imputers setzt voraus, dass alle Werte vom Typ *double* sind.

# COMMAND ----------

from pyspark.ml.feature import Imputer

# konvertiere Spalte vom Typ Integer zum Typ Double
df_modelling = df_modelling.withColumn("bedrooms", col("bedrooms").cast("double"))

# initialisiere imputer Objekt
imputer = Imputer(strategy="mean",
                  inputCols=["bedrooms", "price"],
                  outputCols=["bedrooms", "price"])

# fitte imputer und transformiere den Datensatz
imputer_model = imputer.fit(df_modelling)
df_modelling = imputer_model.transform(df_modelling)

dbutils.data.summarize(df_modelling)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Train / Test Split
# MAGIC Nun ist der Datensatz befreit von *null* Werten.<br>
# MAGIC Als nächstes teilen wir die Daten in einen Trainings- und Testdatensatz auf.<br>
# MAGIC Der Trainingsdatensatz wird zum Anpassen des Modells verwendet.<br>
# MAGIC Auf dem Testdatensatz evaluieren wir die Vorhersagegüte des Modells.

# COMMAND ----------

df_train, df_test = df_modelling.randomSplit([.8, .2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Trainieren eines linearen Modells
# MAGIC Als nächstes benutzen wir unseren Trainingsdatensatz um ein lineares Regressionsmodell zu trainieren.<br>
# MAGIC Feature Spalten müssen dafür zunächst mit dem **VectorAssembler** transformiert werden.<br>
# MAGIC Dieser fasst Spalten zu einer Vektor-Spalte zusammen.<br>
# MAGIC Wir erhalten so das notwendige Format um das lineare Modell zu fitten.

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# VectorAssembler zum Transformieren der Features
vec_assembler = VectorAssembler(inputCols = ["bedrooms"], outputCol="features")
df_train = vec_assembler.transform(df_train)
df_test = vec_assembler.transform(df_test)

display(df_train)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression

# initialisiere und trainiere lineares Modell
linear_regression = LinearRegression(featuresCol = "features", labelCol= "price")
linear_model = linear_regression.fit(df_train)


# COMMAND ----------

# bilde dataframe mit Vorhersagen
df_pred_test = linear_model.transform(df_test)
df_pred_train = linear_model.transform(df_train)

display(df_pred_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluator zur Validierung der Vorhersagegüte
# MAGIC Mithilfe sogenannter Evaluator können wir nun die Fehlermetriken auf den Trainings- oder Testdaten berechnen und so die Vorhersage evaluieren.<br>
# MAGIC Wir verwenden das R2-Bestimmtheitsmaß als Fehlermetrik.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

# Initialisieren eines evaluator Objektes 
evaluator = RegressionEvaluator(predictionCol = "prediction", labelCol= "price", metricName = "r2")
r2_test = evaluator.evaluate(df_pred_test)
r2_train = evaluator.evaluate(df_pred_train)

print(f"R2 Testdatensatz: {r2_test}")
print(f"R2 Trainingsdatensatz: {r2_train}")

