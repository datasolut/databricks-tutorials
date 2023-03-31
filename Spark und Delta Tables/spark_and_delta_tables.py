# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://datasolut.com/wp-content/uploads/2020/01/logo-horizontal.png" alt="Databricks Learning" style="width: 300px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## <img src="https://yt3.googleusercontent.com/ytc/AL5GRJWSDkfSdYxhsFknPhzJUWjYkHMEdYJHRO_AuCzlfQ=s900-c-k-c0x00ffffff-no-rj" alt="ds Logo Tiny" width="100" height="100"/> ETL- Prozess mithilfe von Spark und Delta Tables
# MAGIC 
# MAGIC ### In diesem Video werden wir einen ETL-Prozess durchführen, indem wir
# MAGIC - einen Beispieldatensatz laden
# MAGIC - den Datensatz mithilfe von Apache Spark bearbeiten
# MAGIC - den bearbeiteten Datensatz als Delta Table abspeichern
# MAGIC 
# MAGIC ### ETL: Extract - Transform - Load
# MAGIC 
# MAGIC 
# MAGIC Schlüsselwörter: Spark, Delta
# MAGIC 
# MAGIC Dokumentation Apache Spark - https://spark.apache.org/docs/latest/
# MAGIC 
# MAGIC Besuchen Sie auch gerne die [datasolut-Website](https://datasolut.com/) und unseren [Youtube-Kanal](https://www.youtube.com/@datasolut6213)!

# COMMAND ----------

# MAGIC %md
# MAGIC ### Lade Datensatz Airbnb - San Francisco 
# MAGIC http://insideairbnb.com/get-the-data/

# COMMAND ----------

from pyspark import SparkFiles

url = "http://data.insideairbnb.com/united-states/ca/san-francisco/2022-12-04/data/listings.csv.gz"
spark.sparkContext.addFile(url)    
path  = SparkFiles.get('listings.csv.gz')

 
airbnb_df = (spark.read.option("header", "true")
                       .option("inferSchema", "true")
                       .option("multiLine", "true")
                       .option("escape", "\"")
                       .csv("file://" + path, sep = ",")
)

display(airbnb_df)

print(f"Anzahl an Zeilen: {airbnb_df.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Überblick zu den Metadaten verschaffen

# COMMAND ----------

airbnb_df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Wähle spezielle Spalten aus, welche näher betrachtet werden sollen
# MAGIC In diesemFall wollen wir für weitere Untersuchungen nur die Spalten:
# MAGIC - **`id`**
# MAGIC - **`bedrooms`**
# MAGIC - **`minimum_nights`**
# MAGIC - **`availability_365`**
# MAGIC - **`price`**
# MAGIC 
# MAGIC nutzen.

# COMMAND ----------

columns = [
    "id",
    "bedrooms",
    "minimum_nights",
    "availability_365",
    "price"
]

airbnb_refined_df = airbnb_df.select(columns)

display(airbnb_refined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entferne unbrauchbare Daten
# MAGIC 
# MAGIC Wir haben bemerkt, dass manche Zeilen nicht im Interesse unserer Datenverarbeitung liegen. Dazu gehören z.B. Werte für die
# MAGIC - **`minimum_nights < 14`**
# MAGIC - **`availability_365 > 0`** 
# MAGIC 
# MAGIC Um die Daten nach den gewünschten Werten zu filtern, nutzen wir die Filter-Abfrage.

# COMMAND ----------

from pyspark.sql.functions import col

# Zwei verschiedene Moeglichkeiten, welche sich in der Syntax unterscheiden
airbnb_refined_df = airbnb_refined_df.filter("minimum_nights < 14")

airbnb_refined_df = airbnb_refined_df.filter("availability_365 > 0")

display(airbnb_refined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Entferne Zeilen mit fehlenden Werten (Null-Values)
# MAGIC 
# MAGIC Wir haben bemerkt, dass einige Einträge in den Spalten fehlen. 
# MAGIC 
# MAGIC Diese entfernen wir mit der **`dropna()`**-Funktion.

# COMMAND ----------

airbnb_refined_df = airbnb_refined_df.dropna()

display(airbnb_refined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verändere den Datentyp der Spalte *price* zu double
# MAGIC Wir haben bemerkt, dass die Werte in **`price`** den Datentyp String besitzen. 
# MAGIC 
# MAGIC Diese würden wir gerne zu einem numerischen Datentypen wie z.B. double umwandeln.
# MAGIC 
# MAGIC Dazu nutzen wir die Funktion **`cast()`**.

# COMMAND ----------

from pyspark.sql.functions import regexp_replace

airbnb_refined_df = airbnb_refined_df.withColumn("price", regexp_replace("price", "[$]", "").cast("double"))

display(airbnb_refined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Speichere den Datensatz als Delta Table ab

# COMMAND ----------

# MAGIC %sql  
# MAGIC CREATE SCHEMA IF NOT EXISTS airbnb;

# COMMAND ----------

database_name = "airbnb"
table_name = "airbnb_refined"

airbnb_refined_df.write.format("delta").mode("overwrite").saveAsTable(f"hive_metastore.{database_name}.{table_name}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Entferne ganze Spalten
# MAGIC 
# MAGIC Sollten einige Spalten nicht mehr gebraucht werden, können diese auch vollständig entfernt werden. 
# MAGIC 
# MAGIC Dazu nutzen wir die **`drop()`**-Funktion.
# MAGIC 
# MAGIC Wir entfernen hier die Spalten
# MAGIC - **`availability_365`**
# MAGIC - **`minimum_nights`**

# COMMAND ----------

airbnb_refined_df = airbnb_refined_df.drop("availability_365", "minimum_nights")

display(airbnb_refined_df)

# COMMAND ----------

airbnb_refined_df.write.format("delta").mode("overwrite").saveAsTable(f"hive_metastore.{database_name}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Betrachte die History des Datensatzes
# MAGIC 
# MAGIC Mit der History-Abfrage ist es möglich den Verlauf seiner gespeicherten Delta Tables zu rekonstruieren.

# COMMAND ----------

# MAGIC %sql
# MAGIC DESCRIBE HISTORY airbnb.airbnb_refined

# COMMAND ----------

# MAGIC %md
# MAGIC ### Ältere Versionen wiederherstellen
# MAGIC 
# MAGIC Sollten ältere Versionen benötigt werden, ist dies mit der TimeTravel-Funktion der Delta Tables möglich.
# MAGIC 
# MAGIC Hier können wir spark.read mit der Option **`versionAsOf`** oder **`timestampAsOf`** nutzen.
# MAGIC 
# MAGIC In diesem Beispiel laden die erste Version 0 des Datensatzes in unser Notebook.

# COMMAND ----------

airbnb_refined_df = spark.read.format("delta").option("versionAsOf", 0).table(f"hive_metastore.{database_name}.{table_name}")

display(airbnb_refined_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Beispiel-Abfrage an den neuen Datensatz

# COMMAND ----------

from pyspark.sql.functions import avg

avg_two_bedrooms = airbnb_refined_df.filter("bedrooms == 2").agg(avg("price")).collect()[0][0]
avg_three_bedrooms = airbnb_refined_df.filter("bedrooms == 3").agg(avg("price")).collect()[0][0]

print(f"Durchschnittspreis für 2 Schlafzimmer: {avg_two_bedrooms}")
print(f"Durchschnittspreis für 3 Schlafzimmer: {avg_three_bedrooms}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Kurzer Clean-Up
# MAGIC 
# MAGIC Lösche alle Daten aus dem Hivestore

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP SCHEMA airbnb CASCADE;
