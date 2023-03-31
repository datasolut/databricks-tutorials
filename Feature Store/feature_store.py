# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC 
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://datasolut.com/wp-content/uploads/2020/01/logo-horizontal.png" alt="Databricks Learning" style="width: 300px">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## <img src="https://yt3.googleusercontent.com/ytc/AL5GRJWSDkfSdYxhsFknPhzJUWjYkHMEdYJHRO_AuCzlfQ=s900-c-k-c0x00ffffff-no-rj" alt="ds Logo Tiny" width="100" height="100"/> Feature Store
# MAGIC 
# MAGIC ### In diesem Video werden wir den Feature Store nutzen, um
# MAGIC - eine Feature-Tabelle im Feature Store zu erstellen
# MAGIC - ein Modell mit Daten aus einer Feature-Tabelle trainieren
# MAGIC - Feature-Tabellen upzudaten und neue Modell-Versionen zu erstellen
# MAGIC 
# MAGIC Schlüsselwörter: Feature Store, Delta Tables, Lookup, MLflow
# MAGIC 
# MAGIC Dokumentation Databricks Feature Store - https://docs.databricks.com/machine-learning/feature-store/index.html
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

df = df.select(*column_names)

# mit dem Schema können wir sehen, welche Datentypen angepasst werden müssen
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Konvertiere Spalte price zu numerischem Wert

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace

# konvertiere string Spalte price in einen numerischen Wert vom Typ double
df = df.withColumn("price",
 regexp_replace(col("price"), "\\$", "").cast("double"))

# Antwort- und Akzeptanzrate müssen in ein numerisches Format konvertiert werden
df = df.withColumn(
    "host_response_rate",
    (regexp_replace(col("host_response_rate"), "%", "").cast("float") / 100)
)

df = df.withColumn(
    "host_acceptance_rate",
    (regexp_replace(col("host_acceptance_rate"), "%", "").cast("float") / 100)
)


display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Erstellung von zusätzlicher Spalte mit **`id`**
# MAGIC 
# MAGIC Die Spalte **`id`** dient als "key" der Feature-Tabelle und kann zum Beispiel für Feature Lookups genutzt werden.

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

# Zeilen, die null values enthalten werden entfernt
df = df.dropna()

# hinzufuegen einer Index Spalte mit eindeutigem Zahlenwert
df = df.withColumn("id", monotonically_increasing_id())

# entfernen der price spalte
df_features = df.drop(col("price"))
display(df_features)
dbutils.data.summarize(df_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Erstellen eines Feature-Tables
# MAGIC 
# MAGIC Mit der **create_table** Methode können wir ein Feature-Tabelle erstellen.
# MAGIC 
# MAGIC Die Methode enthält folgende Argumente:
# MAGIC - **name** - Ein Name für die Tabelle in der Form **`<database_name>.<table_name>`**.
# MAGIC - **primary_keys** - Der Spaltenamen, welche als Keys dienen sollen (siehe obige Codezelle) 
# MAGIC - **df** - Daten, die in die Table eingesetzt werden.
# MAGIC - **schema** - Schema des Feature-Tables.
# MAGIC - **description** - Eine Beschreibung des Feature-Tables
# MAGIC - **partition_columns** - Spalten, für eine potentielle Partitionierung.
# MAGIC 
# MAGIC Zunächst werden wir einen Speichertort im Hivestore festlegen

# COMMAND ----------

# erstelle database
database_name = "database_feature_store_demo"
spark.sql(f"CREATE DATABASE IF NOT EXISTS {database_name}")
table_name = f"{database_name}.airbnb_feature_table"
print(table_name)

# COMMAND ----------

from databricks import feature_store

# nur nötig, falls die Tabelle schon exisitert. Ansonsten gibt der folgende Befehl eine Fehlermeldung
spark.sql(f"DROP TABLE IF EXISTS {table_name}")

# initialisiere Feature Store Cleint Objekt
fs = feature_store.FeatureStoreClient()
# vorher muss df im delta Format sein
fs.create_table(
    name=table_name,
    primary_keys=["id"],
    df=df_features,
    schema=df_features.schema,
    description="Features des Airbnb Datensatzes"
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Notiz:
# MAGIC Alternativ kann man auch eine leere Feature-Tabelle erstellen, in der man nur das Schema festlegt. Die Daten können dann mit **fs.write_table** in die Tabelle geschrieben werden.

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC #### Notiz:
# MAGIC Existiert schon eine Delta-Table mit Metastore, kann die Funktion **fs.register_table()** benutzt werden um sie um Feature-Store zu registrieren.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC #### Informationen einer Feature-Tabelle beziehen

# COMMAND ----------

# MAGIC %md
# MAGIC Mit **get_table()** kann über den Feature Store Client auf Details der Feature-Tabelle zugreifen, wie z.B. die Namen der Features oder die Beschreibung der Tabelle.

# COMMAND ----------

# Zugriff auf Feature Namen
fs.get_table(table_name).features

# COMMAND ----------

# Zugriff auf Table Beschreibung
fs.get_table(table_name).description

# COMMAND ----------

# MAGIC %md
# MAGIC ## Trainieren eines Modells
# MAGIC 
# MAGIC ##### Wichtig: Die Spalte **price** sollte nicht im Feature Store vorkommen!
# MAGIC Wir nehmen im kommenden Beispiel an, dass neben den bereits registrierten Features kurzfristig ein neues Feature hinzukommt, welches für sowohl für ein Modell-Training als auch für das Scoring nutzen wollen.<br>
# MAGIC In unserem Beispiel soll dieses Feature die Differenz des Review Scores aus dem letzten Monat darstellen.<br>
# MAGIC Das Feature ist fiktiv und wird aus Zufallszahlen bestimmt. Dies soll nur zu Demonstration dienen, also nicht verwirren lassen an dieser Stelle :)
# MAGIC 
# MAGIC 
# MAGIC Das neu hinzukommende Feature schreiben wir zusammen mit dem Preis und der ID in ein neues DataFrame.

# COMMAND ----------

# erstellen eines weiteren fikitiven Features
from pyspark.sql.functions import rand
df_new_features = df.select("id", "price", (rand() * 0.5-0.25).alias("score_diff_to_last_month")) 
display(df_new_features)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Erstellung des Trainingsdatensatzes mithilfe der Feature-Tabelle
# MAGIC 
# MAGIC Mithilfe der Funktion **fs.create_training_set()** können wir aus dem neuen DataFrame und der Feature-Tabelle den vollständigen Trainingsdatensatz erstellen. 
# MAGIC 
# MAGIC Dafür nutzen wir sogenannte Feature-Lookups. Mit dem Feature-Lookup werden Feature aus der Feature-Tabelle spezifiziert, welche wir in unseren Traningsdatensatz aufnehmen wollen. 
# MAGIC 
# MAGIC Dies steuern wir über das Argument **feature_names**. Falls dieser Wert nicht gesetzt wird, werden alle Feature ausgewählt.

# COMMAND ----------

from databricks.feature_store import FeatureLookup
from sklearn.model_selection import train_test_split

def load_modelling_data(table_name, lookup_key):

    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)] # erstelle Lookup

    # fs.create_training_set sucht nach Features in model_feature_lookups mit dem entsprechenden Schlüssel aus df_new_features
    training_data = fs.create_training_set(df_new_features, model_feature_lookups, label="price", exclude_columns="id")
    training_pandas = training_data.load_df().toPandas()

    # erstelle trainings und test datasets
    X = training_pandas.drop("price", axis=1) # entferne den Preis aus dem Trainingsdatensatz
    y = training_pandas["price"] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, training_data

X_train, X_test, y_train, y_test, training_data = load_modelling_data(table_name, "id")
display(X_train)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Training eines Gradient Boosting Modells
# MAGIC 
# MAGIC Trainiere ein Gradient Boosting Modell und tracke es (nur das Modell!) mithilfe des Feature Stores.<br>
# MAGIC Einzelne Metriken und Parameter loggen wir ebenfalls innerhalb des runs.
# MAGIC 
# MAGIC Das Input-Argument **registered_model_name** in der Funktion **fs.log_model()** wird das Modell unter dem angegbenen Namen regisitrieren oder überschreiben, je nachdem, ob das Modell schon registriert ist.

# COMMAND ----------

import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


def train_and_register_model(X_train, X_test, y_train, y_test, training_data, fs, params):

    ## trainiere und tracke Modell
    with mlflow.start_run(run_name="Gradient Boosting Model") as run:

        # fitte GB Modell und erstelle Vorhersagen auf Trainings- und Testdaten
        gb_model = GradientBoostingRegressor(**params, random_state=42)
        gb_model.fit(X_train, y_train)
        y_pred_test = gb_model.predict(X_test)
        y_pred_train = gb_model.predict(X_train)
        
        # logge Parameter
        mlflow.log_params(params)

        # logge Metriken
        mlflow.log_metric("r2_score_train", r2_score(y_train, y_pred_train))
        mlflow.log_metric("r2_score_test", r2_score(y_test, y_pred_test))

        # logge und registriere Modell
        fs.log_model(
            model=gb_model,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn, # wird als sklearn-Modell geloggt
            training_set=training_data,
            registered_model_name="feature_store_airbnb_model_example",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )



# COMMAND ----------

# setze Parameter für das GB Modell und starte Modell Training
params= {"n_estimators": 100,
         "max_depth": 4,
         "max_features": 0.8,
         "min_samples_split": 4,
         "min_samples_leaf": 4,
}
train_and_register_model(X_train, X_test, y_train, y_test, training_data, fs, params)

# COMMAND ----------

# MAGIC %md
# MAGIC Das **feature_store_model** wird damit auch direkt in der model registry registriert. Außerdem wird es auch in der Feature Store UI angezeigt, in der auch zu sehen ist, welche Feature in das Modell-Training eingegangen sind.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store Batch Scoring
# MAGIC 
# MAGIC Wir werden nun ein Modell, welches mit dem Feature Store geloggt worden ist verwenden, um ein Scoring für neue Daten durchzuführen. Als Input benötigen wir nur die Key-Spalte und das neue kurzfristige Feature. Alles weiteren Daten werden automatisch durch einen Lookup, entsprechendem der Key-Spalte, verwendet. 
# MAGIC 
# MAGIC Die **fs.score_batch()**-Funktion eignet sich sehr gut um ein Modell auf Trainings- oder historische Daten anzuwenden, um die Vorhersagequalität auf diesen Daten zu prüfen.

# COMMAND ----------

# der Einfachheit halber verwenden wir den gleichen Datensatz wie oben und erstellen keine neuen Daten
df_batch_input = df_new_features.drop("price") 
model_uri = "models:/feature_store_airbnb_model_example/1"
df_predictions = fs.score_batch(model_uri, df_batch_input, result_type="double") # die Feature, welche mit dem Modell im Feature Store geloggt sind, werden automatisch genutzt

display(df_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature-Tabelle updaten
# MAGIC 
# MAGIC Zum Abschluss werden wir die oben erstellte Feature-Tabelle updaten, indem wir ein neuen Feature aus den bisherigen Spalten erstellen und einige der Spalten löschen.

# COMMAND ----------

from pyspark.sql.functions import lit, expr

## nur numerische Werte und zusammengefasste Werte der review scores
review_columns = ["review_scores_accuracy", "review_scores_cleanliness", "review_scores_checkin", 
                 "review_scores_communication", "review_scores_location", "review_scores_value"]


df_features_avg_rev = (df_features.drop("price")                      
                                  .withColumn("average_review_score", expr("+".join(review_columns)) / lit(len(review_columns)))
                                  .drop(*review_columns)
                      )
             
display(df_features_avg_rev)

# COMMAND ----------

# MAGIC %md 
# MAGIC Um die Feature-Tabelle upzudaten nutzen wir **overwrite**, innerhalb der **fs.write_table()**-Funktion.

# COMMAND ----------

fs.write_table(
    name=table_name,
    df=df_features_avg_rev,
    mode="overwrite"
)

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Wie wirkt sich das Update auf die Darstellung in der UI aus?

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC In der Feature-Store UI sehen wir:
# MAGIC - Eine neue Spalte wurde hinzugefügt (die Spalte, welche durch das Zusammenfügen von review-columns enstanden ist)
# MAGIC - Gelöschte Spalten sind trotzdem sichtbar. Die gelöschten Werte wurden mit **`null`** ersetzt (beim Einlesen der Features beachten!)

# COMMAND ----------

# MAGIC %md
# MAGIC Mithilfe von **fs.read_table()** kann man die aktuellste Version der Feature-Tabelle einlesen. Um eine vorherige Version einzulesen, kann man zusätzlich **as_of_delta_timestamp** als Timestamp oder String als Argument übergeben.
# MAGIC 
# MAGIC Wichtig: Die gelöschten Spalten werden mit **null** ersetzt.

# COMMAND ----------

display(fs.read_table(name=table_name))

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ### Trainiere das Modell erneut mit dem geupdateten Datensatz
# MAGIC 
# MAGIC Erstelle wieder einen Trainingsdatensatz mithilfe von **`fs.create_training_set()`** und der Key-Spalte für den Lookup.

# COMMAND ----------

def load_data(table_name, lookup_key):

    model_feature_lookups = [FeatureLookup(table_name=table_name, lookup_key=lookup_key)]

    # fs.create_training_set sucht nach Features in model_feature_lookups mit dem entsprechenden Schlüssel aus df_new_features
    training_data = fs.create_training_set(df_new_features, model_feature_lookups, label="price", exclude_columns="id")
    training_pandas = training_data.load_df().drop(*review_columns).toPandas()  # entfernen die Spalten mit null values

    # erstelle Trainings- und Testdaten
    X = training_pandas.drop("price", axis=1)
    y = training_pandas["price"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, training_data

X_train, X_test, y_train, y_test, training_data = load_data(table_name, "id")
display(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC Innerhalb der Funktion **fs.log_model()** werden wir für das Argument **registered_model_name** den gleichen Namen wählen wie beim obigen Aufruf der Funktion, um eine neue Version des Modells zu registieren.<br>
# MAGIC Wir hätten auch direkt die Funktion von oben nehmen können. Sie ist hier zur Übersicht nochmal dargestellt.

# COMMAND ----------

def train_and_register_model(X_train, X_test, y_train, y_test, training_set, fs, params):
    ## fit and log model
    with mlflow.start_run(run_name="New Gradient Boosting Model") as run:

        # fitte neues GB Model
        gb_new = GradientBoostingRegressor(**params, random_state=42)
        gb_new.fit(X_train, y_train)
        y_pred = gb_new.predict(X_test)
        y_pred_train = gb_new.predict(X_train)
        y_pred_test = gb_new.predict(X_test)

        # logge Parameter
        mlflow.log_params(params)
        
        # logge Fehlermetriken
        mlflow.log_metric("r2_score_train", r2_score(y_train, y_pred_train))
        mlflow.log_metric("r2_score_test", r2_score(y_test, y_pred_test))

        # logge und registriere Modell
        fs.log_model(
            model=gb_new,
            artifact_path="feature-store-model",
            flavor=mlflow.sklearn, # wird als sklearn-Modell geloggt
            training_set=training_data,
            registered_model_name="feature_store_airbnb_model_example",
            input_example=X_train[:5],
            signature=infer_signature(X_train, y_train)
        )


# COMMAND ----------

# setze neue Parameter für das GB Modell und starte Modell Training
new_params= {"n_estimators": 150,
             "max_depth": 5,
             "max_features": 0.9,
             "min_samples_split": 4,
             "min_samples_leaf": 4,
}
train_and_register_model(X_train, X_test, y_train, y_test, training_data, fs, new_params)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Was hat sich in der UI geändert?
# MAGIC 
# MAGIC In der UI sehen wir, dass Modell-Version 2 das neu erstellte Feature nutzt.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Store Batch Scoring
# MAGIC 
# MAGIC Wir nutzen nun Batch Scoring für Version 2 unseres Modells.

# COMMAND ----------

## der Einfachheit halber verwenden wir den gleichen Datensatz wie oben und erstellen keine neuen Daten

batch_input_df = df_new_features.drop("price") # entferne Zielvariable aus dem Datensatz
model_uri_2 = "models:/feature_store_airbnb_model_example/2" # Wichtig: Wir nutzen Version 2!!!!!

df_predictions = fs.score_batch(model_uri_2, batch_input_df, result_type="double")
display(df_predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Löschen von Feature-Tables
# MAGIC Eine Feature-Tabelle kann sowohl über die UI als auch programmatisch gelöscht werden.<br>
# MAGIC Wird sie über die UI gelöscht muss beachtet werden, dass die zugehörige Delta-Table **nicht** gelöscht wird.<br>
# MAGIC Mit der Funktion **fs.drop_table()** können wir sie auch mittels Code löschen. Hierbei wird auch die zugehörige Delta-Table entfernt.

# COMMAND ----------

# lösche Feature-Tabelle
fs.drop_table(name = table_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Besuchen Sie auch gerne unsere(n)
# MAGIC - Website: www.datasolut.com
# MAGIC - Youtube-Kanal: https://www.youtube.com/@datasolut6213
