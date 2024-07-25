# Databricks notebook source
# MAGIC %md
# MAGIC <img src ='files/demos/pypsark-demos/datasolut_logo.jpeg' alt="html image" align="center">

# COMMAND ----------

<img src ='https://github.com/datasolut/databricks-tutorials/blob/main/Spark%20Streaming/Databricks-Auto-Loader-Demo/Streaming-Ablauf.png' alt="Streaming Ablauf" text-align="center" width="500px">

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Ingest mit dem Autoloader

# COMMAND ----------

from pyspark.sql import DataFrame

def get_stream_df_from_s3(s3_bucket_path: str, schema: str = None) -> DataFrame:
    """
    Reads streaming data from an S3 bucket and returns it as a DataFrame.

    Parameters:
    s3_bucket_path (str): The path to the S3 bucket containing the streaming data.
    schema (str, optional): The schema of the data. If None, a default schema is used.

    Returns:
    DataFrame: A Spark DataFrame containing the streaming data.
    """
    if schema is None:
        schema = "transaction_id STRING, customer_id STRING, timestamp TIMESTAMP, purchased_products ARRAY<STRING>"

    return (spark.readStream
            .format("cloudFiles")
            .option("cloudFiles.format", "json")
            .schema(schema)
            .load(s3_bucket_path))
    


# COMMAND ----------

input_s3_bucket_path = "s3://demo-autoloader/dummy-data"
df = get_stream_df_from_s3(input_s3_bucket_path)

# COMMAND ----------

df.show()

# COMMAND ----------

def join_with_customer_table(df: DataFrame) -> DataFrame:
    """
    Joins the input DataFrame with the customer table.

    Parameters:
    df (DataFrame): The input DataFrame that needs to be joined with the customer table.

    Returns:
    DataFrame: A DataFrame resulting from the left join with the customer table.
    """
    # Lesen der Kunden-Dimensionstabelle aus der Datenbank
    customer_df = spark.read.table("dbdemos.retail.dim_customers")
    
    # Durchführung eines Left Joins zwischen dem Eingabe-DataFrame und der Kunden-Dimensionstabelle
    return df.join(customer_df, on="customer_id", how="left")

# COMMAND ----------

import pyspark.functions as F

def explode_and_drop_column(df: DataFrame, explode_column: str, new_column: str) -> DataFrame:
    """
    Explodes an array column into multiple rows and drops the original column.

    Args:
        df (DataFrame): The input DataFrame containing the column to be exploded.
        explode_column (str): The name of the column to explode.
        new_column (str): The name of the new column that will contain the exploded values.

    Returns:
        DataFrame: A DataFrame with the exploded column and the original column dropped.
    """
    return (df
            .withColumn(new_column, F.explode(F.col(explode_column)))
            .drop(explode_column))

# COMMAND ----------

from pyspark.sql.streaming import StreamingQuery

def write_stream_to_delta_table(df: DataFrame, table_name: str, checkpoint_path: str) -> StreamingQuery:
    """
    Writes a streaming DataFrame to a Delta table.

    args:
        df (DataFrame): The input streaming DataFrame to be written.
        table_name (str): The name of the target Delta table.
        checkpoint_path (str): The path to the checkpoint directory.

    Returns:
        StreamingQuery: A object that can be used to monitor the progress of the write operation.
    """
    return (df.writeStream
            .format("delta")
            .outputMode("append")
            .option("checkpointLocation", checkpoint_path)
            .toTable(table_name))

# COMMAND ----------

# S3-Bucket und Pfade definieren
input_s3_bucket_path = "s3://demo-autoloader/dummy-data"
output_delta_table_name = "dbdemos.retail.fkt_transactions"
output_checkpoint_path = "dbfs:/FileStore/demos/pypsark-demos/streaming"

# Extract
df = get_stream_df_from_s3(input_s3_bucket_path)

# Transform
df = (df
      .transform(join_with_customer_table)
      .transform(explode_and_drop_column, "purchased_products", "purchased_product"))

display(df)

# COMMAND ----------

# Load
write_stream_to_delta_table(df, output_delta_table_name, output_checkpoint_path)
