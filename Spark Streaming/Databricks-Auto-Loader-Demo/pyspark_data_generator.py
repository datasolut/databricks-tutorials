# Databricks notebook source
# MAGIC %md
# MAGIC <img src ='https://github.com/datasolut/databricks-tutorials/blob/main/images/datasolut_logo_quer.png?raw=true' alt="html image" align="center">

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Data Generator
# MAGIC
# MAGIC In diesem Notebook werden die statischen und Stream-Daten für die Autoloader Demo erstellt.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Erstellung der statischen Kundendimension

# COMMAND ----------

# MAGIC %pip install faker

# COMMAND ----------

from faker import Faker

def generate_fake_data(num_records=100):
    """Generate a list of fake customer data."""
    fake = Faker('en_US') 
    data = []
    for _ in range(num_records):
        data.append((
            str(fake.uuid4()),
            fake.first_name(),
            fake.last_name(),
            fake.street_address(),
            fake.postcode(),
            fake.city(),
            fake.date_of_birth(minimum_age=18, maximum_age=80)
        ))
    return data

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DateType

def define_schema():
    """Define and return the schema for the customer DataFrame."""
    return StructType([
        StructField("customer_id", StringType(), False),
        StructField("firstname", StringType(), False),
        StructField("lastname", StringType(), False),
        StructField("street", StringType(), False),
        StructField("postalcode", StringType(), False),
        StructField("city", StringType(), False),
        StructField("birthday", DateType(), False)
    ])

# COMMAND ----------

def save_to_uniy_catalog(data, catalog_name, schema_name, table_name):
    """Create and return a DataFrame with the given data and schema."""
    schema = define_schema()
    df = spark.createDataFrame(data, schema)
    df.write.mode("overwrite").saveAsTable(f"{catalog_name}.{schema_name}.{table_name}")


# COMMAND ----------

def create_customer_table(catalog_name, schema_name, table_name, n_records=100):
    """
    Create a customer table with generated fake data and save it to a catalog.

    This function generates fake customer data, saves it to a specified catalog, 
    schema, and table, and returns a list of customer IDs (UUIDs).

    Args:
        catalog_name (str): The name of the catalog where the table will be saved.
        schema_name (str): The name of the schema within the catalog.
        table_name (str): The name of the table to save the data.
        n_records (int, optional): The number of customer records to generate. Default is 100.

    Returns:
        list: A list of customer IDs (UUIDs).
    """
    # Generate fake customer data
    data = generate_fake_data(n_records)

    # Save the generated data to the specified catalog, schema, and table
    save_to_uniy_catalog(data, catalog_name, schema_name, table_name)

    # Extract and return a list of customer IDs (UUIDs) from the generated data
    return [person[0] for person in data]

# COMMAND ----------

def get_or_create_customer_table(catalog_name, schema_name, table_name, n_records=100):
    """Create a table with the given name in the given schema in the given catalog."""
    
    if spark.catalog.tableExists(f"{catalog_name}.{schema_name}.{table_name}"):
        print("Customer table already exists.")
        df = spark.table(f"{catalog_name}.{schema_name}.{table_name}")
        return [row["customer_id"] for row in df.select("customer_id").collect()]

    print("Create new customer table.")
    return create_customer_table(catalog_name, schema_name, table_name)

# COMMAND ----------

n_records = 100
uuid_list = get_or_create_customer_table("dbdemos", "retail", "dim_customers", n_records)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Periodische Erstellung neuer Dummy Daten

# COMMAND ----------

def get_aws_credentials(secret_scope):
    """Retrieve AWS credentials securely from Databricks secrets."""
    try:
        aws_access_key = dbutils.secrets.get(secret_scope, "aws_access_key")
        aws_secret_key = dbutils.secrets.get(secret_scope, "aws_secret_key")
        return aws_access_key, aws_secret_key
    except Exception as e:
        print(f"Error retrieving AWS credentials: {e}")
        raise

# COMMAND ----------

import json
import uuid
import random
from datetime import datetime

def generate_transaction_data(n_transactions:int) -> str:
    """
    Generate a dictionary representing a transaction with random data.

    This function generates a transaction record with the following fields:
    - transaktions_id: A unique identifier for the transaction (UUID4).
    - kunden_id: A randomly chosen customer ID from a predefined list.
    - timestamp: The time of the transaction in UTC.
    - products: A list of randomly generated product IDs (UUID4).

    Args:
        n_transactions (int): number of transaction entries in json string
        
    Returns:
        str: A JSON string representation of the transaction data.
    """
    json_data = []
    for _ in range(n_transactions):
        num_products = random.randint(1, 10)  # Random number of products between 1 and 10
        data = {
            "transaction_id": str(uuid.uuid4()),  # Generate a unique transaction ID
            "customer_id": random.choice(uuid_list),  # Select a random customer ID from the list
            "timestamp": datetime.utcnow().isoformat() + 'Z',  # Use UTC time for timestamp
            "purchased_products": [str(uuid.uuid4()) for _ in range(num_products)],  # Generate a list of random product IDs
        }
        json_data.append(data)
    return json.dumps(json_data)

# COMMAND ----------

import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError

def write_data_to_s3(s3_client, bucket_name, s3_path, data):
    """Write data to an S3 bucket."""
    s3_key = f"{s3_path}transaction_data_{datetime.utcnow().strftime('%Y%m%d%H%M%SZ')}.json"
    try:
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=data)
    except (NoCredentialsError, PartialCredentialsError) as e:
        print(f"Credential error: {e}")
        raise
    except Exception as e:
        print(f"Error writing data to S3: {e}")
        raise

# COMMAND ----------

def load_generated_dummy_data_to_s3(n_transactions):
    # AWS credentials
    secret_scope = "demo-autoloader"
    aws_access_key, aws_secret_key = get_aws_credentials(secret_scope)

    # S3 bucket information
    s3_bucket_name = "demo-autoloader"
    s3_path = "dummy-data/"

    # Generate dummy sensor data
    dummy_data = generate_transaction_data(n_transactions)

    # Connect to S3
    s3_client = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    # Write data to S3
    write_data_to_s3(s3_client, s3_bucket_name, s3_path, dummy_data)

# COMMAND ----------

load_generated_dummy_data_to_s3(100)
