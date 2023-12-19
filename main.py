import psycopg2
from my_config import config
import pandas as pd
import os
from sqlalchemy import create_engine
import logging


direct_path = "../data/Hourly_Standard/Cleaned"

# Configure logging
log_file = "database_operations_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

def connect(extrt):
    connection = None
    try:
        params = config
        logging.info('Connecting to the PostgreSQL database...')
        connection = psycopg2.connect(**params)
        crsr = connection.cursor()
        logging.info('PostgreSQL database version:')
        crsr.execute('SELECT version()')
        db_version = crsr.fetchone()
        logging.info(db_version)
        
        if extrt == "insert":
            create_dataframe(crsr, 'process')
        elif extrt == "extract":
            return extract_data_from_table(crsr, "process")

        crsr.close()
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error("Error in connect: {}".format(str(error)))
    finally:
        if connection is not None:
            connection.close()
            logging.info('Database connection terminated.')

def create_dataframe(cursor, table_name):
    try:
        if not os.path.exists(direct_path):
            logging.error("Directory path is wrong!!!")
        else:
            for filename in os.listdir(direct_path):
                file_path = os.path.join(direct_path, filename)
                if filename.endswith(".csv"):
                    df = pd.read_csv(file_path)
                    df.drop("Unnamed: 0", axis=1, inplace=True)
                    insert_dataframe_into_table(df, cursor, table_name)
                    logging.info(f'Data from file {filename} inserted into table {table_name}.')
    except Exception as e:
        logging.error("Error in create_dataframe: {}".format(str(e)))
        raise

def extract_data_from_table(cursor, table_name):
    try:
        cursor.execute(f'SELECT * FROM {table_name}')
        rows = cursor.fetchall()
        logging.info(f'Data from table {table_name}:')
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
        logging.info(df)
        return df
    except psycopg2.Error as e:
        logging.error(f'Error extracting data from {table_name}: {e}')
        raise

def insert_dataframe_into_table(dataframe, cursor, table_name):
    try:
        columns = ', '.join(dataframe.columns)
        values = ', '.join('%s' for _ in dataframe.columns)
        print(columns)
        for row in dataframe.itertuples(index=False, name=None):
            
            insert_query = f"INSERT INTO {table_name} ({columns}) VALUES ({', '.join(['%s' for _ in row])})"
            cursor.execute(insert_query, row)
        cursor.connection.commit()

        logging.info(f'DataFrame inserted into {table_name} successfully using cursor.')
    except Exception as e:
        logging.error(f'Error inserting DataFrame into {table_name}: {e}')
        raise

  
