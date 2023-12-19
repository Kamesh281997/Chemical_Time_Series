import pandas as pd
import os
import argparse
import logging
from Standard_data import  process_all_files
from data_preprocessing import process_csv_files_in_folders
from hourly_data_analysis import get_data
from data_decomposition import decomposition
from model1 import main_model

log_file = "data_input.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

def standardized_file(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                logging.info("Processing file: {}".format(file))
                print(folder_path + "/" + file)
                path = os.path.join(folder_path, file)
                try:
                    df = pd.read_csv(path, header=None, names=['Column'])
                    unique_columns = df['Column'].unique()
                    columns = unique_columns[0]
                    column_name = columns.split(';')
                    lst = [unique_columns[i].split(';') for i in range(1, len(unique_columns))]
                    df2 = pd.DataFrame(lst, columns=column_name)
                    df2['timestamp'] = pd.to_datetime(df2['timestamp'])
                    df2.sort_values(by='timestamp', inplace=True)
                    directory_path = f"{folder_path}/Standard/"
                    if not os.path.exists(directory_path):
                        os.makedirs(f"{folder_path}/Standard/")
                    df2.to_csv(f"{folder_path}/Standard/{file}", index=False)
                    logging.info("File {} processed successfully".format(file))
                except Exception as e:
                    logging.error("Error processing file {}: {}".format(file, str(e)))

    print("Sum of all unique batches are {}".format(sum))
    logging.info("Sum of all unique batches are {}".format(sum))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Input filename")
    parser.add_argument("argument", help="Filename Input:")
    args = parser.parse_args()
    standardized_file(args.argument)
    try:
        source_folder = "..\data\Raw\Process\Standard"
        destination_folder = '..\data\Hourly_Standard\Process'
        process_all_files(source_folder,destination_folder)
    except Exception as e:
        logging.error("Error in main script: {}".format(str(e)))
    try:
        root_folder = '..\data\Hourly_Standard\Process'
        dest_path = '..\data\Hourly_Standard\Cleaned'
        process_csv_files_in_folders(root_folder, dest_path)
    except Exception as e:
        logging.error("Error in main script: {}".format(str(e)))
    try:
        get_data("insert")
    except Exception as e:
        logging.error("Error in main script: {}".format(str(e)))
    try:
        df=get_data("extract")
        app=decomposition(df)
        batch_lst=[62]
        main_model(batch_lst,df)
    except Exception as e:
        logging.error("Error in main script: {}".format(str(e)))
        
