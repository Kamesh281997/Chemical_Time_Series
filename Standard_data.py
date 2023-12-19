import pandas as pd
import os
import argparse
import logging

log_file = "standard_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")

def process_and_save_csv(input_path, output_folder):
    try:
        print("Files::", input_path)
        logging.info("Processing file: {}".format(input_path))
        df = pd.read_csv(input_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hourly_timestamp'] = df['timestamp'].dt.floor('10T')
        hourly_grouped_df = df.groupby(['campaign', 'batch', 'hourly_timestamp']).mean().reset_index()
        campaign_folder = os.path.join(output_folder, f'Campaign_{hourly_grouped_df["campaign"].iloc[0]}')
        os.makedirs(campaign_folder, exist_ok=True)
        output_path = os.path.join(campaign_folder, f'hourly_data_{os.path.basename(input_path)}')
        column_to_drop_index = 0  
        hourly_grouped_df = hourly_grouped_df.drop(df.columns[column_to_drop_index], axis=1,inplace=True)
        hourly_grouped_df.drop('timestamp', axis=1)
        hourly_grouped_df.to_csv(output_path, index=False)
        print(f"Processed and saved hourly data for Campaign {hourly_grouped_df['campaign'].iloc[0]} to {output_path}")
        logging.info("Processed and saved hourly data for Campaign {} to {}".format(hourly_grouped_df['campaign'].iloc[0], output_path))

    except Exception as e:
        print(f"Error processing file {input_path}: {str(e)}")
        logging.error("Error processing file {}: {}".format(input_path, str(e)))

def process_all_files(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_path = os.path.join(input_folder, filename)
            print(input_path)
            process_and_save_csv(input_path, output_folder)
            


