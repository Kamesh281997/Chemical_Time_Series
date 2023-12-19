from main import connect
import pandas as pd
import numpy as np
import psycopg2
import logging
import matplotlib.pyplot as plt
import os
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import plotly.express as px

def get_data(extrt):
    try:
        df=connect(extrt)
        if extrt=="extract":
            if not os.path.exists("../data/all_combined.csv"):
                df.to_csv("../data/all_combined.csv")
            df['hourly_timestamp'] = pd.to_datetime(df['hourly_timestamp'])
            df.set_index('hourly_timestamp',inplace=True)
            stacked_df=df.copy()
            stacked_df.drop(["waste"],axis=1,inplace=True)
            lst=stacked_df.columns
            if extrt=="extract":
                plot_graph(df,lst)
                all_contribution(df)
                column_seasonality(df)
                dct=adfuller_test(stacked_df)
                non_stat_df=pd.DataFrame(dct)
                non_stat_df.to_csv("../data/non_stationary.csv",index=False)
            return df
    except Exception as e:
        logging.error("Error in plot_graph: {}".format(str(e)))
    
def plot_graph(data,lst):
    try:
        output_directory = '../graph/batch_line_charts'
        os.makedirs(output_directory, exist_ok=True)
        for batch_number, batch_df in data.groupby('batch'):
            for val in lst:
                sns.set(style="whitegrid")
                plt.figure(figsize=(12, 6))
                sns.lineplot(batch_df, x=batch_df.index, y=val, label=val, marker='o')
                sns.lineplot(batch_df, x=batch_df.index, y='waste', label='Waste', marker='+')
                plt.title(f'{val} and Waste Over Time for Batch {batch_number}')
                plt.xlabel('Hourly Timestamp')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                output_file_path = os.path.join(output_directory, f'v{val}_batch_{batch_number}_line_chart.png')
                plt.savefig(output_file_path)
                plt.close()
        print("Line charts saved.")
    except Exception as e:
        logging.error("Error in plot_graph: {}".format(str(e)))
   
def all_contribution(stacked_data):
    try:
        dataset = stacked_data
        output_directory = '../graph/all_col_distributions'
        os.makedirs(output_directory, exist_ok=True)
        columns_to_plot = ['tbl_speed', 'fom', 'tbl_fill', 'produced', 'waste', 'ejection']
        batch_groups = dataset.groupby('batch')

        for batch_number, batch_data in batch_groups:
            plt.figure(figsize=(15, 15))
            for i, column in enumerate(columns_to_plot, start=1):
                plt.subplot(len(columns_to_plot), 1, i)
                sns.histplot(batch_data[column], bins=20, kde=True, color='blue', edgecolor='black')
                plt.title(f'Batch {batch_number} - {column}')
                plt.xlabel(column)
                plt.ylabel('Frequency')
            output_file_path = os.path.join(output_directory, f'batch_{batch_number}_plot.png')
            plt.savefig(output_file_path)
            plt.tight_layout()
    except Exception as e:
        logging.error("Error in plot_graph: {}".format(str(e)))

def column_seasonality(df):
    try:
        sns.set(style="darkgrid")
        batch_numbers = df['batch'].unique()
        output_directory = '../graph/columns_distribution'
        os.makedirs(output_directory, exist_ok=True)
        for batch_number in batch_numbers:
            batch_data = df[df['batch'] == batch_number]
            for feature in df.columns.difference(['batch']):
                plt.figure(figsize=(12, 6))
                sns.lineplot(data=batch_data, x=batch_data.index, y=feature, label=feature, marker='o')
                plt.title(f'Batch {batch_number} - {feature} Over Time')
                plt.xlabel('Timestamp')
                plt.ylabel(feature)
                plt.legend()
                output_file_path = os.path.join(output_directory, f'batch_{batch_number}_{feature}_plot.png')
                plt.savefig(output_file_path)
                plt.close() 
    except Exception as e:
        logging.error("Error in column_seasonality: {}".format(str(e)))


def make_stationary(data):
    try:
        data = data.reset_index(drop=True)
        stationary_data = data.diff().dropna()
        stationary_data.fillna(0, inplace=True)
        return stationary_data
    except Exception as e:
        logging.error("Error in make_stationary: {}".format(str(e)))

def adfuller_test(stacked_data):
    batch_numbers = stacked_data['batch'].unique()
    features = stacked_data.columns.difference(['batch', 'campaign'])  # Exclude non-numeric columns  
    non_stationary_batches = {'Batch_number': [], 'Feature': []}
    for feature in features:
        print(f"Feature: {feature}")
        for batch_number in batch_numbers:
            batch_number_int64 = np.int64(batch_number)
            batch_data = stacked_data[(stacked_data['batch'] == batch_number)][feature]

            try:
                test_result = adfuller(batch_data)
                adf_statistic = test_result[0]
                p_value = test_result[1]

                if p_value <= 0.05:
                    print(f"Batch {batch_number_int64}: Reject the null hypothesis - The data is stationary.")
                else:
                    non_stationary_batches['Batch_number'].append(batch_number_int64)
                    non_stationary_batches['Feature'].append(feature)
                    print(f"Batch {batch_number_int64}: Fail to reject the null hypothesis - The data is non-stationary. Applying differencing.")
                print()

            except Exception as e:
                print(f"Error processing Batch {batch_number_int64} for Feature {feature}: {e}")
                logging.error("Error in adfuller_test: {}".format(str(e)))
    return non_stationary_batches
   

    
# if __name__ == "__main__":
#     get_data("extract")