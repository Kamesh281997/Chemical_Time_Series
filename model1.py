import os
import logging
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import math
from sklearn.metrics import mean_squared_error
from numpy import array

log_file = "lstm_log.txt"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(asctime)s - %(levelname)s: %(message)s")
scaler = MinMaxScaler(feature_range=(0, 1))


def find_data(data, batch):
    try:
        batch_data = data[data["batch"] == batch]
        batch_data.sort_values(by="hourly_timestamp", ascending=True, inplace=True)
        return batch_data
    except Exception as e:
        logging.info("Error {} in file model.py:".format(e))
        return None 

def create_data(data):
    try:
        df1 = data.reset_index()["waste"]
        df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
        training_size = int(len(df1) * 0.60)
        test_size = len(df1) - training_size
        train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]
        return df1, train_data, test_data, training_size, test_size
    except Exception as e:
        logging.error("Error in SARIMA model: {}".format(str(e)))

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def split_data(X_train, X_test):
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    return X_train, X_test

def build_model(X_train, y_train, X_test, ytest):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=(30, 1)))
    model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(100))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=50, batch_size=64, verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    return train_predict, test_predict, model

def plot_training_testing_actual(df1, y_train, ytest, train_predict, test_predict, batch):
    output_directory = '../graph/plot_tbl_speed_prediction1'
    os.makedirs(output_directory, exist_ok=True)
    rmse_train = math.sqrt(mean_squared_error(y_train, train_predict))
    rmse_test = math.sqrt(mean_squared_error(ytest, test_predict))
    print("RMSE_TRAIN:",rmse_train)
    print("RMSE_TEST:",rmse_test)
    look_back = 30
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back: len(train_predict) + look_back, :] = train_predict
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(df1) - len(test_predict): len(df1), :] = test_predict
    plt.plot(scaler.inverse_transform(df1), label='Actual Data')
    plt.plot(trainPredictPlot, label='Training Prediction')
    plt.plot(testPredictPlot, label='Testing Prediction')
    plt.savefig(f"{output_directory}/model_produced_training_testing_actual_batch{batch}.png")
    plt.legend()
    plt.show()

def extend_20_days(model, temp_input, x_input):
    lst_output = []
    n_steps = 30
    i = 0
    while i < 20:
        if len(temp_input) > 30:
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            yhat[0][0]=round(yhat[0][0],2)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            lst_output.extend(yhat.tolist())
            i = i + 1
        else:
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i = i + 1
    rounded_lst_output = [[round(value[0], 2)] for value in lst_output]
    return rounded_lst_output

def next20dayspred(df1, lst_output, batch):
    output_directory = '../graph/prediction_tbl_speed_extend1'
    os.makedirs(output_directory, exist_ok=True)
    actual_length = len(df1) - 30
    day_pred = np.arange(actual_length + 1, actual_length + 21)
    actual_data = scaler.inverse_transform(df1[len(df1) - 30:])
    predicted_data = scaler.inverse_transform(lst_output)
    actual_data_for_plot = actual_data[-actual_length:]
    print("Actual_data1:",actual_data_for_plot)
    print("Predicted_data1:",predicted_data)
    plt.plot(np.arange(1, len(actual_data_for_plot) + 1), actual_data_for_plot, label='Actual Data', linestyle='dashed')
    plt.plot(day_pred, np.array(predicted_data ).reshape(-1), label='Predicted Data', linestyle='dashed')
    plt.legend()
    plt.savefig(f"{output_directory}/model_produced_next20days_prediction_batch{batch}.png")
    plt.show()


def combined_plot(df1, lst_output, batch):
    output_directory = '../graph/prediction_tbl_speed_extend_combined'
    os.makedirs(output_directory, exist_ok=True)
    actual_length = len(df1) - 30
    actual_data = scaler.inverse_transform(df1[len(df1) - 30:])
    predicted_data = scaler.inverse_transform(lst_output)
    actual_data_for_plot = actual_data[-actual_length:]
    combined_line = np.concatenate([actual_data_for_plot.flatten(), np.array(predicted_data).reshape(-1)])
    plt.plot(np.arange(1, len(combined_line) + 1), combined_line, label='Combined Data', color='black')
    plt.legend()
    plt.savefig(f"{output_directory}/model_produced_next20days_combined_prediction_batch{batch}.png")
    plt.show()


def main_model(batch_lst,df):
    for val in batch_lst:
        df1 = find_data(df, val)
        df1, train_data, test_data, training_size, test_size = create_data(df1)
        time_step = 30
        X_train, y_train = create_dataset(train_data, time_step)
        X_test, ytest = create_dataset(test_data, time_step)
        X_train, X_test = split_data(X_train, X_test)
        train_predict, test_predict, model = build_model(X_train, y_train, X_test, ytest)
        plot_training_testing_actual(df1, y_train, ytest, train_predict, test_predict, val)
        x_input = test_data[len(test_data) - 30:].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        lst_output = extend_20_days(model, temp_input, x_input)
        next20dayspred(df1, lst_output, val)
        combined_plot(df1, lst_output, val)


# if __name__ == "__main__":
#     batch_lst = [62]
#     main_model(batch_lst)
