import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from features import get_iter_df
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def get_model_losses(data, target, model):
    count_o3 = 0
    count_o2_l3 = 0
    count_o1_l2 = 0
    count_l1 = 0
    prediction = model.predict(data)
    abs_err = abs(target - prediction)
    for i in range(len(abs_err)):
        if abs_err[i] < 1.0:
            count_l1 += 1
        elif abs_err[i] < 2.0:
            count_o1_l2 += 1
        elif abs_err[i] < 3.0:
            count_o2_l3 += 1
        else:
            count_o3 += 1

    print('---------------------------------')
    print('err > 3: %d, %f' % (count_o3, 100 * count_o3 / len(abs_err)))
    print('2 < err < 3: %d, %f' % (count_o2_l3, 100 * count_o2_l3 / len(abs_err)))
    print('1 < err < 2: %d, %f' % (count_o1_l2, 100 * count_o1_l2 / len(abs_err)))
    print('err < 1: %d, %f' % (count_l1, 100 * count_l1 / len(abs_err)))
    print('---------------------------------')

    # abs_err = abs_err.reshape(1, -1)
    # abs_err = abs_err[0]
    # plt.figure(123)
    # plt.xlabel('Предсказания')
    # plt.ylabel('Величина ошибки')
    # plt.scatter(range(len(abs_err)), np.sort(abs_err), s=0.1)


def force_from_torque(model):
    plt.figure(3)
    plt.title('Зависимости нагрузки на штоке от тока двигателя')
    plt.xlabel('Ток двигателя, Arms')
    plt.ylabel('Усилие на штоке, kN')
    x = np.linspace(0.0, 10, num=1000)
    step = 10.0
    t = np.arange(10.0, 80 + step/2, step)
    const_speed = 30.0
    for i in range(len(t)):
        tmp_data = pd.DataFrame()
        tmp_data['speed'] = [const_speed for i in range(len(x))]
        tmp_data['torque'] = x
        tmp_data['temp'] = t[i]

        # lr_model = LinearRegression()
        # lr_model.fit(x.reshape(-1, 1), model.predict(tmp_data))
        # y = lr_model.coef_ * x + lr_model.intercept_
        # y[y < 0] = 0

        m_linestyle = 'solid'
        if i % 4 == 1:
            m_linestyle = 'dashed'
        elif i % 4 == 2:
            m_linestyle = 'dotted'
        elif i % 4 == 3:
            m_linestyle = 'dashdot'

        force = model.predict(tmp_data).reshape(-1, 1)
        force[force < 0] = 0

        m_color = 'silver'
        if i > 3:
            m_color = 'black'
        plt.plot(x, force, label='%d °C' % t[i], linestyle=m_linestyle, c=m_color)

    plt.legend(fontsize=14)
    plt.grid(alpha=0.3)


def get_network_model(data, target, validation):
    X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)

    model = keras.Sequential()
    model.add(keras.Input(shape=(3,)))
    for i in range(1):
        model.add(Dense(units=100, activation=keras.activations.selu))
    model.add(Dense(units=1))

    model.compile(loss=keras.losses.mean_absolute_error, optimizer=keras.optimizers.Adam(learning_rate=0.001))

    history = model.fit(X_train, Y_train, epochs=100, use_multiprocessing=True, validation_split=0.3, workers=8)

    model.save('./models/train_on_all_data/network_model_100_selu_test')

    plt.figure(1)
    plt.title('Losses')
    plt.plot(history.history['loss'][1:], label='Training data', c='black', linestyle='solid')
    plt.plot(history.history['val_loss'][1:], label='Validation data', c='grey', linestyle='dashed')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=14)

    plt.figure(33)
    plt.title('Predictions')
    Y_test = Y_test.T[0]
    X_test['force'] = Y_test.T
    X_test = X_test.sort_values(by='force')
    X_test1 = X_test.copy()
    X_test1 = X_test1.drop(['force'], axis=1)
    y_predict = model.predict(X_test1)
    test_losses = abs(X_test['force'].values - y_predict.T[0])
    print('Test mean loss: ', test_losses.mean())
    print('Test max loss: ', test_losses.max())
    plt.scatter(range(len(X_test1)), y_predict, s=0.1)
    plt.scatter(range(len(X_test)), X_test['force'], s=0.1)

    return model


def force_from_data(model):
    df = pd.DataFrame()
    m_num = 50
    df['speed'] = np.linspace(-150.0, 150.0, num=m_num)
    df['torque'] = np.linspace(-10.0, 10.0, num=m_num)
    df['temp'] = np.linspace(0.0, 100.0, num=m_num)

    fig = plt.figure(4)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('Force(speed, torque)')
    ax.set_xlabel('Speed')
    ax.set_ylabel('Torque')
    ax.set_zlabel('Force')

    tmp_df = df.copy()

    tmp_df['temp'] = 50.0
    for speed in df['speed']:
        tmp_df['speed'] = speed
        ax.scatter(tmp_df['speed'], tmp_df['torque'], model.predict(tmp_df))


def create_points_table(model):
    # Содание dataframe размера N*N*N для всех комбинаций скорости, тока и температуры
    # и выгрузка его в points_table.csv
    df = pd.DataFrame()
    n = 10
    df['speed'] = np.linspace(-150.0, 150.0, num=n)
    df['torque'] = np.linspace(-10.0, 10.0, num=n)
    df['temp'] = np.linspace(0.0, 100.0, num=n)

    result_df = pd.DataFrame()
    tmp_df = pd.DataFrame()
    for speed in range(n):
        for torque in range(n):
            tmp_df['speed'] = pd.Series(df['speed'][speed], index=np.arange(n))
            tmp_df['torque'] = df['torque'][torque]
            tmp_df['temp'] = df['temp']
            result_df = pd.concat([result_df, tmp_df], axis=0)
            # table[speed][torque] = model.predict(tmp_df).T  # очень дорого для каждого массива

    result_df['force'] = model.predict(result_df)
    result_df.to_csv('points_table.csv')


def using_points_table():
    # Использование points_table.csv для быстрого получения усилия по заданным параметрам
    # на низком уровне. Параметры могут не содержаться в таблице!
    table = pd.read_csv(os.path.abspath('points_table.csv'))
    table = table.drop(table.columns[[0]], axis=1)

    input = [np.random.uniform(-150.0, 150.0), np.random.uniform(-10.0, 10.0), np.random.uniform(0.0, 100.0)]
    print(input)


def test_model(model):
    df = pd.read_csv(os.path.abspath('1file_dataset.csv'))
    df = get_iter_df(df)  # прореживание данных

    target = df[['force']].values
    data = df.drop(['force', 'time'], axis=1)
    data = data.drop(data.columns[[0]], axis=1)

    plt.figure(22)
    plt.title('Predictions on another dataset')
    plt.scatter(range(len(data)), model.predict(data), s=0.1)
    plt.scatter(range(len(target)), target, s=0.1)

    get_model_losses(data, target, model)


def start():
    all_df = pd.read_csv(os.path.abspath('dataset.csv'))
    all_df = get_iter_df(all_df)  # прореживание данных
    all_target = all_df[['force']].values
    all_data = all_df.drop(['force', 'time'], axis=1)
    all_data = all_data.drop(all_data.columns[[0]], axis=1)

    df = pd.read_csv(os.path.abspath('2files_dataset.csv'))
    df = get_iter_df(df)  # прореживание данных
    target = df[['force']].values
    data = df.drop(['force', 'time'], axis=1)
    data = data.drop(data.columns[[0]], axis=1)

    df1 = pd.read_csv(os.path.abspath('1file_dataset.csv'))
    df1 = get_iter_df(df)  # прореживание данных
    val_target = df1[['force']].values
    val_data = df1.drop(['force', 'time'], axis=1)
    val_data = val_data.drop(val_data.columns[[0]], axis=1)

    # ОПРЕДЕЛЕНИЕ МОДЕЛИ (обучение/загрузка)
    # model = get_network_model(all_data, all_target, (val_data, val_target))
    model = keras.models.load_model('./models/train_on_all_data/network_model_100_selu_5kE')

    # X_train, X_test, Y_train, Y_test = train_test_split(data, target, test_size=0.2)

    # plt.figure(2)
    # plt.title('Predictions')
    # Y_test = Y_test.T[0]
    # X_test['force'] = Y_test.T
    # X_test = X_test.sort_values(by='force')
    # X_test1 = X_test.copy()
    # X_test1 = X_test1.drop(['force'], axis=1)
    # y_predict = model.predict(X_test1)
    # test_losses = abs(X_test['force'].values - y_predict.T[0])
    # print('Test mean loss: ', test_losses.mean())
    # print('Test max loss: ', test_losses.max())
    # plt.scatter(range(len(X_test1)), y_predict, s=0.1)
    # plt.scatter(range(len(X_test)), X_test['force'], s=0.1)

    plt.figure(2)
    plt.title('Predictions')
    plt.scatter(range(len(all_data)), model.predict(all_data), s=0.1, c='gray', label='Prediction')
    plt.scatter(range(len(all_target)), all_target, s=0.1, c='black', label='Target')
    classes = ['Prediction', 'Target']
    recs = [mpatches.Rectangle((0, 0), 1, 1, fc='gray'), mpatches.Rectangle((0, 0), 1, 1, fc='black')]
    plt.legend(recs, classes, loc=3, fontsize=14)
    plt.grid(alpha=0.3)

    force_from_torque(model)

    get_model_losses(all_data, all_target, model)

    # test_model(model)

    # force_from_data(model)

    # create_points_table(model)

    # using_points_table()
