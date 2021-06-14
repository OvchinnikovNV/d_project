import numpy as np


def draw_plot_df(df, figure_index):
    import matplotlib.pyplot as plt

    fig = plt.figure(figure_index)
    ax = fig.add_subplot(projection='3d')

    x = df['reg_position_output'].values
    y = df['torque'].values
    z = df['temp'].values

    pos = 150
    torque = 10
    temp = 100
    for j in range(len(x)):
        if x[j] > pos:
            x[j] = pos
        if x[j] < -pos:
            x[j] = -pos

        if y[j] > torque:
            y[j] = torque
        if y[j] < -torque:
            y[j] = -torque

        if z[j] > temp:
            z[j] = temp
        if z[j] < -temp:
            z[j] = -temp

    # m_color = [((i * 50 + 500) % 255) / 255, ((i * 70 + 700) % 255) / 255, ((i * 90 + 900) % 255) / 255]
    ax.scatter(x, y, z, s=1)

    ax.set_xlabel('Speed')
    ax.set_ylabel('Torque')
    ax.set_zlabel('Temp')
    # plt.show()


# прореживание данных
def get_iter_df(df):
    it = []
    cond1 = np.sign(df['reg_position_output']) == np.sign(df['torque'])
    cond2 = np.sign(df['reg_position_output']) == np.sign(df['force'] + df['reg_position_output'] * 0.00001)
    cond3 = np.sign(df['torque']) == np.sign(df['force'] + df['torque'] * 0.00001)
    cond4 = abs(df['reg_position_output']) > 0.5

    return df.query("@cond1 & @cond2 & @cond3 & @cond4")


def linear_regression_step(theta, data, target, alfa):
    N = data.shape

    new_theta = theta.copy()

    # вычисляем ошибку гипотезы
    error_hyp = data.dot(theta) - target

    for i in range(N[1]):
        tmp_data = np.reshape(data[:, i], (len(data), 1))
        update_value = np.sum(error_hyp * tmp_data) / N[0]
        # update_value = np.sum( error_hyp * data[:, i] ) / N[0]
        new_theta[i] = theta[i] - alfa * update_value

    return new_theta.copy()


def f(index, N, theta, data, error_hyp, alfa):
    tmp_data = np.reshape(args[3][:, args[0]], (len(data), 1))
    update_value = np.sum(error_hyp * tmp_data) / args[1]
    return args[2][args[0]] - alfa * update_value


def mp_linear_regression_step(theta, data, target, alfa):
    from multiprocessing import Pool

    N = data.shape

    new_theta = theta.copy()

    # вычисляем ошибку гипотезы
    error_hyp = data.dot(theta) - target

    # for i in range(N[1]):
    #     tmp_data = np.reshape(data[:, i], (len(data), 1))
    #     update_value = np.sum(error_hyp * tmp_data) / N[0]
    #     new_theta[i] = theta[i] - alfa * update_value

    args = [[0] * 6 for i in range(N[1])]
    for i in range(len(args)):
        args[i][0] = i
        args[i][1] = N[0]
        args[i][2] = theta
        args[i][3] = data
        args[i][4] = error_hyp
        args[i][5] = alfa
    pool = Pool(processes=4)
    for i in range(len(args)):
        result = pool.map(f, args)
    print('asd')
    # return new_theta.copy()


def get_features():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(os.path.abspath('dataset.csv'))

    figure_index = 1
    # draw_plot_df(df, figure_index)
    # figure_index += 1

    # прореживание данных
    df = get_iter_df(df)
    target = df['force'].values
    target = np.reshape(target, (len(target), 1))

    # draw_plot_df(df, figure_index)
    # figure_index += 1

    m = 19
    features = np.zeros((len(df.index), m))
    KT = 300
    KD = 600

    features[:, 0] = -np.sign(df['reg_position_output'])
    features[:, 1] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 0.5)
    features[:, 2] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 1)
    features[:, 3] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 1.5)
    features[:, 4] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 2)
    features[:, 5] = df['torque']

    features[:, 6] = features[:, 0] * ((KT - df['temp']) / KD)
    features[:, 7] = features[:, 1] * ((KT - df['temp']) / KD)
    features[:, 8] = features[:, 2] * ((KT - df['temp']) / KD)
    features[:, 9] = features[:, 3] * ((KT - df['temp']) / KD)
    features[:, 10] = features[:, 4] * ((KT - df['temp']) / KD)
    features[:, 11] = features[:, 5] * ((KT - df['temp']) / KD)

    features[:, 12] = features[:, 0] * (((KT - df['temp']) / KD) ** 2)
    features[:, 13] = features[:, 1] * (((KT - df['temp']) / KD) ** 2)
    features[:, 14] = features[:, 2] * (((KT - df['temp']) / KD) ** 2)
    features[:, 15] = features[:, 3] * (((KT - df['temp']) / KD) ** 2)
    features[:, 16] = features[:, 4] * (((KT - df['temp']) / KD) ** 2)
    features[:, 17] = features[:, 5] * (((KT - df['temp']) / KD) ** 2)
    features[:, 18] = 1

    # пирведение всех показателей в один диапазон
    K = np.zeros((2, m))
    K[0] = 1
    K[1] = 0

    # plt.figure(figure_index)
    # for i in range(len(features[0])):
    #     plt.scatter(range(len(features[:, i])), features[:, i], s=0.1)
    # figure_index += 1

    for i in range(m - 1):
        K[0][i] = np.max(features[:, i]) - np.min(features[:, i])
        K[1][i] = np.mean(features[:, i])

        features[:, i] = (features[:, i] - K[1][i]) / K[0][i]

    # plt.figure(figure_index)
    # for i in range(len(features[0])):
    #     plt.scatter(range(len(features[:, i])), features[:, i], s=0.1)
    # figure_index += 1

    # инициализация модели
    theta = np.zeros((m, 1))
    # линейная регрессия
    N = 600  # число шагов
    alfa = 0.6  # скорость обучения
    theta = linear_regression_step(theta, features, target, alfa)
    # цикл обучения
    for i in range(N):
        theta = linear_regression_step(theta, features, target, alfa)
        # mp_linear_regression_step(theta, features, target, alfa)

    # вывод резульатов
    hypot = features.dot(theta)
    plt.figure(figure_index)
    plt.scatter(range(len(hypot)), hypot, s=0.1)
    plt.scatter(range(len(target)), target, s=0.1)
    figure_index += 1

    # t = np.arange(-40, 121, 10)
    # sizer = t.shape
    #
    # w = np.zeros(sizer) + 145
    #
    # points = np.zeros((sizer[0], m))
    #
    # points[:, 0] = -np.sign(w)
    # points[:, 1] = -np.sign(w) * abs(w) ** 0.5
    # points[:, 2] = -np.sign(w) * abs(w) ** 1
    # points[:, 3] = -np.sign(w) * abs(w) ** 1.5
    # points[:, 4] = -np.sign(w) * abs(w) ** 2
    # points[:, 5] = 0
    # points[:, 6] = points[:, 0] * ((KT - t) / KD)
    # points[:, 7] = points[:, 1] * ((KT - t) / KD)
    # points[:, 8] = points[:, 2] * ((KT - t) / KD)
    # points[:, 9] = points[:, 3] * ((KT - t) / KD)
    # points[:, 10] = points[:, 4] * ((KT - t) / KD)
    # points[:, 11] = 0
    # points[:, 12] = points[:, 0] * ((KT - t) / KD) ** 2
    # points[:, 13] = points[:, 1] * ((KT - t) / KD) ** 2
    # points[:, 14] = points[:, 2] * ((KT - t) / KD) ** 2
    # points[:, 15] = points[:, 3] * ((KT - t) / KD) ** 2
    # points[:, 16] = points[:, 4] * ((KT - t) / KD) ** 2
    # points[:, 17] = 0
    # points[:, 18] = 1
    #
    # b = np.zeros(sizer)
    # vals = np.zeros(points.shape)
    # for i in range(m):
    #     vals[:, i] = (points[:, i] - K[1][i]) / K[0][i]
    #
    # k = np.zeros(sizer)
    # for i in range(sizer[0]):
    #     b[i] = np.sum(np.dot(vals[i, :], theta))
    #     k[i] = (theta[5] / K[0][5]) \
    #            + (theta[11] / K[0][11]) * ((KT - t[i]) / KD) \
    #            + (theta[17] / K[0][17]) * ((KT - t[i]) / KD) ** 2
    #     print(k[i], b[i])
    #
    # plt.figure(figure_index)
    # for i in range(sizer[0]):
    #     x = np.linspace(0, 15, num=1000)
    #     y = x * k[i] + b[i]
    #     # y[y < 0] = 0
    #     plt.scatter(x, y, s=0.1)
    # figure_index += 1

    # plt.show()


def m_get_features():
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import PolynomialFeatures
    from scipy.optimize import curve_fit

    df = pd.read_csv(os.path.abspath('dataset.csv'))

    # прореживание данных
    df = get_iter_df(df)
    target = df[['force']]
    data = df.drop(['force', 'time'], axis=1)
    data = data.drop(data.columns[[0]], axis=1)

    m = 19
    features = np.zeros((len(df.index), m))
    KT = 300
    KD = 600

    features[:, 0] = -np.sign(df['reg_position_output'])
    features[:, 1] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 0.5)
    features[:, 2] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 1)
    features[:, 3] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 1.5)
    features[:, 4] = -np.sign(df['reg_position_output']) * (abs(df['reg_position_output']) ** 2)
    features[:, 5] = df['torque']

    features[:, 6] = features[:, 0] * ((KT - df['temp']) / KD)
    features[:, 7] = features[:, 1] * ((KT - df['temp']) / KD)
    features[:, 8] = features[:, 2] * ((KT - df['temp']) / KD)
    features[:, 9] = features[:, 3] * ((KT - df['temp']) / KD)
    features[:, 10] = features[:, 4] * ((KT - df['temp']) / KD)
    features[:, 11] = features[:, 5] * ((KT - df['temp']) / KD)

    features[:, 12] = features[:, 0] * (((KT - df['temp']) / KD) ** 2)
    features[:, 13] = features[:, 1] * (((KT - df['temp']) / KD) ** 2)
    features[:, 14] = features[:, 2] * (((KT - df['temp']) / KD) ** 2)
    features[:, 15] = features[:, 3] * (((KT - df['temp']) / KD) ** 2)
    features[:, 16] = features[:, 4] * (((KT - df['temp']) / KD) ** 2)
    features[:, 17] = features[:, 5] * (((KT - df['temp']) / KD) ** 2)
    features[:, 18] = 1

    # пирведение всех показателей в один диапазон
    K = np.zeros((2, m))
    K[0] = 1
    K[1] = 0
    for i in range(m - 1):
        K[0][i] = np.max(features[:, i]) - np.min(features[:, i])
        K[1][i] = np.mean(features[:, i])

        features[:, i] = (features[:, i] - K[1][i]) / K[0][i]

    model = LinearRegression()
    model.fit(features, target)
    print('Features r-squared:', model.score(features, target))

    plt.figure(10)
    plt.title('Features')
    plt.scatter(range(len(features)), model.predict(features), s=0.1)
    plt.scatter(range(len(target)), target, s=0.1)
