import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from re import search, sub
from parser import parse_log


def draw_plot(data):
    K_m = 0.5546
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    for i in range(len(data)):
        x = np.array(data[i][1])
        y = np.array(data[i][2])
        y *= K_m
        z = np.array(data[i][3])

        for j in range(len(x)):
            if x[j] > 150:
                x[j] = 150
            if x[j] < -150:
                x[j] = -150

            if y[j] > 10:
                y[j] = 10
            if y[j] < -10:
                y[j] = -10

            if z[j] > 100:
                z[j] = 100
            if z[j] < -100:
                z[j] = -100

        m_color = [((i * 50 + 500) % 255) / 255, ((i * 70 + 700) % 255) / 255, ((i * 90 + 900) % 255) / 255]
        ax.scatter(x, y, z, s=1, c=[m_color])

    ax.set_xlabel('Reg_position_output')
    ax.set_ylabel('Torque')
    ax.set_zlabel('Temperature')
    plt.show()


def parse_filename(filename):
    sign = -1 if search('fs', filename.lower()) else 1
    match = search('\d+kn', filename.lower())
    return sign * int(sub('\D', '', match[0]))


def get_iter(ref_speed):
    n = len(ref_speed)
    ref_acc = pd.Series(ref_speed[1:n] - ref_speed[0:n-1])
    return list(ref_acc[ref_acc == 0].index)


def prepare_data(paths):
    data = []
    force = []

    for path in paths:
        for dirpath, dirnames, filenames in os.walk(path):
            for file in filenames:
                if search('fc_20kN\.log', file):
                    data.append(parse_log(path, file))
                    force.append(parse_filename(file))

    # get iter for "reg_position_output" column
    it = []
    for d in data:
        it.append(get_iter(d[1]))

    new_data = []
    for k in range(len(data)):
        new_data.append([None, None, None, None])
        # new_data.append([None, None, None, None, None, None, None, None])
        for i in range(4):
            new_data[k][i] = np.take(data[k][i], it[k])

    # draw_plot(new_data)

    column1 = []  # time
    column2 = []  # reg_position_output
    column3 = []  # torque
    column4 = []  # temperature
    column5 = []  # force
    for i in range(len(new_data)):
        column1.extend(data[i][0])
        column2.extend(data[i][1])
        column3.extend(data[i][2])
        column4.extend(data[i][3])
        for j in range(len(data[i][0])):
            column5.append(force[i])

    # column1 = []  # time
    # column2 = []  # reg_position_output
    # column3 = []  # torque
    # column4 = []  # temperature
    # column5 = []  # force
    # for i in range(len(new_data)):
    #     column1.extend(new_data[i][0])
    #     column2.extend(new_data[i][1])
    #     column3.extend(new_data[i][2])
    #     column4.extend(new_data[i][3])
    #     for j in range(len(new_data[i][0])):
    #         column5.append(force[i])

    # column1 = []  # time
    # column2 = []  # reg_position_output
    # column3 = []  # torque
    # column4 = []  # temperature
    # column5 = []  # speed
    # column6 = []  # 9_position
    # column7 = []  # reg_speed_output
    # column8 = []  # trj_target_position
    # column9 = []  # force
    # for i in range(len(new_data)):
    #     column1.extend(new_data[i][0])
    #     column2.extend(new_data[i][1])
    #     column3.extend(new_data[i][2])
    #     column4.extend(new_data[i][3])
    #     column5.extend(new_data[i][4])
    #     column6.extend(new_data[i][5])
    #     column7.extend(new_data[i][6])
    #     column8.extend(new_data[i][7])
    #     for j in range(len(new_data[i][0])):
    #         column9.append(force[i])

    df = pd.DataFrame({'time': column1, 'speed': column2, 'torque': column3, 'temp': column4,
                       'force': column5})
    # df['torque'] *= 0.5546
    df.to_csv('_dataset.csv')
