def parse_log(path, file):
    import numpy as np
    from re import search
    from copy import deepcopy

    # read file in raw_data
    fileName = path + file
    raw_data = []
    with open(fileName, 'r') as file:
        headline = file.readline().split(',')

        # get { column: index }
        indices = dict.fromkeys(['time', '10_speed', 'torque', 'temp'])
        for i in range(len(headline)):
            for key in list(indices.keys()):
                if search(key, headline[i].lower()):
                    indices[key] = i

        indices = list(indices.values())

        for line in file:
            arr = line.split(',')
            arr.pop()
            raw_data.append(arr)

    # str to float
    for i in range(len(raw_data)):
        for j in range(len(raw_data[0])):
            if raw_data[i][j] == 'None':
                continue
            raw_data[i][j] = float(raw_data[i][j])

    # remove None in first line
    for i in range(len(raw_data[0])):
        j = 0
        while raw_data[j][i] == 'None':
            j += 1

            if j == len(raw_data) - 1:
                break

            for k in range(0, j):
                raw_data[k][i] = raw_data[j][i]

    # remove another None
    for i in range(1, len(raw_data)):
        for j in range(1, len(raw_data[0])):
            if raw_data[i][j] == 'None':
                k = 1
                while i+k < len(raw_data) and raw_data[i+k][j] == 'None':
                    k += 1

                if i+k >= len(raw_data):
                    raw_data[i][j] = raw_data[i-1][j]
                else:
                    dx = raw_data[i+k][j] - raw_data[i-1][j]
                    dt = raw_data[i+k][0] - raw_data[i-1][0]
                    dx_dt = dx / dt
                    t0 = raw_data[i-1][0]

                    for g in range(1, k+1):
                        raw_data[i+g-1][j] = round(raw_data[i-1][j] + (raw_data[i+g-1][0] - t0) * dx_dt, 5)

    tmp_data = deepcopy(raw_data)

    # time
    tmp_data[0][0] = 0.0
    turn = 0
    for i in range(1, len(tmp_data)):
        if raw_data[i-1][0] > raw_data[i][0]:
            turn += 1

        tmp_data[i][0] = round( ( float(raw_data[i][0]) + turn * 65535 - float(raw_data[0][0]) ) * 0.001, 4)

    # reverse data, remove unused columns
    data = []
    for i in indices:
        if i not in indices:
            continue
        line = []
        for j in range(len(tmp_data)):
            line.append(tmp_data[j][i])
        data.append(np.array(line))

    # data = np.array(data)

    return data
