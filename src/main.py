from prepare_data import prepare_data
from features import get_features, m_get_features
import time
import matplotlib.pyplot as plt
import library_neural_network

if __name__ == '__main__':
    # prepare_data(['/home/nikita/Documents/Diakont/ForceControl/IsobarTest/NGV_without_stroke/',
    #               '/home/nikita/Documents/Diakont/ForceControl/IsobarTest/move_file_other_igv/',
    #               '/home/nikita/Documents/Diakont/ForceControl/IsobarTest/move_file_experiment/'])
    # prepare_data(['/home/nikita/Documents/Diakont/ForceControl/IsobarTest/move_file_other_igv/',
    #               '/home/nikita/Documents/Diakont/ForceControl/IsobarTest/move_file_experiment/'])

    start_time = time.time()

    # get_features()
    # m_get_features()

    library_neural_network.start()

    print('Время: ', time.time() - start_time)

    plt.show()
