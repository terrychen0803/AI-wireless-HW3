import os
from Train import train
from Test import test
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
class sysconfig(object):
    Pilots = 8        # number of pilots
    with_CP_flag = True 
    SNR = 20
    Clipping = False
    Train_set_path = r'C:\python_projects\wcmlbook\wcmlbook-main\ch3\Exercise_3.1\H_dataset\\'
    Test_set_path = r'C:\python_projects\wcmlbook\wcmlbook-main\ch3\Exercise_3.1\H_dataset\\'
    Model_path = r'C:\python_projects\wcmlbook\wcmlbook-main\ch3\Exercise_3.1\Models\\'
    pred_range = np.arange(16,32)
    learning_rate = 0.001
    learning_rate_decrease_step = 2000     
    

def main():
    snr_list = [5, 10, 15, 20, 25]
    pilot_list = [8, 16, 64]

    for P in pilot_list:
        for snr in snr_list:
            print("=" * 60)
            print("Running QPSK case: SNR =", snr, "dB, Pilots =", P)
            print("=" * 60)

            tf.reset_default_graph()

            config = sysconfig()
            config.Pilots = P
            config.SNR = snr
            config.pred_range = np.arange(16, 32)

            os.makedirs(config.Model_path + 'SNR_' + str(snr), exist_ok=True)

            train(config)
if __name__ == '__main__':
    main()

