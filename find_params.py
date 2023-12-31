import os
import utils.logger
from utils.functional import load_data
from utils.util import dotdict
import json
import argparse
import numpy as np
from evaluation.evaluator import LightEvaluator
from datetime import datetime
from matplotlib import pyplot as plt
from pathlib import Path

# to evaluate a dataset by itself to get insights in evaluation scores

# def sweep_beat_align_boundaries (evaluator):
#     evaluator.reset_all_generators()
#     fig, ax = plt.subplots(nrows=2, sharex=True)
#
#     r = range(15)
#
#     x = [i / 30 for i in r]
#     y_max = []
#     y_min = []
#     y_max_rand = []
#     y_min_rand = []
#     y_max_offbeat = []
#     y_min_offbeat = []
#
#     for i in r:
#         evaluator.set_beataling_frame_boundary(i)
#         mtr = evaluator.evaluate_beatalign(None)
#         y_max.append(mtr['Max Beat Align Score'])
#         y_min.append(mtr['Min Beat Align Score'])
#
#     evaluator.set_and_activate_base_generator('random')
#
#     for i in r:
#         evaluator.set_beataling_frame_boundary(i)
#         mtr = evaluator.evaluate_beatalign(None)
#         y_max_rand.append(mtr['Max Beat Align Score'])
#         y_min_rand.append(mtr['Min Beat Align Score'])
#
#     evaluator.reset_all_generators()
#     evaluator.set_and_activate_dim_generator('offbeat')
#
#     for i in r:
#         evaluator.set_beataling_frame_boundary(i)
#         mtr = evaluator.evaluate_beatalign(None)
#         y_max_offbeat.append(mtr['Max Beat Align Score'])
#         y_min_offbeat.append(mtr['Min Beat Align Score'])
#
#     ax[0].plot(x, y_max, label='Light', color='r')
#
#     ax[0].plot(x, y_max_rand, label='Random', color='b')
#
#     ax[0].plot(x, y_max_offbeat, label='Offbeat', color='g')
#
#     ax[1].set_title('Max Beat Align')
#
#     ax[1].plot(x, y_min, label='Light', color='r')
#
#     ax[1].plot(x, y_min_rand, label='Random', color='b')
#
#     ax[1].plot(x, y_min_offbeat, label='Offbeat', color='g')
#
#     ax[0].set_title('Max Beat Align')
#     ax[1].set_title('Min Beat Align')
#     ax[1].set_xlabel('Seconds')
#
#     fig.suptitle('Sweep over Beat Align Boundary Size', fontsize=16)
#     ax[0].legend(loc="upper left")
#     ax[1].legend(loc="upper left")
#
#     plt.show()

def sweep_beat_align_sigma (evaluator):
    evaluator.reset_all_generators()
    fig, ax = plt.subplots(nrows=2, sharex=True)

    r = np.arange(0.1,2,0.1)

    x = r
    y_max = []
    y_min = []
    y_max_rand = []
    y_min_rand = []
    y_max_offbeat = []
    y_min_offbeat = []
    y_max_onbeat = []
    y_min_onbeat = []

    for i in r:
        evaluator.set_beataling_sigma(i)
        mtr = evaluator.evaluate_beatalign(None)
        y_max.append(mtr['Max Beat Align Score'])
        y_min.append(mtr['Min Beat Align Score'])

    evaluator.set_and_activate_base_generator('random')

    for i in r:
        evaluator.set_beataling_sigma(i)
        mtr = evaluator.evaluate_beatalign(None)
        y_max_rand.append(mtr['Max Beat Align Score'])
        y_min_rand.append(mtr['Min Beat Align Score'])

    evaluator.reset_all_generators()
    evaluator.set_and_activate_dim_generator('onbeat')

    for i in r:
        evaluator.set_beataling_sigma(i)
        mtr = evaluator.evaluate_beatalign(None)
        y_max_onbeat.append(mtr['Max Beat Align Score'])
        y_min_onbeat.append(mtr['Min Beat Align Score'])


    evaluator.reset_all_generators()
    evaluator.set_and_activate_dim_generator('offbeat')

    for i in r:
        evaluator.set_beataling_sigma(i)
        mtr = evaluator.evaluate_beatalign(None)
        y_max_offbeat.append(mtr['Max Beat Align Score'])
        y_min_offbeat.append(mtr['Min Beat Align Score'])

    ax[0].plot(x, y_max, label='real', color='r')

    ax[0].plot(x, y_max_rand, label='random', color='b')

    ax[0].plot(x, y_max_offbeat, label='offbeat', color='g')

    ax[0].plot(x, y_max_onbeat, label='onbeat', color='y')

    ax[1].plot(x, y_min, label='real', color='r')

    ax[1].plot(x, y_min_rand, label='random', color='b')

    ax[1].plot(x, y_min_offbeat, label='offbeat', color='g')

    ax[1].plot(x, y_min_onbeat, label='onbeat', color='y')

    ax[1].set_xlabel('$\sigma$ (frames)')
    ax[0].set_ylabel('$\Gamma_\mathrm{beat \leftrightarrow peak}$')
    ax[1].set_ylabel('$\Gamma_\mathrm{beat \leftrightarrow valley}$')
    ax[0].set_ylim(0,1.05)
    ax[1].set_ylim(0, 1.05)

    fig.suptitle('$\Gamma_\mathrm{beat \leftrightarrow peak}$ and $\Gamma_\mathrm{beat \leftrightarrow valley}$ for different $\sigma$', fontsize=16)
    ax[0].legend(loc="upper left")
    ax[1].legend(loc="upper left")

    plt.show()

def sweep_rms_corr_window (evaluator):
    evaluator.reset_all_generators()
    fig, ax = plt.subplots()

    r = [1,5,10,30,60,90,120,150,180]
    x = [i/30 for i in r]


    y = []
    y_rand = []

    for i in r:
        evaluator.set_rms_window_size(i)
        mtr = evaluator.evaluate_rms(None)
        y.append(mtr['RMS'])

    evaluator.set_and_activate_base_generator('random')

    for i in r:
        evaluator.set_rms_window_size(i)
        mtr = evaluator.evaluate_rms(None)
        y_rand.append(mtr['RMS'])

    ax.plot(x, y, label='real', color='r')

    ax.plot(x, y_rand, label='random', color='b')

    fig.suptitle('$\Gamma_\mathrm{loud \leftrightarrow bright}$ for different window sizes', fontsize=16)
    ax.legend(loc="upper left")
    ax.set_xlabel('seconds')

    plt.show()

def sweep_onset_corr_window (evaluator):
    evaluator.reset_all_generators()
    fig, ax = plt.subplots()

    r = [1,5,10,30,60,90,120,150,180]
    x = [i/30 for i in r]
    y = []
    y_rand = []

    for i in r:
        evaluator.set_onset_window_size(i)
        mtr = evaluator.evaluate_onset(None)
        y.append(mtr['Onset_Score'])

    evaluator.set_and_activate_base_generator('random')

    for i in r:
        evaluator.set_onset_window_size(i)
        mtr = evaluator.evaluate_onset(None)
        y_rand.append(mtr['Onset_Score'])

    ax.plot(x, y, label='real', color='r')

    ax.plot(x, y_rand, label='random', color='b')

    fig.suptitle('$\Gamma_\mathrm{change}$ for different window sizes', fontsize=16)
    ax.legend(loc="upper left")
    ax.set_xlabel('seconds')

    plt.show()

def sweep_novelty_detection_window (evaluator):
    evaluator.reset_all_generators()

    fig, ax = plt.subplots()

    r = range(1,20)
    r = [i*0.5 for i in r]

    x = r
    y = []
    y_rand = []

    for i in r:
        utils.logger.log('[Info] r set to' + str(i))
        evaluator.SMMC.set_mir_segment_window(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y.append(mtr['F_Nov'])

    evaluator.set_and_activate_base_generator('random')

    for i in r:
        utils.logger.log('[Info] r set to' + str(i))
        evaluator.SMMC.set_mir_segment_window(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y_rand.append(mtr['F_Nov'])

    ax.plot(x, y, label='real', color='r')

    ax.plot(x, y_rand, label='random', color='b')

    fig.suptitle('$\Gamma_\mathrm{boundary}$ for different detection window sizes', fontsize=16)
    ax.legend(loc="upper left")
    ax.set_xlabel('seconds')

    plt.show()

def sweep_novelty_kernel_size(evaluator):
    evaluator.reset_all_generators()

    r = range(11,121,10)

    x = r
    y_f_nov = []
    y_f_nov_rand = []
    y_nov_corr = []
    y_nov_corr_rand = []

    for i in r:
        utils.logger.log('[Info] r set to' + str(i))
        evaluator.SMMC.set_novelty_kernel_size(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y_f_nov.append(mtr['F_Nov'])
        y_nov_corr.append(mtr['Nov_Corr'])

    evaluator.set_and_activate_base_generator('random')

    for i in r:
        utils.logger.log('[Info] r set to' + str(i))
        evaluator.SMMC.set_novelty_kernel_size(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y_f_nov_rand.append(mtr['F_Nov'])
        y_nov_corr_rand.append(mtr['Nov_Corr'])

    # fig, ax = plt.subplots()
    # ax.plot(x, y_f_nov, label='real', color='r')
    #
    # ax.plot(x, y_f_nov_rand, label='random', color='b')
    #
    # fig.suptitle('Sweep over nov kernel size for nov f1 score', fontsize=16)
    # ax.legend(loc="upper left")
    # ax.set_xlabel('seconds')
    #
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y_nov_corr, label='real', color='r')

    ax.plot(x, y_nov_corr_rand, label='random', color='b')

    fig.suptitle('$\Gamma_\mathrm{novelty}$ for different kernel filter sizes', fontsize=16)
    ax.legend(loc="upper left")
    ax.set_xlabel('frames')

    plt.show()

def sweep_ssm_smoothing_filter_size(evaluator):
    evaluator.reset_all_generators()

    r = range(11,301,10)

    x = r
    y_f_nov = []
    y_f_nov_rand = []
    y_nov_corr = []
    y_nov_corr_rand = []
    y_ssm_corr = []
    y_ssm_corr_rand =[]

    for i in r:
        evaluator.SMMC.set_smoothing_filter_size(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y_f_nov.append(mtr['F_Nov'])
        y_nov_corr.append(mtr['Nov_Corr'])
        y_ssm_corr.append(mtr['SSM_Corr'])

    evaluator.set_and_activate_base_generator('random')

    for i in r:
        evaluator.SMMC.set_smoothing_filter_size(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y_f_nov_rand.append(mtr['F_Nov'])
        y_nov_corr_rand.append(mtr['Nov_Corr'])
        y_ssm_corr_rand.append(mtr['SSM_Corr'])

    # fig, ax = plt.subplots()
    # ax.plot(x, y_f_nov, label='real', color='r')
    #
    # ax.plot(x, y_f_nov_rand, label='random', color='b')
    #
    # fig.suptitle('Sweep over ssm smoothing filter size for nov f1 score', fontsize=16)
    # ax.legend(loc="upper left")
    # ax.set_xlabel('seconds')
    #
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(x, y_nov_corr, label='real', color='r')
    #
    # ax.plot(x, y_nov_corr_rand, label='random', color='b')
    #
    # fig.suptitle('Sweep over ssm smoothing filter size for nov corr score', fontsize=16)
    # ax.legend(loc="upper left")
    # ax.set_xlabel('seconds')
    #
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y_ssm_corr, label='real', color='r')

    ax.plot(x, y_ssm_corr_rand, label='random', color='b')

    fig.suptitle('$\Gamma_\mathrm{structure}$ for different smoothing filter sizes', fontsize=16)
    ax.legend(loc="upper left")
    ax.set_xlabel('frames')

    plt.show()

def sweep_ssm_downsampling_size(evaluator):
    evaluator.reset_all_generators()

    r = range(5, 60, 5)

    x = r
    y_f_nov = []
    y_f_nov_rand = []
    y_nov_corr = []
    y_nov_corr_rand = []
    y_ssm_corr = []
    y_ssm_corr_rand = []

    for i in r:
        utils.logger.log('[Info] r set to' + str(i))
        evaluator.SMMC.set_downsampling_size(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y_f_nov.append(mtr['F_Nov'])
        y_nov_corr.append(mtr['Nov_Corr'])
        y_ssm_corr.append(mtr['SSM_Corr'])

    evaluator.set_and_activate_base_generator('random')

    for i in r:
        utils.logger.log('[Info] r set to' + str(i))
        evaluator.SMMC.set_downsampling_size(i)
        mtr = evaluator.evaluate_all_ssm_metrics(None)
        y_f_nov_rand.append(mtr['F_Nov'])
        y_nov_corr_rand.append(mtr['Nov_Corr'])
        y_ssm_corr_rand.append(mtr['SSM_Corr'])

    # fig, ax = plt.subplots()
    # ax.plot(x, y_f_nov, label='real', color='r')
    #
    # ax.plot(x, y_f_nov_rand, label='random', color='b')
    #
    # fig.suptitle('Sweep over ssm downsampling size for nov f1 score', fontsize=16)
    # ax.legend(loc="upper left")
    # ax.set_xlabel('seconds')
    #
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.plot(x, y_nov_corr, label='real', color='r')
    #
    # ax.plot(x, y_nov_corr_rand, label='random', color='b')
    #
    # fig.suptitle('Sweep over ssm downsampling size for nov corr score', fontsize=16)
    # ax.legend(loc="upper left")
    # ax.set_xlabel('seconds')
    #
    # plt.show()

    fig, ax = plt.subplots()
    ax.plot(x, y_ssm_corr, label='real', color='r')

    ax.plot(x, y_ssm_corr_rand, label='random', color='b')

    fig.suptitle('$\Gamma_\mathrm{structure}$ for different downsampling factors', fontsize=16)
    ax.legend(loc="upper left")
    ax.set_xlabel('frames')

    plt.show()


def main():
    """ Main function """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    parser.add_argument('--dataset', type=str, required=True,
                        help='name of dataset')

    parser.add_argument('--path_plots', type=str, default='plots',
                        help='where to safe plots to')

    parser.add_argument('--base_gen', type=str, default='',
                        help='what generator to use as base for all values (zero,random, onset, novelty)')

    parser.add_argument('--dim_gen', type=str,
                        default='',
                        help='what dimmer generator to use (random, onbeat, rms, onset)')

    parser.add_argument('--color_gen', type=str,
                        default='',
                        help='what color generator to use (random, onset)')

    parser.add_argument('--pos_gen', type=str,
                        default='',
                        help='what position generator to use (random, onset)')

    base_args = parser.parse_args()

    d = {}
    config = dotdict(d)
    config.data_dir = os.path.join("preprocessing","datasets",base_args.dataset)
    config.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
    config.name = "dataset_evaluation"
    config.log_dir = "log/"
    base_args.is_data_set_eval = True

    prepro_conf_path = os.path.join(config.data_dir, 'preprocess_config.json')
    with open(prepro_conf_path, 'r') as fp:
        config.prepro_config = json.load(fp)

    train_data, test_data, val_data = load_data(config)

    data = np.concatenate((train_data,test_data,val_data))



    utils.logger.init(config)
    utils.logger.log('[Info] Preparing Evaluation Data')

    evaluator = LightEvaluator(None, None, config, data, base_args, is_dataset_eval=True)

    utils.logger.log('[Info] Starting Evaluation')

    #sweep_beat_align_boundaries(evaluator) # depcrecated
    #sweep_beat_align_sigma(evaluator)
    #sweep_rms_corr_window(evaluator)
    #sweep_onset_corr_window(evaluator)
    sweep_novelty_detection_window(evaluator)
    #sweep_novelty_kernel_size(evaluator)
    #sweep_ssm_smoothing_filter_size(evaluator)
    #sweep_ssm_downsampling_size(evaluator)

if __name__ == '__main__':
    main()