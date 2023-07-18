import os
import utils.logger
from utils.functional import load_data
from utils.util import dotdict
import json
import argparse
import numpy as np
from evaluation.evaluator import LightEvaluator
from datetime import datetime

# to evaluate a dataset by itself to get insights in evaluation scores

def main():
    """ Main function """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    parser.add_argument('--dataset', type=str, required=True,
                        help='name of dataset')

    parser.add_argument('--path_plots', type=str, default='plots',
                        help='where to safe plots to')

    parser.add_argument('--plot_nov', type=bool, default=False,
                        help='if novelty function graphs should be plotted')

    parser.add_argument('--plot_beat_align', type=bool, default=False,
                        help='if beat align function graphs should be plotted')

    parser.add_argument('--plot_rms', type=bool, default=False,
                        help='if rms dtw graphs should be plotted')

    parser.add_argument('--plot_onset_env', type=bool, default=False,
                        help='if onset env graphs should be plotted')

    parser.add_argument('--base_gen', type=str, default='',
                        help='what generator to use as base for all values (zero,random, onset, novelty)')

    parser.add_argument('--dim_gen', type=str,
                        default='',
                        help='what dimmer generator to use (random, onbeat, rms, onset, offbeat)')

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
    utils.logger.log(base_args)

    evaluator = LightEvaluator(None, None, config, data, base_args, is_dataset_eval=True)

    utils.logger.log('Starting Evaluation')

    mtr = evaluator.evaluate_dataset()

    utils.logger.log.log_eval(mtr)

if __name__ == '__main__':
    main()