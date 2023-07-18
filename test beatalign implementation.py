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

    parser.add_argument('--plot_beat_align', type=bool, default=True,
                        help='if beat align function graphs should be plotted')

    base_args = parser.parse_args()

    d = {}
    config = dotdict(d)
    config.data_dir = os.path.join("preprocessing","datasets",base_args.dataset)
    config.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
    config.name = "test_beatalign_implementation"
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

    evaluator.feedback.reset()

    N = len(evaluator.test_data)

    evaluator.set_and_activate_base_generator('offbeat')

    mtr = evaluator.evaluate_beatalign(None)

    utils.logger.log(mtr)



if __name__ == '__main__':
    main()