# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the training process. """
import argparse
import run


def main():
    """ Main function """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str, default=None,
                        help='the directory of train data')

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    parser.add_argument('--saved_model', type=str, default=None,
                        help='which saved_model to load')
    parser.add_argument('--model_directory', type=str, default='checkpoints',
                        help='where to load model from')

    parser.add_argument('--used_startup_frames', type=int, default=0,
                        help='how many frames to use from target at the start in evaluation')

    base_args = parser.parse_args()

    runner = run.Runner(base_args.model_config, base_args.gpu_id)
    model = runner.initialize_model(base_args)

    runner.train(model)

if __name__ == '__main__':
    main()
