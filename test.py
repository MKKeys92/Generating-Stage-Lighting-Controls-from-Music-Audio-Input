# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


""" This script handling the test process. """
import argparse
import run


def main():
    """ Main function """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    parser.add_argument('--saved_model', type=str, required=True,
                        help='which saved_model to load')

    parser.add_argument('--model_directory', type=str, default='checkpoints',
                        help='where to load model from')

    parser.add_argument('--used_startup_frames', type=int, default=0,
                        help='how many frames to use from target at the start')

    parser.add_argument('--path_plots', type=str, default='plots',
                        help='where to safe plots to')

    parser.add_argument('--plot_nov', type=bool, default=False,
                        help='if novelty function graphs should be plotted')

    parser.add_argument('--plot_beat_align', type=bool, default=False,
                        help='if beat align function graphs should be plotted')

    parser.add_argument('--plot_rms_dtw', type=bool, default=False,
                        help='if rms dtw graphs should be plotted')

    parser.add_argument('--plot_onset_env', type=bool, default=False,
                        help='if onset env graphs should be plotted')

    base_args = parser.parse_args()

    runner = run.Runner(None, gpu_id=base_args.gpu_id)

    model = runner.initialize_model(base_args)

    runner.evaluate(model)

if __name__ == '__main__':
    main()