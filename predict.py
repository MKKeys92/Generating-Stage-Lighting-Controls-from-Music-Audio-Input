# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.

import argparse
import run

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    parser.add_argument('--audio_dir', type=str, default='predict/input')
    parser.add_argument('--output_dir', type=str, default='predict/output')
    parser.add_argument('--saved_model', type=str, required=True,
                        help='which saved_model to load')
    parser.add_argument('--model_directory', type=str, default='checkpoints',
                        help='where to load model from')
    parser.add_argument('--used_startup_frames', type=int, default=0,
                        help='how many frames to use from target at the start')

    predict_args = parser.parse_args()

    runner = run.Runner(None, predict_args.gpu_id)
    model = runner.initialize_model(predict_args, eval=False)

    runner.predict(model)

if __name__ == '__main__':
    main()