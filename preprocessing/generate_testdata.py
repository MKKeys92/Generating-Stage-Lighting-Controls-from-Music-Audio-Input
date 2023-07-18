import os
import utils.logger
from utils.functional import load_data
from utils.util import dotdict
import json
import argparse
import numpy as np
from datetime import datetime
import pickle
import yaml
from preprocessing.audio_extractor import extract_acoustic_feature
import pyhocon
from preprocessing.audio_extractor import FeatureExtractor
import librosa
from evaluation.evaluator import AttributeMasks
from evaluation.data_generator import DataGenerator



class RawDataGenerator:
    def __init__(self, config):
        self.config = config
        self.config.lighting_dim = self.count_light_dim()
        self.create_attri_table()
        self.create_attri_table()
        self.AttrMasks = AttributeMasks(self.attri, self.config.lighting_dim)
        self.data_generator = DataGenerator(vars(self.config), self.AttrMasks, self.config)

    def count_light_dim(self):
        d = self.config.lighting_config
        count = 0
        for k, v in d.items():
            g = k
            for di in v:
                for name, idx in di.items():
                    count += 1

        return count

    def create_attri_table(self):
        d = self.config.lighting_config
        n = self.config.lighting_dim
        self.attri = np.empty(shape=(n,2), dtype = object)
        for k, v in d.items():
            g = k
            for di in v:
                for name, idx in di.items():
                    self.attri[int(idx)-1][0] = g
                    self.attri[int(idx) - 1][1] = name

    def generate(self):
        fnames = sorted(os.listdir(self.config.input_audio_dir))

        extractor  = FeatureExtractor

        for audio_fname in fnames:
            if not audio_fname.endswith((".wav", ".m4a", ".mp3", ".wave")):
                continue
            audio_file = os.path.join(self.config.input_audio_dir, audio_fname)
            print(f'Process -> {audio_file}')
            ### load audio ###

            audio = librosa.load(audio_file, sr=self.config.sampling_rate)[0]
            audio_harmonic, audio_percussive = extractor.get_hpss(audio, vars(self.config))
            onset_env = extractor.get_onset_strength(audio_percussive, vars(self.config))
            beats = extractor.get_onset_beat(onset_env, vars(self.config)).flatten()

            name = audio_fname[:-4]
            T = beats.size
            light = np.zeros((T, self.config.lighting_dim))

            light = self.data_generator.generate_or_modify_data(light, audio, beats=beats)

            sample_dict = {
                'id': name,
                'lighting_array': light.tolist(),
            }
            path = os.path.join(self.config.output_dir, name + '.json')
            with open(path, 'w') as fp:
                json.dump(sample_dict, fp)
        par_path = os.path.abspath(os.path.join(self.config.output_dir, os.pardir))
        ppar_path = os.path.abspath(os.path.join(par_path, os.pardir))
        path = os.path.join(ppar_path, 'generation_config' + '.json')
        with open(path, 'w') as fp:
            json.dump(vars(self.config), fp)



def main(config):
    """ Main function """


    pre_config = pyhocon.ConfigFactory.parse_file("prepro.conf")['base']
    with open(config.light_config_dir + "/" + "lighting_jsons_config.yaml", "r") as stream:
        try:
            config.lighting_config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    config.sampling_rate = pre_config.sampling_rate
    config.hop_length = pre_config.hop_length
    config.window_size = pre_config.window_size

    generator = RawDataGenerator(config)
    generator.generate()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--light_config_dir', type=str,
                        default='raw_generated',
                        help='where light config is saved')

    parser.add_argument('--input_audio_dir', type=str,
                        default='raw_generated_test/GeneratedMichael/audio',
                        help='directory of audios to use for generation')

    parser.add_argument('--output_dir', type=str,
                        default='raw_generated_test/GeneratedMichael/light',
                        help='output_directory')

    parser.add_argument('--base_gen', type=str, default='random',
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


    config = parser.parse_args()

    main(config)