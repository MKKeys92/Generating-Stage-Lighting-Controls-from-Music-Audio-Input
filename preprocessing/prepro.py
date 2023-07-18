# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this open-source project.


import os
import json
import random
import argparse
import numpy as np
from preprocessing.audio_extractor import extract_acoustic_feature
import pyhocon
from datetime import datetime
import pickle
import yaml
from evaluation.evaluator import AttributeMasks
from evaluation.data_generator import DataGenerator


def load_light_data(light_dir):
    print('---------- Loading light data ----------')
    light_sequences = []
    fnames = sorted(os.listdir(light_dir))
    # dir_names = dir_names[:20]  # for debug
    print(f'light seq file names: {fnames}')
    # fnames = fnames[:60]  # For debug
    for fname in fnames:
        path = os.path.join(light_dir, fname)
        with open(path) as f:
            sample_dict = json.loads(f.read())
            np_light_seq = np.array(sample_dict['lighting_array'])
            light_sequences.append(np_light_seq)

    return light_sequences


def align(music, light_sequence, waveform, config):
    print('---------- Align the frames of music and dance ----------')

    min_seq_len = min(len(music), len(light_sequence))

    #assert len(music)>= len(light_sequence), "music should never be shorter than light seq"

    music = np.array(music[:min_seq_len])
    waveform = np.array(waveform[:(min_seq_len*config.hop_length-1)])
    light_sequence = light_sequence[:min_seq_len, :]

    return music, light_sequence, waveform


def split_data(data, args):
    print('---------- Calculating split indices----------')

    assert args.test_split + args.train_split + args.val_split == 1, "Split config not supported. sum of split values not equal to 1!"

    indices = list(range(len(data)))
    random.shuffle(indices)
    n = len(indices)
    train_idx = indices[:int(n * args.train_split)]
    val_idx = indices[int(n * args.train_split) : int(n * args.train_split) + int(n * args.val_split)]
    test_idx = indices[int(n * args.train_split) + int(n * args.val_split):]

    return train_idx, test_idx, val_idx


def search_for_music(musics, names, light_name):
    for i in range(len(names)):
        if light_name in names[i]:
            return i

    return None


def create_dataset(config):


    with open(config.input_dir + "/" + "lighting_jsons_config.yaml", "r") as stream:
        try:
            config['lighting_config'] = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    use_data_gen = False
    if config.dim_gen or config.color_gen or config.base_gen or config.pos_gen:
        use_data_gen = True

    AMasks = None
    DataGen = None

    dataset = {}
    data = []

    dirs = os.listdir(config.input_dir)

    for sub_dir in dirs:
        if not os.path.isdir(config.input_dir + "/" + sub_dir):
            continue
        print(sub_dir)
        print('---- extracting audio features -----')
        musics, anames, waveforms = extract_acoustic_feature(os.path.join(config.input_dir, sub_dir, 'audio'), config)
        print('---- extracting light features -----')
        light_seqs = load_light_data(os.path.join(config.input_dir, sub_dir, 'light'))

        fnames = sorted(os.listdir(os.path.join(config.input_dir, sub_dir, 'light')))

        for idx in range(len(light_seqs)):
            name = fnames[idx]

            if name.endswith('.json'):
                name = name[:-5]

            m_idx = search_for_music(musics, anames, name.split('_')[0])

            assert (m_idx is not None)

            music = musics[m_idx]
            light_seq = light_seqs[idx]

            waveform = waveforms[m_idx]

            final_music, final_light_seq, final_waveform= align(music, light_seq, waveform, config)

            if use_data_gen:
                if AMasks is None:
                    config['lighting_dim'] = len(light_seq[0])
                    AMasks = AttributeMasks(config['lighting_config'], config['lighting_dim'])
                    DataGen = DataGenerator(config, AMasks, config)
                final_light_seq = DataGen.generate_or_modify_data(final_light_seq, final_waveform)

            sample_dict = {
                'id': name,
                'music_array': final_music,
                'lighting_array': final_light_seq,
                'waveform': final_waveform,
            }
            data.append(sample_dict)

    train_idx, test_idx, val_idx = split_data(data, config)
    train_idx = sorted(train_idx)
    print(f'train ids: {[idx for idx in train_idx]}')
    test_idx = sorted(test_idx)
    print(f'test ids: {[idx for idx in test_idx]}')
    val_idx = sorted(val_idx)
    print(f'test ids: {[idx for idx in val_idx]}')

    config['music_dim'] = len(data[0]['music_array'][0])
    config['lighting_dim'] = len(data[0]['lighting_array'][0])

    dataset['preprocess_config'] = config
    dataset['train_data'] = []
    dataset['test_data'] = []
    dataset['val_data'] = []

    for idx in range(len(data)):
        if idx in train_idx:
            dataset['train_data'].append(data[idx])
        if idx in test_idx:
            dataset['test_data'].append(data[idx])
        if idx in val_idx:
            dataset['val_data'].append(data[idx])

    date_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
    dataset_name = config.name + '_' + date_suffix

    path = os.path.join("datasets", dataset_name)
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    data_path = os.path.join(path,"data.p")
    config_path = os.path.join(path,"preprocess_config.json")
    with open(data_path,"wb") as fp:
        pickle.dump(dataset, fp)
    with open(config_path, 'w') as fp:
        json.dump(config, fp)

    if os.path.isfile(config.input_dir + "/" + "generation_config.json"):
        generation_config_path = os.path.join(path,"generation_config.json")
        with open(config.input_dir + "/" + "generation_config.json") as f:
            sample_dict = json.loads(f.read())
            with open(generation_config_path, 'w') as fp:
                json.dump(sample_dict, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='base')
    parser.add_argument('--input_dir', type=str, default='raw_data')

    base_args = parser.parse_args()

    config = pyhocon.ConfigFactory.parse_file("prepro.conf")[base_args.config]
    config.name = base_args.config
    config.input_dir = base_args.input_dir

    if not os.path.exists("datasets"):
        os.makedirs("datasets", exist_ok=True)

    create_dataset(config)

