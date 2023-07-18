from SSMCalculator import SSMCalculator
import json
import numpy as np
from matplotlib import pyplot as plt
import libfmp.b
from scipy.signal import find_peaks
import librosa
import os
import pyhocon

config = pyhocon.ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), os.pardir, 'experiments.conf'))['base']
pc = pyhocon.ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), os.pardir, 'preprocessing/prepro.conf'))['base']

config['prepro_config'] = pc

SMMC = SSMCalculator(config)

path_light = os.path.join(os.path.dirname(__file__), os.pardir, 'new_data/Lighting_JSONs/User1/ID08MR02_features_v01_1800_high_v01.json')
path_audio = os.path.join(os.path.dirname(__file__), os.pardir, 'new_data/Audio_RAW_Files/User1/ID08MR02_1800.m4a')

if __name__ == '__main__':
    with open(path_light) as f:
        sample_dict = json.loads(f.read())
    np_light_seq = np.array(sample_dict['lighting_array'])
    x_light, SSM_light = SMMC.calc_light_novelty(np_light_seq)
    x, Fs = librosa.load(path_audio, 15360)

    x_audio, SSM_audio = SMMC.calc_audio_novelty(x)
    cmap = 'gray_r'

    l_peaks = find_peaks(x_light, distance=15)
    a_peaks = find_peaks(x_audio, distance=15)

    p, r, f = SMMC.eval_all_SSM_metrics(np_light_seq, x)

    libfmp.b.plot_matrix(SSM_audio, Fs=1, cmap=cmap,
                         title='SSM audio', xlabel='Time (frames)', ylabel='Time (frames)', colorbar=True);
    fig, ax, line = libfmp.b.plot_signal(x_audio, Fs=3, color='k', title='Nov audio')
    libfmp.b.plot_matrix(SSM_light, Fs=1, cmap=cmap,
                         title='SSM light', xlabel='Time (frames)', ylabel='Time (frames)', colorbar=True);
    fig, ax, line = libfmp.b.plot_signal(x_light, Fs=3, color='k', title='Nov light')
    print(l_peaks)
    print(a_peaks)
    plt.show()
