import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
from preprocessing.audio_extractor import FeatureExtractor
import pyhocon
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='base')
parser.add_argument('--input_dir', type=str, default='raw_data')

base_args = parser.parse_args()

config = pyhocon.ConfigFactory.parse_file("prepro.conf")[base_args.config]
config.name = base_args.config
config.input_dir = base_args.input_dir

y, sr = librosa.load("/Users/michaelkohl/PycharmProjects/alternator_v1.1/preprocessing/raw_data_songbased/HAW/audio/ID07HG02.wav", sr=config.sampling_rate)

extractor = FeatureExtractor()

S = extractor.get_melspectrogram(y,config)
librosa.feature.mfcc(S=librosa.power_to_db(S))

mfccs = extractor.get_mfcc(S,config)

chromagram_cqt = extractor.get_chroma_cqt(y,config)
chromagram_sftf = extractor.get_chroma_cqt(y, config)

fig, ax = plt.subplots(nrows=2, sharex=True)
img = librosa.display.specshow(chromagram_sftf, x_axis='time', y_axis='chroma', ax=ax[0])
fig.colorbar(img, ax=[ax[0]])
ax[0].set(title='SFTF chromagram')
ax[0].
img = librosa.display.specshow(chromagram_cqt, x_axis='time', y_axis='chroma', ax=ax[1])
fig.colorbar(img, ax=[ax[1]])
ax[1].set(title='CQT chromagram')
#plt.subplots_adjust(hspace = 0.2)
plt.show()

fig, ax = plt.subplots(nrows=1, sharex=True)
img = librosa.display.specshow(S,
                               x_axis='time', y_axis='mel', fmax=8000,
                               ax=ax)
fig.colorbar(img, ax=[ax])
ax.set(title='Mel spectrogram')
ax.label_outer()

plt.show()

fig, ax = plt.subplots(nrows=1, sharex=True)
img = librosa.display.specshow(mfccs, x_axis='time', ax=ax)
fig.colorbar(img, ax=[ax])
ax.set(title='MFCC', )
plt.ylabel('MFCC index')
plt.yticks(np.arange(4,mfccs.shape[0],5))
ax.set_yticklabels([5,10,15,20])
plt.show()