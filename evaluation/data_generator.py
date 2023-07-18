import numpy as np

from preprocessing.audio_extractor import FeatureExtractor
import utils.logger
import librosa
import librosa.display

class DataGenerator:
    def __init__(self, prepro_config, AttriMasks, args_gen):
        self.prepro_config = prepro_config
        self.AttriMasks = AttriMasks
        self.args_gen = args_gen

    def onbeat_sawtooth(self,beats):
        beats_idx = np.argwhere(beats == 1)
        curve = beats.copy()
        last_b_idx = 0
        l = beats.shape[0]
        for b_idx in beats_idx:
            b_idx = b_idx[0]
            d = b_idx - last_b_idx
            if (last_b_idx) == 0:
                curve[0:b_idx + 1] = np.linspace(0, 1, num=d + 1)
            else:
                d_2 = int(d / 2)
                d_half = last_b_idx + d_2
                curve[last_b_idx:d_half + 1] = np.linspace(1, 0, num=d_2 + 1)
                curve[d_half: b_idx + 1] = np.linspace(0, 1, num=d - d_2 + 1)
            last_b_idx = b_idx
        if last_b_idx != l:
            curve[last_b_idx:] = np.linspace(1, 0, num=l - last_b_idx)

        return curve

    def offbeat_sawtooth(self,beats):
        curve = self.onbeat_sawtooth(beats)
        ones = np.ones(curve.shape)
        curve = ones - curve
        return curve

    def get_onset_curve(self,onset):
        curve = onset
        for i in range(1,len(curve)):
            curve[i] = curve[i] + curve[i-1]
        curve /= curve[-1]
        return curve


    def count_light_dim(self):
        d = self.prepro_config.lighting_config
        count = 0
        for k, v in d.items():
            g = k
            for di in v:
                for name, idx in di.items():
                    count += 1

        return count

    def set_base_gen(self, val):
        self.args_gen.base_gen = val

    def set_dim_gen(self,val):
        self.args_gen.dim_gen = val

    def set_col_gen(self,val):
        self.args_gen.color_gen = val

    def set_pos_gen(self,val):
        self.args_gen.pos_gen = val

    def generate_or_modify_data(self, r_in, audio, **kwargs):
        extractor = FeatureExtractor()

        beats = None
        if 'beats' in kwargs.keys():
            beats = kwargs['beats'].copy().flatten()

        onset = None
        if 'onset' in kwargs.keys():
            onset = kwargs['onset']

        light = r_in

        if self.args_gen.base_gen == "random":
            light = np.random.rand(r_in.size)
            light = light.reshape(r_in.shape)
        elif self.args_gen.base_gen == "onset":
            if onset is None:
                onset = extractor.get_onset_strength(audio, self.prepro_config)
            curve = self.get_onset_curve(onset)
            light = np.tile(curve, (r_in.shape[1], 1)).transpose()
        elif self.args_gen.base_gen == "novelty":
            C = librosa.feature.chroma_stft(y=audio, sr=self.prepro_config['sampling_rate'], hop_length = self.prepro_config['hop_length'])
            light = np.tile(C,(int(r_in.shape[1] / C.shape[0]),1)).transpose()
        elif self.args_gen.base_gen == "zeros":
            light = np.zeros(r_in.shape)
        elif self.args_gen.base_gen == "static":
            light = np.ones(r_in.shape)
        elif self.args_gen.base_gen == "rms":
            rms = extractor.get_rms(audio, self.prepro_config)
            rms = rms.flatten()
            normed_rms = rms / np.max(rms)
            light = np.tile(normed_rms, (r_in.shape[1], 1)).transpose()
        elif self.args_gen.base_gen == "onbeat":
            if beats is None:
                audio_harmonic, audio_percussive = extractor.get_hpss(audio, self.prepro_config)
                onset_env = extractor.get_onset_strength(audio_percussive, self.prepro_config)
                beats = extractor.get_onset_beat(onset_env, self.prepro_config).flatten()
            ob_saw = self.onbeat_sawtooth(beats)
            light = np.tile(ob_saw, (r_in.shape[1], 1)).transpose()
        elif self.args_gen.base_gen == "offbeat":
            if beats is None:
                audio_harmonic, audio_percussive = extractor.get_hpss(audio, self.prepro_config)
                onset_env = extractor.get_onset_strength(audio_percussive, self.prepro_config)
                beats = extractor.get_onset_beat(onset_env, self.prepro_config).flatten()
            ob_saw = self.offbeat_sawtooth(beats)
            light = np.tile(ob_saw, (r_in.shape[1], 1)).transpose()


        # dimmer variants
        if self.args_gen.dim_gen == 'onbeat':
            if beats is None:
                audio_harmonic, audio_percussive = extractor.get_hpss(audio, self.prepro_config)
                onset_env = extractor.get_onset_strength(audio_percussive, self.prepro_config)
                beats = extractor.get_onset_beat(onset_env, self.prepro_config).flatten()
            ob_saw = self.onbeat_sawtooth(beats)
            light[:, self.AttriMasks.int_mask == 1] = np.tile(ob_saw, (sum(self.AttriMasks.int_mask), 1)).transpose()
        elif self.args_gen.dim_gen == 'offbeat':
            if beats is None:
                audio_harmonic, audio_percussive = extractor.get_hpss(audio, self.prepro_config)
                onset_env = extractor.get_onset_strength(audio_percussive, self.prepro_config)
                beats = extractor.get_onset_beat(onset_env, self.prepro_config).flatten()
            ob_saw = self.offbeat_sawtooth(beats)
            light[:, self.AttriMasks.int_mask == 1] = np.tile(ob_saw, (sum(self.AttriMasks.int_mask), 1)).transpose()
        elif self.args_gen.dim_gen == 'rms':
            rms = extractor.get_rms(audio, self.prepro_config)
            rms = rms.flatten()
            normed_rms = rms / np.max(rms)
            light[:, self.AttriMasks.int_mask == 1] = np.tile(normed_rms, (sum(self.AttriMasks.int_mask), 1)).transpose()
        elif self.args_gen.dim_gen == 'random':
            light[:, self.AttriMasks.int_mask == 1] = np.random.rand(r_in.shape[0], sum(self.AttriMasks.int_mask))
        elif self.args_gen.pos_gen == 'onset':
            if onset is None:
                onset = extractor.get_onset_strength(audio, self.prepro_config)
            curve = self.get_onset_curve(onset)
            light[:, self.AttriMasks.int_mask == 1] = np.tile(curve, (sum(self.AttriMasks.int_mask), 1)).transpose()

        # position variants
        if self.args_gen.pos_gen == 'onset':
            if onset is None:
                onset = extractor.get_onset_strength(audio, self.prepro_config)
            curve = self.get_onset_curve(onset)
            light[:, self.AttriMasks.pos_mask == 1] = np.tile(curve, (sum(self.AttriMasks.pos_mask), 1)).transpose()
        elif self.args_gen.pos_gen == 'random':
            light[:, self.AttriMasks.pos_mask == 1] = np.random.rand(r_in.shape[0], sum(self.AttriMasks.pos_mask))

        # color variants
        if self.args_gen.color_gen == 'onset':
            if onset is None:
                onset = extractor.get_onset_strength(audio, self.prepro_config)
            curve = self.get_onset_curve(onset)
            light[:, self.AttriMasks.col_mask == 1] = np.tile(curve, (sum(self.AttriMasks.col_mask), 1)).transpose()
        elif self.args_gen.color_gen == 'random':
            light[:, self.AttriMasks.col_mask == 1] = np.random.rand(r_in.shape[0], sum(self.AttriMasks.col_mask))

        return light
