import numpy
import sklearn.metrics as metrics
import numpy as np
from scipy.signal import find_peaks
from evaluation.FIDCalculator import FIDCalculator
from evaluation.SSMCalculator import SSMCalculator
from evaluation.data_generator import DataGenerator
from preprocessing.audio_extractor import FeatureExtractor
import utils.logger
import librosa
import librosa.display
import os
from scipy import signal
from scipy.stats import pearsonr

from matplotlib import pyplot as plt
from scipy.ndimage.interpolation import shift

# def gen_dict_extract(key, var):
#     if hasattr(var,'items'): # hasattr(var,'items') for python 3
#         for k, v in var.items(): # var.items() for python 3
#             if k == key:
#                 yield int(v)- 1
#             if isinstance(v, dict):
#                 for result in gen_dict_extract(key, v):
#                     yield result
#             elif isinstance(v, list):
#                 for d in v:
#                     for result in gen_dict_extract(key, d):
#                         yield result

def set_bp_edge(bp, color):
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=color)
    plt.setp(bp["fliers"], markeredgecolor=color)

def set_bp_fill(bp, color):
    for patch in bp['boxes']:
        patch.set(facecolor=color)

class AttributeMasks:
    def __init__(self, d, n):
        self.N = n
        self.create_attri_table(d)
        self.create_masks()


    def create_attri_table(self, d):
        self.attri = np.empty(shape=(self.N,2), dtype = np.object)
        for k, v in d.items():
            g = k
            for di in v:
                for name, idx in di.items():
                    self.attri[int(idx)-1][0] = g
                    self.attri[int(idx) - 1][1] = name


    def create_masks(self):
        self.int_mask = np.zeros(self.N)
        self.hue_mask = np.zeros(self.N)
        self.sat_mask = np.zeros(self.N)
        self.pan_mask = np.zeros(self.N)
        self.tilt_mask = np.zeros(self.N)
        self.all_val_mask = np.zeros(self.N)
        self.pos_mask = np.zeros(self.N)
        self.col_mask = np.zeros(self.N)

        int_peak = (self.attri[:,1] == 'Intensity of Peak absolute')
        intensity = (self.attri[:,1] == 'Intensity')
        pan_val_l = (self.attri[:,1] == 'Val Max Left Pan')
        pan_val_r = (self.attri[:, 1] == 'Val Max Right Pan')
        tilt_val = (self.attri[:,1] == 'Value of Max Tilt')
        pan_mean = (self.attri[:,1] == 'Pan Mean')
        tilt_mean = (self.attri[:,1] == 'Tilt Mean')
        hue_lead = (self.attri[:,1] == 'Col Hue Lead')
        hue_back = (self.attri[:,1] == 'Col Hue Back')
        hue = (self.attri[:,1] == 'Col Hue')
        sat_lead = (self.attri[:,1] == 'Col Sat Lead')
        sat_back = (self.attri[:,1] == 'Col Sat Back')
        sat = (self.attri[:,1] == 'Col Sat')

        assert(np.sum(int_peak)>0)
        assert (np.sum(intensity) > 0)
        assert (np.sum(pan_val_l) > 0)
        assert (np.sum(pan_val_r) > 0)
        assert (np.sum(tilt_val) > 0)
        assert (np.sum(pan_mean) > 0)
        assert (np.sum(tilt_mean) > 0)
        assert (np.sum(hue_lead) > 0)
        assert (np.sum(hue_back) > 0)
        assert (np.sum(hue) > 0)
        assert (np.sum(sat_lead) > 0)
        assert (np.sum(sat_back) > 0)
        assert (np.sum(sat) > 0)

        self.int_mask = np.ma.mask_or(int_peak, intensity)

        self.hue_mask = np.ma.mask_or(hue,hue_back)
        self.hue_mask = np.ma.mask_or(self.hue_mask, hue_lead)

        self.sat_mask = np.ma.mask_or(sat,sat_back)
        self.sat_mask = np.ma.mask_or(self.sat_mask, sat_lead)

        self.pan_mask = np.ma.mask_or(pan_val_l,pan_mean)
        self.pan_mask = np.ma.mask_or(self.pan_mask, pan_val_r)

        self.tilt_mask = np.ma.mask_or(tilt_val,tilt_mean)

        self.pos_mask = np.ma.mask_or(self.tilt_mask, self.pan_mask)
        self.col_mask = np.ma.mask_or(self.hue_mask, self.sat_mask)

        self.all_val_mask = np.ma.mask_or(self.int_mask,self.hue_mask)
        self.all_val_mask = np.ma.mask_or(self.all_val_mask, self.sat_mask)
        self.all_val_mask = np.ma.mask_or(self.all_val_mask, self.pan_mask)
        self.all_val_mask = np.ma.mask_or(self.all_val_mask, self.tilt_mask)

        return


class LightEvaluatorFeedback:
    def __init__(self):
        self.missing_max_beat = 0
        self.count_max_beat = 0
        self.missing_min_beat = 0
        self.count_min_beat = 0
    def reset(self):
        self.missing_max_beat = 0
        self.count_max_beat = 0
        self.missing_min_beat = 0
        self.count_min_beat = 0

    def print_feedback(self):
        utils.logger.log(
            'eval has not found a max beat peak for '+ str(self.missing_max_beat) + ' out of ' + str(self.count_max_beat))
        utils.logger.log(
            'eval has not found a min beat peak for ' + str(self.missing_min_beat) + ' out of ' + str(self.count_min_beat))

class LightEvaluator:
    def __init__(self, data_types, eval_weight, model_config, test_data, runtime_args, **kwargs):
        self.data_types = data_types
        self.eval_weight = eval_weight
        self.model_config = model_config
        self.movement_score_window_sizes = [2,5,15]
        self.FIDCalculator = FIDCalculator(model_config)
        self.test_data = test_data
        self.rms_filter_size = 100
        self.rms_window_size = 120
        self.onset_window_size = 120
        self.beat_align_frame_boundary = 0 #0 frames next to beat is also perfect
        self.beat_align_sigma = 0.5
        self.beat_align_shift = 3 #shifting peaks, taking best value of +-val
        self.is_dataset_eval = False
        self.use_dataset_generator = False

        # check if all values a normed correctly
        for k in range(len(self.test_data)):
            t = self.test_data[k]['lighting_array']
            min = np.min(t)
            max = np.max(t)
            name = self.test_data[k]['id']
            assert max <= 1, "data is not normed correctly: error for file " + name
            assert min >= 0, "data is not normed correctly: error for file " + name

        if "dim_gen" not in runtime_args:
            runtime_args.dim_gen = ''
        if "color_gen" not in runtime_args:
            runtime_args.color_gen = ''
        if "pos_gen" not in runtime_args:
            runtime_args.pos_gen = ''
        if "base_gen" not in runtime_args:
            runtime_args.base_gen = ''


        if "is_dataset_eval" in kwargs.keys():
            self.is_dataset_eval = kwargs['is_dataset_eval']

        if "data_generator" in runtime_args:
            self.data_generator = runtime_args.data_generator

        if "path_plots" in runtime_args:
            if not os.path.exists(runtime_args.path_plots):
                os.makedirs(runtime_args.path_plots, exist_ok=True)

            if"is_data_set_eval" in runtime_args:
                self.path_plots = os.path.join(runtime_args.path_plots,"dataset_"+runtime_args.dataset )
                if runtime_args.base_gen:
                    self.use_dataset_generator = True
                    self.path_plots += '_base_' + runtime_args.base_gen
                if runtime_args.dim_gen:
                    self.use_dataset_generator = True
                    self.path_plots += '_dim_' + runtime_args.dim_gen
                if runtime_args.color_gen:
                    self.use_dataset_generator = True
                    self.path_plots += '_col_' + runtime_args.color_gen
                if runtime_args.pos_gen:
                    self.use_dataset_generator = True
                    self.path_plots += '_pos_' + runtime_args.pos_gen

            else:
                self.path_plots = os.path.join(runtime_args.path_plots,"model_"+model_config.name+'_'+model_config.loaded_model)

            self.path_nov = os.path.join(self.path_plots,"novelty")
            if not os.path.exists(self.path_nov):
                os.makedirs(self.path_nov, exist_ok=True)

            self.path_ba = os.path.join(self.path_plots, "beatalign")
            if not os.path.exists(self.path_ba):
                os.makedirs(self.path_ba, exist_ok=True)

            self.path_rms = os.path.join(self.path_plots, "rms")
            if not os.path.exists(self.path_rms):
                os.makedirs(self.path_rms, exist_ok=True)

            self.path_onset = os.path.join(self.path_plots, "onset")
            if not os.path.exists(self.path_onset):
                os.makedirs(self.path_onset, exist_ok=True)

        if "plot_nov" in runtime_args:
            self.plot_nov = runtime_args.plot_nov
        else:
            self.plot_nov = False
        if "plot_beat_align" in runtime_args:
            self.plot_beat_align = runtime_args.plot_beat_align
        else:
            self.plot_beat_align = False

        if "plot_rms" in runtime_args:
            self.plot_rms = runtime_args.plot_rms
        else:
            self.plot_rms = False

        if "plot_onset_env" in runtime_args:
            self.plot_onset_env = runtime_args.plot_onset_env
        else:
            self.plot_onset_env = False

        extractor = FeatureExtractor()

        #preparing audio features
        for d in self.test_data:
            w = d['waveform']
            audio_harmonic, audio_percussive = extractor.get_hpss(w, self.model_config.prepro_config)
            onset_env_beat = extractor.get_onset_strength(audio_percussive, self.model_config.prepro_config)
            onset_beat = extractor.get_onset_beat(onset_env_beat, self.model_config.prepro_config)
            d['onset_beat'] = onset_beat
            d['onset_env'] = extractor.get_onset_strength(w, self.model_config.prepro_config)
            onset_env = librosa.onset.onset_strength(y=w, sr=self.model_config.prepro_config['sampling_rate'], hop_length =self.model_config.prepro_config['hop_length'] )
            mel = extractor.get_melspectrogram(w, self.model_config.prepro_config)
            rms = extractor.get_rms(w,self.model_config.prepro_config).flatten()

            # smoothing both data, to get overall behaviour
            # rms = rms.flatten()/rms.max()
            # rms = (rms- np.mean(rms)) / np.std(rms)
            # win = signal.windows.hann(self.rms_filter_size)
            # rms = signal.convolve(rms, win, mode='same') / sum(win)
            d['normed_rms'] = rms

        self.shift_window_beat_algin = 2 #2 frames in every direction
        self.AttrMasks = AttributeMasks(self.model_config.prepro_config['lighting_config'], self.model_config.prepro_config['lighting_dim'])
        self.attri = self.AttrMasks.attri
        self.feedback = LightEvaluatorFeedback()
        self.data_generator = DataGenerator(self.model_config.prepro_config, self.AttrMasks, runtime_args)
        if self.plot_nov:
            self.SMMC = SSMCalculator(self.model_config, runtime_args, self.plot_nov, self.path_nov)
        else:
            self.SMMC = SSMCalculator(self.model_config, runtime_args, self.plot_nov, None)

    def set_beataling_frame_boundary(self, v):
        self.beat_align_frame_boundary = int (v)

    def set_beataling_sigma(self, v):
        self.beat_align_sigma = v

    def set_rms_window_size(self, v):
        self.rms_window_size = int (v)

    def set_onset_window_size(self, v):
        self.onset_window_size = int (v)

    def set_and_activate_base_generator(self, v):
        if v == "None":
            self.data_generator.set_base_gen('')
            self.use_dataset_generator = False
        else:
            self.data_generator.set_base_gen(v)
            self.use_dataset_generator = True

    def set_and_activate_dim_generator(self, v):
        self.data_generator.set_dim_gen(v)
        self.use_dataset_generator = True

    def set_and_activate_pos_generator(self, v):
        self.data_generator.set_pos_gen(v)
        self.use_dataset_generator = True

    def set_and_activate_col_generator(self, v):
        self.data_generator.set_col_gen(v)
        self.use_dataset_generator = True

    def reset_all_generators(self):
        self.data_generator.set_base_gen('')
        self.data_generator.set_pos_gen('')
        self.data_generator.set_col_gen('')
        self.data_generator.set_dim_gen('')
        self.use_dataset_generator = False

    def evaluate_beatalign(self, results):
        self.feedback.reset()
        smin_beat_align_score = 0
        smax_beat_align_score = 0
        nmax_beat_align_score = 0
        nmin_beat_align_score = 0
        N = len(self.test_data)
        mtr = {}

        for j in range(N):
            t = self.test_data[j]['lighting_array'].copy()
            w = self.test_data[j]['waveform'].copy()
            b = self.test_data[j]['onset_beat'].copy()
            rms = self.test_data[j]['normed_rms'].copy()
            onset_env = self.test_data[j]['onset_env'].copy()
            i = self.test_data[j]['music_array'].copy()

            r = t
            if not self.is_dataset_eval: # normal evaluation
                r = results[j].cpu().numpy()
            else:
                if self.use_dataset_generator:
                    r = self.data_generator.generate_or_modify_data(r, w, beats=b, onset=onset_env)

            bs_max, bs_min, nb_max, nb_min = self.calc_beat_align_score(r,b, j)
            smax_beat_align_score += bs_max
            smin_beat_align_score += bs_min
            nmax_beat_align_score += nb_max
            nmin_beat_align_score += nb_min

        if nmax_beat_align_score:
            max_beat_align_score = smax_beat_align_score / nmax_beat_align_score
        else:
            max_beat_align_score = 0
        if nmin_beat_align_score:
            min_beat_align_score = smin_beat_align_score / nmin_beat_align_score
        else:
            min_beat_align_score = 0

        self.feedback.print_feedback()

        mtr['Max Beat Align Score'] = max_beat_align_score
        mtr['Min Beat Align Score'] = min_beat_align_score

        return mtr

    def evaluate_rms(self, results):
        self.feedback.reset()
        N = len(self.test_data)
        mtr = {}
        rms_score = 0

        predictions=[]

        for j in range(N):
            t = self.test_data[j]['lighting_array'].copy()
            w = self.test_data[j]['waveform'].copy()
            b = self.test_data[j]['onset_beat'].copy()
            rms = self.test_data[j]['normed_rms'].copy()
            onset_env = self.test_data[j]['onset_env'].copy()
            i = self.test_data[j]['music_array'].copy()

            r = t
            if not self.is_dataset_eval: # normal evaluation
                r = results[j].cpu().numpy()
                predictions.append(r)
            else:
                if self.use_dataset_generator:
                    r = self.data_generator.generate_or_modify_data(r, w, beats=b, onset=onset_env)

            rms_score += self.get_rms_score(r, rms, j)


        rms_score /= N

        self.feedback.print_feedback()

        mtr['RMS'] = rms_score

        return mtr

    def evaluate_onset(self, results):
        self.feedback.reset()
        N = len(self.test_data)
        mtr = {}
        onset_score = 0

        predictions=[]

        for j in range(N):
            t = self.test_data[j]['lighting_array'].copy()
            w = self.test_data[j]['waveform'].copy()
            b = self.test_data[j]['onset_beat'].copy()
            rms = self.test_data[j]['normed_rms'].copy()
            onset_env = self.test_data[j]['onset_env'].copy()
            i = self.test_data[j]['music_array'].copy()

            r = t
            if not self.is_dataset_eval: # normal evaluation
                r = results[j].cpu().numpy()
                predictions.append(r)
            else:
                if self.use_dataset_generator:
                    r = self.data_generator.generate_or_modify_data(r, w, beats=b, onset=onset_env)

            onset_score += self.get_onset_score(r, onset_env, j)


        onset_score /= N

        self.feedback.print_feedback()

        mtr['Onset_Score'] = onset_score

        return mtr

    def evaluate_all_ssm_metrics(self, results):
        self.feedback.reset()
        N = len(self.test_data)
        mtr = {}
        p_nov, r_nov, f_nov = 0, 0, 0
        ssm_corr_score = 0
        nov_corr_score = 0

        predictions=[]

        for j in range(N):
            t = self.test_data[j]['lighting_array'].copy()
            w = self.test_data[j]['waveform'].copy()
            b = self.test_data[j]['onset_beat'].copy()
            rms = self.test_data[j]['normed_rms'].copy()
            onset_env = self.test_data[j]['onset_env'].copy()
            i = self.test_data[j]['music_array'].copy()

            r = t
            if not self.is_dataset_eval: # normal evaluation
                r = results[j].cpu().numpy()
                predictions.append(r)
            else:
                if self.use_dataset_generator:
                    r = self.data_generator.generate_or_modify_data(r, w, beats=b, onset=onset_env)

            p, re, f, ssm_corr, nov_corr = self.SMMC.eval_all_SSM_metrics(r, w, self.test_data[j]['id'])
            p_nov += p
            r_nov += re
            f_nov += f
            ssm_corr_score += ssm_corr
            nov_corr_score += nov_corr

        ssm_corr_score /= N
        nov_corr_score /= N
        p_nov /= N
        r_nov /= N
        f_nov /= N

        self.feedback.print_feedback()

        mtr['P_Nov'] = p_nov
        mtr['R_Nov'] = r_nov
        mtr['F_Nov'] = f_nov
        mtr['SSM_Corr'] = ssm_corr_score
        mtr['Nov_Corr'] = nov_corr_score

        return mtr

    def eval_single_datapoint(self,j, results):
        t = self.test_data[j]['lighting_array'].copy()
        w = self.test_data[j]['waveform'].copy()
        b = self.test_data[j]['onset_beat'].copy()
        rms = self.test_data[j]['normed_rms'].copy()
        onset_env = self.test_data[j]['onset_env'].copy()
        i = self.test_data[j]['music_array'].copy()

        r = t
        if not self.is_dataset_eval:  # normal evaluation
            r = results[j].cpu().numpy()
        elif self.use_dataset_generator:
            r = self.data_generator.generate_or_modify_data(r, w, beats=b, onset=onset_env)

        bs_max, bs_min, nb_max, nb_min = self.calc_beat_align_score(r, b, j)

        rms_score = self.get_rms_score(r, rms, j)
        onset_score = self.get_onset_score(r, onset_env, j)
        p, re, f, SSM_Corr, nov_corr = self.SMMC.eval_all_SSM_metrics(r, w, self.test_data[j]['id'])

        int, col, pan, tilt = self.calc_std_dev(r)

        return bs_max, bs_min, nb_max, nb_min, rms_score, onset_score, p, re, f, SSM_Corr, int, col, pan, tilt, nov_corr

    def plot_distribution_and_metric_plots(self):
        self.feedback.reset()

        N = len(self.test_data)

        generators = ["None", "random", "onset", "novelty", "zeros", "static", "rms", "onbeat", "offbeat"]

        var_names = ['$\Gamma_\mathrm{beat \leftrightarrow peak}$', '$\Gamma_\mathrm{beat \leftrightarrow valley}$', 'N_BA_Max', 'N_BA_Min', '$\Gamma_\mathrm{loud \leftrightarrow bright}$',
                     '$\Gamma_\mathrm{change}$', 'P Nov', 'R Nov', '$\Gamma_\mathrm{boundary}$', '$\Gamma_\mathrm{structure}$', '$\Psi_\mathrm{intensity}$', '$\Psi_\mathrm{color}$',
                     '$\Psi_\mathrm{pan}$','$\Psi_\mathrm{tilt}$', '$\Gamma_\mathrm{novelty}$']

        name_plot_gen = generators.copy()
        name_plot_gen[0] = "real"
        name_plot_gen[3] = "chroma"

        vl_dic = {}
        for vl in generators:
            vl_dic[vl] = []
            for k in range(len(var_names)):
                vl_dic[vl].append([])

            self.set_and_activate_base_generator(vl)
            for j in range(N):
                v = self.eval_single_datapoint(
                    j, None)

                for k in range(len(var_names)):
                    a = v[k]
                    if var_names[k] == '$\Gamma_\mathrm{beat \leftrightarrow peak}$' or var_names[k] == '$\Gamma_\mathrm{beat \leftrightarrow valley}$':
                        if v[k + 2] > 0:
                            a /= v[k + 2]
                        else:
                            continue
                    vl_dic[vl][k].append(a)


        idx = [0,1,4,5,9,14,8] #which eval scores are interesiting

        fig, ax = plt.subplots(4,2)
        ax_flat = ax.flatten()
        co = 0
        for i in idx:
            cax = ax_flat[co]
            name = var_names[i]
            data = []
            for g in generators:
                data.append(vl_dic[g][i])

            cax.set_title(var_names[i], fontsize=16)
            bp1 = cax.boxplot(data)
            cax.set_xticklabels(name_plot_gen, rotation=45, ha='right')
            if co in [0,1,6]:
                cax.set_ylim(-0.05,1.05)
            else:
                cax.set_ylim(-1.05,1.05)
            co += 1
        #fig.tight_layout()
        plt.subplots_adjust(hspace=0.6)
        plt.show()



        x=[]
        y=[]
        dy = []
        y_rand = []
        dy_rand = []
        data_real = []
        data_random = []
        data_static = []

        for i in idx:
            x.append(var_names[i])
            data_real.append(vl_dic["None"][i])
            data_random.append(vl_dic["random"][i])
            data_static.append(vl_dic["static"][i])

            data_mean = np.mean(vl_dic["None"][i])
            rand_mean = np.mean(vl_dic["random"][i])
            rand_std = np.std(vl_dic["random"][i])
            data_std = np.std(vl_dic["None"][i])
            static_mean = "ND"
            static_std = "ND"
            if len(vl_dic["static"][i]) > 0:
                static_mean = np.mean(vl_dic["static"][i])
                static_std = np.std(vl_dic["static"][i])
            y.append(data_mean)
            y_rand.append(rand_mean)
            dy.append(data_std)
            dy_rand.append(rand_std)
            utils.logger.log(f"Var {var_names[i]}: data mean {data_mean}, data std {data_std}, rand mean {rand_mean}, rand std {rand_std}, static mean {static_mean}, static std {static_std}")

        pos_random = np.array(range(len(x)))*2
        pos_real = list(pos_random-0.6)
        pos_static = list(pos_random+0.6)
        pos_random = list(pos_random)

        fig, ax = plt.subplots()
        bp1 = ax.boxplot(data_real, positions=pos_real, patch_artist=True)
        bp2 = ax.boxplot(data_random, positions=pos_random, patch_artist=True)
        bp3 = ax.boxplot(data_static, positions=pos_static, patch_artist=True)

        set_bp_edge(bp1, 'green')
        set_bp_fill(bp1, 'white')

        set_bp_edge(bp2, 'red')
        set_bp_fill(bp2, 'white')

        set_bp_edge(bp3, 'blue')
        set_bp_fill(bp3, 'white')

        ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['real', 'random', 'static'])
        plt.xticks(pos_random, x)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.show()

        self.feedback.print_feedback()

    def evaluate_subroutine(self, results):
        self.feedback.reset()
        mse = 0
        mae = 0
        smin_beat_align_score = 0
        smax_beat_align_score = 0
        nmax_beat_align_score = 0
        nmin_beat_align_score = 0
        mov_scores = np.zeros((len(self.movement_score_window_sizes)))
        N = len(self.test_data)
        mtr = {}
        std_int = 0
        std_pan = 0
        std_tilt = 0
        std_col = 0
        p_nov, r_nov, f_nov = 0 , 0, 0
        rms_score = 0
        onset_score = 0
        ssm_corr_score = 0
        nov_corr_score = 0

        predictions=[]

        for j in range(N):
            t = self.test_data[j]['lighting_array'].copy()

            if not self.is_dataset_eval:  # normal evaluation
                r = results[j].cpu().numpy()
                predictions.append(r)
                mse += metrics.mean_squared_error(r, t)
                mae += metrics.mean_absolute_error(r, t)
                mov_scores += self.get_movement_score(r, t)

            bs_max, bs_min, nb_max, nb_min, rms_s, onset_s, p, re, f, SSM_Corr, int, col, pan, tilt, nov_corr = self.eval_single_datapoint(j, results)

            smax_beat_align_score += bs_max
            smin_beat_align_score += bs_min
            nmax_beat_align_score += nb_max
            nmin_beat_align_score += nb_min

            rms_score += rms_s
            onset_score += onset_s
            p_nov += p
            r_nov += re
            f_nov += f
            ssm_corr_score += SSM_Corr
            std_int += int
            std_col += col
            std_pan += pan
            std_tilt += tilt
            nov_corr_score += nov_corr


        ssm_corr_score /= N
        nov_corr_score /= N
        mse /= N
        mae /= N
        mov_scores /= N
        if nmax_beat_align_score:
            max_beat_align_score = smax_beat_align_score / nmax_beat_align_score
        else:
            max_beat_align_score = 0
        if nmin_beat_align_score:
            min_beat_align_score = smin_beat_align_score / nmin_beat_align_score
        else:
            min_beat_align_score = 0
        p_nov /= N
        r_nov /= N
        f_nov /= N
        std_int /= N
        std_col /= N
        std_pan /= N
        std_tilt /= N
        rms_score /= N
        onset_score /= N

        self.feedback.print_feedback()

        if not self.is_dataset_eval:
            pfid, sfid = self.FIDCalculator.process(predictions, self.test_data)
            mtr['MSE'] = mse
            mtr['MAE'] = mae
            mtr['PFID'] = pfid
            mtr['SFID'] = sfid

            for i in range(len(self.movement_score_window_sizes)):
                mtr['Movement Score WS ' + str(self.movement_score_window_sizes[i])] = mov_scores[i]

        mtr['Onset_Score'] = onset_score
        mtr['Max Beat Align Score'] = max_beat_align_score
        mtr['Min Beat Align Score'] = min_beat_align_score
        mtr['P_Nov'] = p_nov
        mtr['R_Nov'] = r_nov
        mtr['F_Nov'] = f_nov
        mtr['STD_Int'] = std_int
        mtr['STD_Col'] = std_col
        mtr['STD_Pan'] = std_pan
        mtr['STD_Tilt'] = std_tilt
        mtr['RMS'] = rms_score
        mtr['SSM_Corr'] = ssm_corr_score
        mtr['Nov_Corr'] = nov_corr_score

        return mtr

    def evaluate(self, results):
        return self.evaluate_subroutine(results)

    def evaluate_dataset(self):
        return self.evaluate_subroutine(None)

    def get_movement_score(self, out_seq, tgt_seq):
        l = len(tgt_seq)

        scores = np.zeros(len(self.movement_score_window_sizes))
        n = out_seq.shape[1]

        for j in range(len(self.movement_score_window_sizes)):
            v = self.movement_score_window_sizes[j]
            w = int(v)
            s = np.zeros(n)
            c = 0
            i = w
            i_old = 0
            while i < l:
                d_t = tgt_seq[i] - tgt_seq[i_old]
                d_g = out_seq[i] - out_seq[i_old]
                x = d_t - d_g
                x = np.abs(x)
                s += x
                i_old = i
                i += w
                c += 1
            s /= c

            # return this measure only for special params as dimmer
            s = s * self.AttrMasks.int_mask #dimmer only
            s = sum(s)
            s = s / self.AttrMasks.int_mask.sum()

            scores[j] = s

        return scores

    def get_rms_score(self, light, rms_in, index):
        mask = self.AttrMasks.int_mask # only dimmers

        rms = rms_in.flatten()

        sum_seq = None

        for i in range(len(mask)):
            if not mask[i]:
                continue
            if sum_seq is None:
                sum_seq = light[:, i].copy()
            else:
                sum_seq += light[:, i]

        std = np.std(sum_seq)
        if std == 0:
            utils.logger.log(self.test_data[index]['id'] + " has 0 standard deviation in all dimmer values!")

        nr_batches = int(sum_seq.size / self.rms_window_size)

        rms_batches = np.array([np.sum(item) for item in np.array_split(rms, nr_batches)])
        sum_seq_batches = np.array([np.sum(item) for item in np.array_split(sum_seq, nr_batches)])

        rms_batches = list(rms_batches/np.max(rms_batches))

        if np.max(sum_seq_batches) == 0:
            return 0

        sum_seq_batches = list(sum_seq_batches/ np.max(sum_seq_batches))


        corr, _ = pearsonr(rms_batches, sum_seq_batches)

        if self.plot_rms:
            fig, ax = plt.subplots(nrows=2, figsize=(10, 15))
            fig.suptitle('$\Gamma_\mathrm{loud}$', fontsize=16)
            ax[0].scatter(rms_batches, sum_seq_batches)
            ax[0].set_ylabel("brightness")
            ax[0].set_xlabel("rms")

            x = np.arange(len(sum_seq_batches))*(self.rms_window_size / 30)

            ax[1].plot(x,sum_seq_batches, label="brightness")
            ax[1].plot(x,rms_batches, label="rms")
            ax[1].legend(loc="upper left")
            ax[1].set_xlabel("seconds")

            p = os.path.join(self.path_rms, self.test_data[index]['id'] + '.png')
            plt.savefig(p)
            plt.close(fig)

        if np.isnan(corr):
            corr = 0.5
            utils.logger.log(self.test_data[index]['id'] + " does not work for rms score. Delete it! Now!")

        return corr

    def get_onset_score(self, out_seq, onset_in, index):

        onset_in = onset_in.flatten()
        mask = self.AttrMasks.all_val_mask
        nr_cut_frames_start = 30

        dif_light = out_seq.copy()

        for i in range(1,out_seq.shape[0]):
            dif_light[i] = np.abs(out_seq[i] -  out_seq[i-1])

        dif_light = dif_light[:,mask]

        dif_light_sum = np.sum(dif_light, axis=1)

        #cut first nr_cut_frames_start frames

        dif_light_sum = dif_light_sum[nr_cut_frames_start:]
        onset_in = onset_in[nr_cut_frames_start:]

        nr_batches = int(dif_light_sum.size / self.onset_window_size)

        os_batches = [np.sum(item) for item in np.array_split(onset_in, nr_batches)]
        dif_batches = [np.sum(item) for item in np.array_split(dif_light_sum, nr_batches)]

        corr, _ = pearsonr(os_batches, dif_batches)

        if self.plot_rms:
            fig, ax = plt.subplots(nrows=2, figsize=(20, 30))
            fig.suptitle('Onset Correlation', fontsize=16)
            ax[0].scatter(os_batches, dif_batches)
            ax[0].set_ylabel("light movement")
            ax[0].set_xlabel("audio onset")

            x = np.arange(len(dif_batches)) * (self.onset_window_size / 30)

            ax[1].plot(x,dif_batches, label="Light Movement")
            ax[1].plot(x,os_batches, label = "Audio Onset")
            ax[1].legend(loc="upper left")

            p = os.path.join(self.path_onset, self.test_data[index]['id'] + '.png')
            plt.savefig(p)
            plt.close(fig)

        if np.isnan(corr):
            corr = 0.5
            utils.logger.log(self.test_data[index]['id'] + " does not work for onset score. Delete it! Now!")


        return corr

    def calc_std_dev(self, out_seq):
        std = np.std(out_seq, 0)
        std_int = std * self.AttrMasks.int_mask
        std_hue = std * self.AttrMasks.hue_mask
        std_hue = std_hue[self.AttrMasks.hue_mask != 0]
        std_sat = std * self.AttrMasks.sat_mask
        std_sat = std_sat[self.AttrMasks.sat_mask != 0]
        std_pan = std * self.AttrMasks.pan_mask
        std_pan = std_pan[self.AttrMasks.pan_mask != 0]
        std_tilt = std * self.AttrMasks.tilt_mask
        std_tilt = std_tilt[self.AttrMasks.tilt_mask != 0]

        int = np.sum(std_int) / np.sum(self.AttrMasks.int_mask)
        std_col = np.max(np.array([std_sat,std_hue]),0)
        col = np.mean(std_col)
        pan = np.sum(std_pan) / np.sum(self.AttrMasks.pan_mask)
        tilt = np.sum(std_tilt) / np.sum(self.AttrMasks.tilt_mask)


        return int, col, pan, tilt


    def calc_eval_weight(self, category_weights, data_types):
        n = len(data_types)
        final_weights = {}
        for v in set(data_types):
            final_weights[v] = 0
        for v in data_types:
            final_weights[v] += 1
        counts = final_weights.copy()
        for k, v in final_weights.items():
            final_weights[k] /= n
        for k, v in final_weights.items():
            if k not in category_weights.keys():
                final_weights[k] = 0

        for k, v in category_weights.items():
            if k in final_weights.keys():
                final_weights[k] *= v
            else:
                final_weights[k] = 0

        s = 0
        for v in final_weights.values():
            s += v

        for k, v in final_weights.items():
            final_weights[k] /= (s * counts[k])

        return final_weights

    def downsample_signal(self, signal, resolution):
        signal *= resolution
        return np.floor(signal) / resolution

    def get_beat_align_peaks(self, signal):
        peak_distance = 16
        peak_prominence = 0.15

        peaks, d = find_peaks(signal, prominence=peak_prominence, distance=peak_distance)

        # set peaks to start of new value
        for m in range(len(peaks)):
            idx = peaks[m]
            vs = signal[idx]
            while idx - 1 > 0 and abs(vs - signal[idx - 1]) < 0.001:
                idx -= 1
                peaks[m] = idx

        # delete following peak copies with same prominence boundaries
        to_delete = []
        rb = d['right_bases']
        lb = d['left_bases']
        lp = len(peaks)
        for m in range(lp):
            if m in to_delete:
                continue
            a = 1
            while m + a < lp and rb[m] == rb[m + a] and lb[m] == lb[m + a]:
                signal_slice = signal[peaks[m]:peaks[m+a]]
                if np.max(signal_slice) - np.min(signal_slice) > peak_prominence:
                    break
                to_delete.append(m + a)
                a += 1
        if to_delete:
            peaks = np.delete(peaks, np.array(to_delete))

        return peaks

    def get_closest_beat_distance(self, beats, idx):
        closest_beat = np.abs(idx - beats.flatten()).min()
        closest_beat = closest_beat - self.beat_align_frame_boundary
        closest_beat = max(0, closest_beat)

        return closest_beat

    def get_best_beat_align_shift_values(self,test_output, nz_base, dimmer_resolution, mask_max, mask_min):

        max_scores = []
        min_scores = []
        shifts = range(-self.shift_window_beat_algin, self.shift_window_beat_algin + 1)
        for sw in shifts:
            s_max = 0
            s_min = 0
            nz_s = nz_base + sw

            for i in range(len(mask_max)):
                if mask_max[i] or mask_min[i]:
                    seq = test_output[:, i]
                    seq_f = seq.copy().flatten()
                    seq_f = self.downsample_signal(seq_f, dimmer_resolution)

                    seq_max = seq_f.copy()
                    seq_min = 1 - seq_f.copy()

                if mask_max[i]:
                    peaks = self.get_beat_align_peaks(seq_max)

                    for v in peaks:
                        closest_beat = self.get_closest_beat_distance(nz_s, v)
                        s_max += np.exp(- closest_beat / (2 * (self.beat_align_sigma ** 2)))

                if mask_min[i]:
                    peaks = self.get_beat_align_peaks(seq_min)

                    for v in peaks:
                        closest_beat = self.get_closest_beat_distance(nz_s, v)
                        s_min += np.exp(- closest_beat / (2 * (self.beat_align_sigma ** 2)))

            max_scores.append(s_max)
            min_scores.append(s_min)

        best_shift_max = shifts[max_scores.index(max(max_scores))]
        best_shift_min = shifts[max_scores.index(max(max_scores))]

        return best_shift_max, best_shift_min

    def calc_beat_align_score(self, test_output, beats, index):
        dimmer_resolution = 32

        mask_min = self.AttrMasks.int_mask # only dimmers
        mask_max = self.AttrMasks.int_mask # only dimmers
        beats = beats.copy().flatten()
        nz_base = np.argwhere(beats == 1)

        n_max_peaks = 0
        n_min_peaks = 0
        T_coef = np.arange(len(beats)) / 30

        shifts = range(-self.shift_window_beat_algin, self.shift_window_beat_algin + 1)
        max_results = np.zeros(len(shifts))
        min_results = np.zeros(len(shifts))

        if self.plot_beat_align:
            fig,axes = plt.subplots(nrows = sum(self.AttrMasks.int_mask), figsize = (75,50))
            for ax in axes:
                ax.set_xlabel("Seconds")

            fig.suptitle(self.test_data[index]['id'] + " beat align", fontsize=16)

        k = 0
        for i in range(len(mask_max)):
            if mask_max[i] or mask_min[i]:
                seq = test_output[:, i]
                seq_f = seq.copy().flatten()
                seq_f = self.downsample_signal(seq_f, dimmer_resolution)

                seq_max = seq_f.copy()
                seq_min = 1 - seq_f.copy()


                if self.plot_beat_align:
                    ax = axes[k]
                    ax.set_ylabel(self.attri[i][0] + ' ' + self.attri[i][1], rotation=0)
                    ax.set_ylim([-0.05, 1.05])
                    ax.plot(T_coef,seq)
                    ymin, ymax = ax.get_ylim()
                k += 1

            if mask_max[i]:
                self.feedback.count_max_beat += 1

                peaks = self.get_beat_align_peaks(seq_max)

                n_max_peaks += len(peaks)

                for v in peaks:
                    for idxx in range(len(shifts)):
                        nz_c = nz_base.copy() + shifts[idxx]
                        closest_beat = self.get_closest_beat_distance(nz_c, v)
                        max_results[idxx] += np.exp(- closest_beat / (2*(self.beat_align_sigma**2)))

                if not len(peaks):
                    self.feedback.missing_max_beat += 1
                if len(peaks) > 100:
                    utils.logger.log("More than 100 max peaks for " + self.test_data[index]['id'])

                if self.plot_beat_align:
                    if k==1:
                        ax.vlines(x=T_coef[peaks], ymin = ymin, ymax = ymax, label='Max', ls='--', colors=['lime'])
                    else:
                        ax.vlines(x=T_coef[peaks], ymin=ymin, ymax=ymax, ls='--', colors=['lime'])


            if mask_min[i]:
                self.feedback.count_min_beat += 1
                peaks = self.get_beat_align_peaks(seq_min)

                n_min_peaks += len(peaks)

                for v in peaks:
                    for idxx in range(len(shifts)):
                        nz_c = nz_base.copy() + shifts[idxx]
                        closest_beat = self.get_closest_beat_distance(nz_c, v)
                        min_results[idxx] += np.exp(- closest_beat / (2*(self.beat_align_sigma**2)))
                if not len(peaks):
                    self.feedback.missing_min_beat += 1

                if len(peaks) > 100:
                    utils.logger.log("More than 100 min peaks for " + self.test_data[index]['id'])

                if self.plot_beat_align:
                    if k==1:
                        ax.vlines(x=T_coef[peaks], ymin = ymin, ymax = ymax, label='Min', ls='--', colors=['tab:red'])
                    else:
                        ax.vlines(x=T_coef[peaks], ymin=ymin, ymax=ymax, ls='--', colors=['tab:red'])

        if self.plot_beat_align:
            for i in range(len(axes)):
                ax = axes[i]
                ymin, ymax = ax.get_ylim()
                if i==0:
                    ax.vlines(x=T_coef[nz_base], ymin = ymin, ymax = ymax, label='Beats', ls='--', colors=['tab:gray'])
                else:
                    ax.vlines(x=T_coef[nz_base], ymin=ymin, ymax=ymax, ls='--', colors=['tab:gray'])

            fig.subplots_adjust(hspace=0.78)
            fig.legend(loc="upper left")

            p = os.path.join(self.path_ba, self.test_data[index]['id'] + '.png')
            plt.savefig(p)
            plt.close(fig)

        best_shift_max = numpy.argmax(max_results)
        best_shift_min = numpy.argmax(min_results)

        max_sum_score = max_results[best_shift_max]
        min_sum_score = min_results[best_shift_min]

        return max_sum_score, min_sum_score, n_max_peaks, n_min_peaks