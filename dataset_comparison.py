import os
import utils.logger
from utils.functional import load_data
from utils.util import dotdict
import json
import argparse
import numpy as np
from evaluation.evaluator import LightEvaluator
from datetime import datetime
import librosa
from preprocessing.audio_extractor import FeatureExtractor
import pickle
from evaluation.evaluator import AttributeMasks
from evaluation.evaluator import set_bp_edge
from evaluation.evaluator import set_bp_fill
import matplotlib.pyplot as plt
# to evaluate a dataset by itself to get insights in evaluation scores

def load_dataset_and_create_evaluator(p_data, base_args):

    d = {}
    config = dotdict(d)
    config.data_dir = os.path.join("preprocessing","datasets",p_data)
    config.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
    config.name = "dataset_evaluation"
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

    return data, evaluator, config

def main():
    """ Main function """

    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=0,
                        help='which GPU to use, CPU has value -1')

    base_args = parser.parse_args()

    d_HAW, e_HAW, c_HAW = load_dataset_and_create_evaluator('base_full_songs_HAW',base_args)

    d_DK, e_DK, c_DK = load_dataset_and_create_evaluator('base_full_songs_DK', base_args)

    d_NC, e_NC, c_NC = load_dataset_and_create_evaluator('base_full_songs_NC', base_args)

    r_HAW = {}

    r_DK = {}

    r_NC = {}

    results = [r_HAW,r_DK,r_NC]
    data = [d_HAW,d_DK,d_NC]
    evals = [e_HAW, e_DK, e_NC]

    sets = ['Snippets', 'HipHop', 'Pop']
    labels = ['number of songs','durations','bpms', 'colors']

    AM = AttributeMasks(c_NC.prepro_config['lighting_config'], c_NC.prepro_config['lighting_dim'])

    for i in range(len(data)):
        d = data[i]
        r = results[i]

        #number of songs
        n = len(d)
        r[labels[0]]= n
        print('number of recordings ' + sets[i] + ': ' + str(n))

        #durations
        durations = []
        for da in d:
            durations.append(len(da['lighting_array'])/1800)
        r[labels[1]] = durations
        mean_duration = np.mean(durations)
        print('overall duration ' + sets[i] + ': ' + str(sum(durations)))
        print('mean duration ' + sets[i] + ': ' + str(mean_duration))

        #bpms
        bpms = []
        extractor = FeatureExtractor()
        for da in d:
            tempo = extractor.get_tempo(da['waveform'], c_DK.prepro_config)
            bpms.append(tempo)
        r[labels[2]] = bpms
        mean_bpm = np.mean(bpms)
        print('mean bpm ' + sets[i] + ': ' + str(mean_bpm))


        # colors
        colors = []
        for da in d:
            l = da['lighting_array'].copy()
            sm = AM.sat_mask
            l = l[:, sm ==1]
            size = l.size
            l = l>0
            n_c = np.sum(l)
            colors.append(n_c/size*100)
        r[labels[3]] = colors
        mean_colors = np.mean(colors)
        print('mean colors ' + sets[i] + ': ' + str(mean_colors))





    data_durations = []
    for r in results:
        data_durations.append(r[labels[1]])

    data_bpms = []
    for r in results:
        data_bpms.append(r[labels[2]])

    data_colors = []
    for r in results:
        data_colors.append(r[labels[3]])

    data_plot = [data_durations,data_bpms,data_colors]
    graph_labels=['duration', 'bpm', 'color']
    y_labels = ['min', 'bpm', 'percent']
    fig, ax = plt.subplots(1, 3)
    for i in range(len(sets)):
        bp = ax[i].boxplot(data_plot[i], labels=sets)
        ax[i].set_title(graph_labels[i])
        ax[i].set_ylabel(y_labels[i])
    plt.show()

    #eval for every dataset

    var_names = ['$\Gamma_\mathrm{beat \leftrightarrow peak}$', '$\Gamma_\mathrm{beat \leftrightarrow valley}$',
                 'N_BA_Max', 'N_BA_Min', '$\Gamma_\mathrm{loud \leftrightarrow bright}$',
                 '$\Gamma_\mathrm{change}$', 'P Nov', 'R Nov', '$\Gamma_\mathrm{boundary}$',
                 '$\Gamma_\mathrm{structure}$', '$\Psi_\mathrm{intensity}$', '$\Psi_\mathrm{color}$',
                 '$\Psi_\mathrm{pan}$', '$\Psi_\mathrm{tilt}$', '$\Gamma_\mathrm{novelty}$']


    vl_dic = {}
    for dsi in range(len(sets)):
        s_label = sets[dsi]
        vl_dic[s_label] = []
        for k in range(len(var_names)):
            vl_dic[s_label].append([])

        dataset = data[dsi]
        N = len(dataset)
        ev = evals[dsi]

        for j in range(N):
            v = evals[dsi].eval_single_datapoint(
                j, None)

            for k in range(len(var_names)):
                a = v[k]
                if var_names[k] == '$\Gamma_\mathrm{beat \leftrightarrow peak}$' or var_names[
                    k] == '$\Gamma_\mathrm{beat \leftrightarrow valley}$':
                    if v[k + 2] > 0:
                        a /= v[k + 2]
                    else:
                        continue
                vl_dic[s_label][k].append(a)

    idx = [0, 1, 4, 5, 9, 14, 8]  # which eval scores are interesiting
    x = []
    y = []
    dy = []
    y_rand = []
    dy_rand = []
    data_snippet = []
    data_HipHop = []
    data_Pop = []

    for i in idx:
        x.append(var_names[i])
        data_snippet.append(vl_dic["Snippets"][i])
        data_HipHop.append(vl_dic["HipHop"][i])
        data_Pop.append(vl_dic["Pop"][i])

        Snippets_mean = np.mean(vl_dic["Snippets"][i])
        HipHop_mean = np.mean(vl_dic["HipHop"][i])
        Pop_mean = np.mean(vl_dic["Pop"][i])


        Snippets_std = np.std(vl_dic["Snippets"][i])
        HipHop_std = np.std(vl_dic["HipHop"][i])
        Pop_std = np.std(vl_dic["Pop"][i])


        utils.logger.log(
            f"Var {var_names[i]}: Snippets mean {Snippets_mean}, Snippets std {Snippets_std}, HipHop mean {HipHop_mean}, HipHop std {HipHop_std}, Pop mean {Pop_mean}, Pop std {Pop_std}")

    pos_mid = np.array(range(len(x))) * 2
    pos_left = list(pos_mid - 0.6)
    pos_right = list(pos_mid + 0.6)
    pos_mid = list(pos_mid)

    fig, ax = plt.subplots()
    bp1 = ax.boxplot(data_snippet, positions=pos_left, patch_artist=True)
    bp2 = ax.boxplot(data_HipHop, positions=pos_mid, patch_artist=True)
    bp3 = ax.boxplot(data_Pop, positions=pos_right, patch_artist=True)

    set_bp_edge(bp1, 'green')
    set_bp_fill(bp1, 'white')

    set_bp_edge(bp2, 'red')
    set_bp_fill(bp2, 'white')

    set_bp_edge(bp3, 'blue')
    set_bp_fill(bp3, 'white')

    ax.legend([bp1["boxes"][0], bp2["boxes"][0], bp3["boxes"][0]], ['Snippets', 'HipHop', 'Pop'])
    plt.xticks(pos_mid, x)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

if __name__ == '__main__':
    main()