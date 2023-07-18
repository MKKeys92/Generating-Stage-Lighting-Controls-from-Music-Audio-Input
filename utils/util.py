from os import makedirs
import os.path
import numpy as np
import pyhocon
import utils.logger
import torch
import random
import json


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def initialize_config(config_name, name_suffix):
    config = pyhocon.ConfigFactory.parse_file(os.path.join(os.path.dirname(__file__), os.pardir, 'experiments.conf'))[config_name]
    config.name = config_name
    config.name_suffix = name_suffix

    utils.logger.init(config)
    utils.logger.log("Running experiment: {}".format(config_name))
    #config['log_dir'] = join(config["log_root"], config_name)
    #makedirs(config['log_dir'], exist_ok=True)

    #config['tb_dir'] = join(config['log_root'], 'tensorboard')
    #makedirs(config['tb_dir'], exist_ok=True)

    utils.logger.log(pyhocon.HOCONConverter.convert(config, "hocon"))
    return config

def set_seed(seed, set_gpu=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if set_gpu and torch.cuda.is_available():
        # Necessary for reproducibility; lower performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)
    utils.logger.log('Random seed is set to %d' % seed)

def load_startup_light_data(dir, audio_fnames, predict_args, config):
    utils.logger.log('loading startup light data')
    l_data = []
    for name in audio_fnames:
        if name.endswith('m4a') or name.endswith('wav'):
            name = name[:-3]
            name = name + "json"
            with open(os.path.join(dir, name)) as f:
                sample_dict = json.loads(f.read())
                np_light_seq = np.array(sample_dict['lighting_array'])
                if np_light_seq.shape[0] < predict_args.used_startup_frames:
                    raise ValueError("light seq to short" + name)
                if np_light_seq.shape[1] != config.prepro_config['lighting_dim']:
                    raise ValueError("light seq has wrong dimension" + name)
                l_data.append(np_light_seq)
        else:
            raise TypeError("unable to handle audio file " + name)
    return l_data