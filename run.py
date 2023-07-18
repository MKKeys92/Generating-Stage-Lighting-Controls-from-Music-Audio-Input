import torch
import torch.utils.data
from utils import util
import warnings
import os
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from dataset import LightDataset, paired_collate_fn, PredictionDataset
from models.rnn_model import Encoder, Decoder, Model
from models.transformer_model import PureTransformerModel
from datetime import datetime
from os.path import join
import utils.logger
from utils.functional import load_data
from preprocessing.audio_extractor import extract_acoustic_feature
from tqdm import tqdm
import json
import pyhocon
import numpy as np
from evaluation.evaluator import LightEvaluator
from models.layers import STDN1Loss

warnings.filterwarnings('ignore')

class Runner:
    def __init__(self, config_name, gpu_id=0):

        self.name_suffix = datetime.now().strftime('%b%d_%H-%M-%S')
        self.gpu_id = gpu_id

        # Set up config
        if config_name is not None:
            self.name = config_name
            self.config = util.initialize_config(config_name, self.name_suffix)

            prepro_conf_path = os.path.join(self.config.data_dir, 'preprocess_config.json')
            with open(prepro_conf_path, 'r') as fp:
                self.config['prepro_config'] = json.load(fp)

            # Set up seed
            # if self.config.seed != -1:
            #     util.set_seed(self.config.seed)

        # Set up device
        self.device = torch.device('cpu' if gpu_id == -1 else f'cuda:{gpu_id}')
        self.runtime_args = {}

    def initialize_model(self, args, eval = True):
        self.runtime_args = args
        if args.saved_model:
             checkpoint = self.load_model_checkpoint(args.saved_model, args.model_directory)

             if 'config' in checkpoint.keys():
                 self.config = checkpoint['config']
             else: #to open models of old type
                 self.config = util.initialize_config('compatibility_config')
                 self.config.name = 'compatibility_config'
             self.name = self.config.name
             self.config.loaded_model = self.config.name_suffix
             self.config.name_suffix = self.name_suffix
             utils.logger.init(self.config)
             # Set up seed
             if self.config.seed != -1:
                 util.set_seed(self.config.seed)
        else:
            self.config.last_epoch_count = 0
            utils.logger.init(self.config)


        self.config.d_model = self.config.music_emb_size

        if not os.path.exists(self.config.output_dir):
            os.makedirs(self.config.output_dir)


        print(self.config)
        if self.config.model_type == "RNNDecoder":
            encoder = Encoder(max_seq_len=self.config.max_seq_len,
                              input_size=self.config.prepro_config['music_dim'],
                              d_word_vec=self.config.music_emb_size,
                              n_layers=self.config.n_layers,
                              n_head=self.config.n_head,
                              d_k=self.config.d_k,
                              d_v=self.config.d_v,
                              d_model=self.config.d_model,
                              d_inner=self.config.d_inner,
                              dropout=self.config.dropout)

            decoder = Decoder(input_size=self.config.prepro_config['lighting_dim'],
                              d_word_vec=self.config.lighting_emb_size,
                              hidden_size=self.config.d_inner,
                              encoder_d_model=self.config.d_model,
                              dropout=self.config.dropout)

            model = Model(encoder, decoder,
                          condition_step=self.config.condition_step,
                          sliding_window_size=self.config.sliding_windown_size,
                          lambda_v=self.config.lambda_v,
                          config = self.config,
                          device=self.device)
        elif self.config.model_type == "Transformer":
            model = PureTransformerModel(self.config, self.device)

        print(model)

        for name, parameters in model.named_parameters():
            print(name, ':', parameters.size())

        #ToDo: Support Data Parallel again

        if args.saved_model:
            model.load_state_dict(checkpoint['model'])
            utils.logger.log('[Info] Trained model loaded %s' % args.saved_model)

        if(eval):
            utils.logger.log('[Info] Preparing Evaluation Data')
            train_data, test_data, val_data = load_data(self.config)
            self.evaluator = LightEvaluator(None, None, self.config, test_data, self.runtime_args)

        model.to(self.device)

        return model

    def prepare_dataloader(self, data, args, is_training = False):
        if(is_training):
            data_loader = torch.utils.data.DataLoader(
                LightDataset(data),
                num_workers=0,
                batch_size=args.batch_size,
                collate_fn=paired_collate_fn,
                pin_memory=True,
                shuffle=True
            )
        else:
            data_loader = torch.utils.data.DataLoader(
                LightDataset(data),
                num_workers=0,
                batch_size=args.batch_size,
                collate_fn=paired_collate_fn,
                pin_memory=True,
                shuffle=False
            )
        return data_loader

    def prepare_predict_dataloader(self, music_data, light_data, args):
        data_loader = torch.utils.data.DataLoader(
            PredictionDataset(music_data, light_data),
            num_workers=0,
            batch_size=args.batch_size,
            collate_fn=paired_collate_fn,
            pin_memory=True,
            shuffle=False
        )

        return data_loader

    def train(self, model):
        """ Start training """

        # Loading training data
        train_data, test_data, val_data = load_data(self.config)
        training_data = self.prepare_dataloader(train_data, self.config, is_training = True)

        optimizer = optim.Adam(filter(
            lambda x: x.requires_grad, model.parameters()), lr=self.config.lr)

        #criterion = nn.MSELoss()
        #criterion = STDN1Loss()
        criterion = nn.L1Loss()
        updates = 0  # global step

        start = self.config.last_epoch_count

        for epoch_i in range(start, self.config.epoch + 1):
            utils.logger.log.set_progress(epoch_i, len(training_data))
            model.train()

            for batch_i, batch in enumerate(training_data):
                # prepare data
                src_seq, src_pos, tgt_seq = map(lambda x: x.to(self.device), batch)

                # forward
                optimizer.zero_grad()

                loss = model.train_step(src_seq, src_pos, tgt_seq, optimizer, criterion, epoch_i)

                # update parameters
                optimizer.step()

                stats = {
                    'updates': updates,
                    'loss': loss.item()
                }
                utils.logger.log.update(stats)
                updates += 1

            if epoch_i % self.config.save_per_epochs == 0 or epoch_i == 1:
                self.save_model_checkpoint(model, epoch_i)

            if epoch_i % self.config.eval_frequency == 0:
                self.evaluate(model)
                pass

            self.config.last_epoch_count = epoch_i

        self.evaluate(model)

    def evaluate(self, model):

        utils.logger.log('Starting Evaluation')

        train_data, test_data, val_data = load_data(self.config)

        test_loader = self.prepare_dataloader(test_data, self.config)

        results = []
        for batch in test_loader:
            src_seq, src_pos, tgt_seq = map(lambda x: x.to(self.device), batch)
            b_res = model.predict(src_seq, src_pos, tgt_seq = tgt_seq[:,:self.runtime_args.used_startup_frames])
            for i in range(b_res.size()[0]):
                results.append(b_res[i])

        mtr = self.evaluator.evaluate(results)

        utils.logger.log.log_eval(mtr)

        return

    def predict(self, model):

        if not os.path.exists(self.runtime_args.output_dir):
            os.makedirs(self.runtime_args.output_dir)

        music_data, audio_fnames, _ = extract_acoustic_feature(self.runtime_args.audio_dir, self.config.prepro_config)
        # print(audio_fnames)

        np_music_data = []
        for d in music_data:
            np_music_data.append(np.array(d))

        utils.logger.log('Predicting %d samples...' % len(np_music_data))

        if self.runtime_args.used_startup_frames != 0:
            light_data = util.load_startup_light_data(self.runtime_args.audio_dir, audio_fnames, self.runtime_args, model.prepro_config)
        else:
            light_data = None

        test_loader = self.prepare_predict_dataloader(music_data, light_data, self.config)

        results = []
        for batch in tqdm(test_loader, desc='Generating lighting data'):
            src_seq, src_pos, tgt_seq = batch
            src_seq = src_seq.to(self.device)
            src_pos = src_pos.to(self.device)
            if tgt_seq is not None:
                tgt_seq = tgt_seq.to(self.device)
                tgt_seq = tgt_seq[:, :self.runtime_args.used_startup_frames]

            # generate_num = 5
            # for _ in range(generate_num):
            b_res = model.predict(src_seq, src_pos, tgt_seq)
            for i in range(b_res.size()[0]):
                results.append(b_res[i])

        res_idx = len(audio_fnames)

        for idx in range(res_idx):
            name = audio_fnames[idx]
            if name.endswith('.m4a'):
                name = name[:-4]
            with open(os.path.join(self.runtime_args.output_dir, f'{name}.json'), 'w') as f:
                sample_dict = {
                    'id': audio_fnames[idx],
                    'lighting_array': results[idx].tolist(),
                    'lighting_array_dim': len(results[0][0])
                }
                json.dump(sample_dict, f)

        return

    def save_model_checkpoint(self, model, step):
        # if step < 30000:
        #     return  # Debug

        if not os.path.exists(self.config.checkpoint_dir):
            os.makedirs(self.config.checkpoint_dir)

        s_path = join(self.config.checkpoint_dir, self.config.name + self.name_suffix)

        if not os.path.exists(s_path):
            os.makedirs(s_path)

        path_ckpt = join(s_path, f'model_{self.name}_{self.name_suffix}_{step}.bin')

        checkpoint = {
            'model': model.state_dict(),
            'config': self.config,
            'epoch': step
        }

        torch.save(checkpoint, path_ckpt)
        utils.logger.log('Saved model to %s' % path_ckpt)

    def load_model_checkpoint(self, suffix, load_directory = None):

        path = None
        target_file_name = suffix
        dir = None
        if load_directory:
            dir = load_directory
        else:
            dir = self.config.checkpoint_dir

        for dirpath, dirnames, filenames in os.walk(dir):
            for file in filenames:
                if file == target_file_name:
                    path = os.path.join(dirpath, file)

        assert path is not None

        checkpoint = torch.load(path, map_location=self.device)

        return checkpoint