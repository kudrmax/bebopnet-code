# coding: utf-8
import configargparse
import csv
import time
import math
import os
import itertools
import shutil
import json
import sys
from functools import partial
import random
from typing import Optional

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from jazz_rnn.B_next_note_prediction.transformer.data_utils import JazzCorpus
from jazz_rnn.B_next_note_prediction.transformer.mem_transformer import MemTransformerLM
from jazz_rnn.B_next_note_prediction.transformer.utils.exp_utils import create_exp_dir
from jazz_rnn.utilspy.meters import AverageMeter


def init_weight(weight, args):
    if args.init == 'uniform':
        nn.init.uniform_(weight, -args.init_range, args.init_range)
    elif args.init == 'normal':
        nn.init.normal_(weight, 0.0, args.init_std)


def init_bias(bias):
    nn.init.constant_(bias, 0.0)


def weights_init(m, args):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight, args)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('AdaptiveEmbedding') != -1:
        if hasattr(m, 'emb_projs'):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    nn.init.normal_(m.emb_projs[i], 0.0, args.proj_init_std)
    elif classname.find('Embedding') != -1:
        if hasattr(m, 'weight'):
            init_weight(m.weight, args)
    elif classname.find('ProjectedAdaptiveLogSoftmax') != -1:
        if hasattr(m, 'cluster_weight') and m.cluster_weight is not None:
            init_weight(m.cluster_weight, args)
        if hasattr(m, 'cluster_bias') and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, 'out_projs'):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, args.proj_init_std)
    elif classname.find('LayerNorm') != -1:
        if hasattr(m, 'weight'):
            nn.init.normal_(m.weight, 1.0, args.init_std)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)
    elif classname.find('TransformerLM') != -1:
        if hasattr(m, 'r_emb'):
            init_weight(m.r_emb, args)
        if hasattr(m, 'r_w_bias'):
            init_weight(m.r_w_bias, args)
        if hasattr(m, 'r_r_bias'):
            init_weight(m.r_r_bias, args)
        if hasattr(m, 'r_bias'):
            init_bias(m.r_bias)


def update_dropout(m, args):
    classname = m.__class__.__name__
    if classname.find('Dropout') != -1:
        if hasattr(m, 'p'):
            m.p = args.dropout


def update_dropatt(m, args):
    if hasattr(m, 'dropatt'):
        m.dropatt.p = args.dropatt


EPOCHS_CSV_HEADER = [
    "step",
    "elapsed_sec",
    "lr",
    "val_loss",
    "val_nll",
    "val_p_top1",
    "val_d_top1",
    "val_p_entropy",
    "val_d_entropy",
]
TRAIN_STATE_FILE = "train_state.json"
SUMMARY_FILE = "summary.json"
EPOCHS_CSV_FILE = "epochs.csv"


def _save_train_state(
    work_dir: str,
    train_step: int,
    best_val_loss: Optional[float],
    best_val_p_acc: float,
    best_val_d_acc: float,
    train_time_sec: float,
) -> None:
    state = {
        "train_step": int(train_step),
        "best_val_loss": float(best_val_loss) if best_val_loss is not None else None,
        "best_val_p_acc": float(best_val_p_acc),
        "best_val_d_acc": float(best_val_d_acc),
        "train_time_sec": float(train_time_sec),
    }
    with open(os.path.join(work_dir, TRAIN_STATE_FILE), "w") as f:
        json.dump(state, f, indent=2)


def _load_train_state(restart_dir: str) -> Optional[dict]:
    path = os.path.join(restart_dir, TRAIN_STATE_FILE)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _append_epoch_csv(work_dir: str, row: dict) -> None:
    path = os.path.join(work_dir, EPOCHS_CSV_FILE)
    write_header = not os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EPOCHS_CSV_HEADER)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in EPOCHS_CSV_HEADER})


class Trainer:
    def __init__(self, in_args):
        if not hasattr(self, 'model_class'):
            self.model_class = MemTransformerLM
        self.parse_args(in_args)
        self.setup_logging()
        self.seed()

        self.device = torch.device('cuda' if self.args.cuda else 'cpu')
        self.load_data()
        self.build_model()
        self.init_optimizer()
        self.best_val_loss = 1e6
        self.best_val_p_acc = 0
        self.best_val_d_acc = 0
        self.meters_list = [
            'loss', 'nll', 'p_nll', 'd_nll',
            'p_top1', 'p_top3', 'p_top5', 'd_top1', 'd_top3',
            'p_entropy', 'd_entropy', 't_entropy'
        ]

    def setup_logging(self):
        path_prefix = os.path.join(os.getcwd(), 'jazz_rnn', 'B_next_note_prediction', 'transformer')
        self.logging = create_exp_dir(self.args.work_dir,
                                      scripts_to_save=[os.path.join(path_prefix, 'train.py'),
                                                       os.path.join(path_prefix, 'mem_transformer.py')])

        self.writer = SummaryWriter(log_dir=self.args.work_dir)

    def seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available():
            if not self.args.cuda:
                print('WARNING: You have a CUDA device, so you should probably run with --cuda')
            else:
                torch.cuda.manual_seed_all(self.args.seed)

    def parse_args(self, in_args):
        parser = configargparse.ArgumentParser(description='PyTorch Transformer Language Model')
        parser.add_argument('--config', type=str, required=True, is_config_file=True,
                            default='./configs/train_nnp.yml',
                            help='configuration file for the rest of the arguments')

        logging_parser = parser.add_argument_group('Logging args')
        logging_parser.add_argument('--save_name', type=str, default='',
                                    help='experiment name')
        logging_parser.add_argument('--log_interval', type=int, default=200,
                                    help='report interval')
        logging_parser.add_argument('--eval_interval', type=int, default=4000,
                                    help='evaluation interval')
        logging_parser.add_argument('--work_dir', default='results/training_results/transformer', type=str,
                                    help='experiment directory.')
        logging_parser.add_argument('--no_timestamp', action='store_true',
                                    help='do not append a timestamp suffix to work_dir '
                                         '(stable directory across restarts — needed for resume idempotency)')

        model_parser = parser.add_argument_group('Model args')
        model_parser.add_argument('--n_layer', type=int, default=6,
                                  help='number of total layers')
        model_parser.add_argument('--n_head', type=int, default=10,
                                  help='number of heads')
        model_parser.add_argument('--d_head', type=int, default=41,
                                  help='head dimension')
        model_parser.add_argument('--d_embed', type=int, default=-1,
                                  help='embedding dimension')
        model_parser.add_argument('--d_model', type=int, default=410,
                                  help='model dimension')
        model_parser.add_argument('--d_inner', type=int, default=2100,
                                  help='inner dimension in FF')
        model_parser.add_argument('--dropout', type=float, default=0.0,
                                  help='global dropout rate')
        model_parser.add_argument('--dropatt', type=float, default=0.0,
                                  help='attention probability dropout rate')
        model_parser.add_argument('--init', default='normal', type=str,
                                  help='parameter initializer to use.')
        model_parser.add_argument('--init_range', type=float, default=0.1,
                                  help='parameters initialized by U(-init_range, init_range)')
        model_parser.add_argument('--init_std', type=float, default=0.02,
                                  help='parameters initialized by N(0, init_std)')
        model_parser.add_argument('--proj_init_std', type=float, default=0.01,
                                  help='parameters initialized by N(0, init_std)')
        model_parser.add_argument('--not_tied', action='store_true',
                                  help='do not tie the word embedding and softmax weights')
        model_parser.add_argument('--pitch_emsize', type=int, default=205,
                                  help='size of pitch embeddings')
        model_parser.add_argument('--dur_emsize', type=int, default=205,
                                  help='size of duration embeddings')
        model_parser.add_argument('--offset_emsize', type=int, default=16,
                                  help='size of offset embeddings (0 for no offset conditioning)')
        model_parser.add_argument('--chord_bias', action='store_true',
                                  help='Add chord conditioning')

        optim_parser = parser.add_argument_group('Optimization args')
        optim_parser.add_argument('--optim', default='adam', type=str,
                                  choices=['adam', 'sgd', 'adagrad', 'ranger'],
                                  help='optimizer to use.')
        optim_parser.add_argument('--lr', type=float, default=0.00025,
                                  help='initial learning rate (0.00025|5 for adam|sgd)')
        optim_parser.add_argument('--mom', type=float, default=0.0,
                                  help='momentum for sgd')
        optim_parser.add_argument('--scheduler', default='cosine', type=str,
                                  choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant'],
                                  help='lr scheduler to use.')
        optim_parser.add_argument('--warmup_step', type=int, default=20000,
                                  help='upper epoch limit')
        optim_parser.add_argument('--decay_rate', type=float, default=0.5,
                                  help='decay factor when ReduceLROnPlateau is used')
        optim_parser.add_argument('--lr_min', type=float, default=0.0,
                                  help='minimum learning rate during annealing')
        optim_parser.add_argument('--clip', type=float, default=0.25,
                                  help='gradient clipping')
        optim_parser.add_argument('--patience', type=int, default=0,
                                  help='patience')
        optim_parser.add_argument('--eta_min', type=float, default=0.0,
                                  help='min learning rate for cosine scheduler')

        experiment_parser = parser.add_argument_group('Experiment args')
        experiment_parser.add_argument('--data_pkl', type=str, default='./results/dataset_pkls/',
                                       help='location of the pickled data corpus')
        experiment_parser.add_argument('--max_step', type=int, default=200000,
                                       help='upper epoch limit')
        experiment_parser.add_argument('--batch_size', type=int, default=32,
                                       help='batch size')
        experiment_parser.add_argument('--tgt_len', type=int, default=150,
                                       help='number of tokens to predict')
        experiment_parser.add_argument('--eval_tgt_len', type=int, default=150,
                                       help='number of tokens to predict for evaluation')
        experiment_parser.add_argument('--ext_len', type=int, default=0,
                                       help='length of the extended context')
        experiment_parser.add_argument('--mem_len', type=int, default=64,
                                       help='length of the retained previous heads')
        experiment_parser.add_argument('--seed', type=int, default=1111,
                                       help='random seed')
        experiment_parser.add_argument('--no_cuda', action='store_true',
                                       help='don\'t use CUDA')
        experiment_parser.add_argument('--pre_lnorm', action='store_true',
                                       help='apply LayerNorm to the input instead of the output')
        experiment_parser.add_argument('--varlen', action='store_true',
                                       help='use variable length')
        experiment_parser.add_argument('--restart', action='store_true',
                                       help='restart training from the saved checkpoint')
        experiment_parser.add_argument('--restart_dir', type=str, default='',
                                       help='restart dir')
        experiment_parser.add_argument('--clamp_len', type=int, default=-1,
                                       help='use the same pos embeddings after clamp_len')
        experiment_parser.add_argument('--max_eval_steps', type=int, default=-1,
                                       help='max eval steps')
        # fp16 is named in configs/train_model.yml but not used in this script;
        # accept it so configargparse does not reject the existing config.
        experiment_parser.add_argument('--fp16', type=str, default='False',
                                       help='[unused] kept for config compatibility')

        self.update_args_defaults(parser)

        self.args = parser.parse_args(in_args)
        self.args.tied = not self.args.not_tied
        self.args.cuda = not self.args.no_cuda

        if self.args.d_embed < 0:
            self.args.d_embed = self.args.d_model

        assert self.args.ext_len >= 0, 'extended context length must be non-negative'

        if not self.args.no_timestamp:
            if self.args.save_name:
                self.args.work_dir = os.path.join(self.args.work_dir,
                                                  self.args.save_name + '_' + time.strftime('%Y%m%d-%H%M%S'))
            else:
                self.args.work_dir = os.path.join(self.args.work_dir, time.strftime('%Y%m%d-%H%M%S'))

    def update_args_defaults(self, parser):
        pass

    def load_data(self):
        self.corpus = JazzCorpus(self.args.data_pkl, transpose=True)

        # eval_batch_size = 10
        self.tr_iter = self.corpus.get_iterator('train', self.args.batch_size, self.args.tgt_len,
                                                device=self.device, ext_len=self.args.ext_len)
        self.va_iter = self.corpus.get_iterator('val', self.args.batch_size, self.args.eval_tgt_len,
                                                device=self.device, ext_len=self.args.ext_len)

    def build_model(self):
        args = self.args
        model_kwargs = {'n_layer': args.n_layer, 'n_head': args.n_head, 'd_model': args.d_model,
                        'd_head': args.d_head, 'd_inner': args.d_inner, 'dropout': args.dropout,
                        'dropatt': args.dropatt, 'tie_weight': args.tied, 'd_embed': args.d_embed,
                        'pre_lnorm': args.pre_lnorm,
                        'tgt_len': args.tgt_len, 'ext_len': args.ext_len, 'mem_len': args.mem_len,
                        'clamp_len': args.clamp_len,
                        'pitch_sizes': (130, args.pitch_emsize),
                        'duration_sizes': (self.corpus.converter.max_durations(), args.dur_emsize),
                        'offset_sizes': (48, args.offset_emsize),
                        'converter': self.corpus.converter,
                        'chord_bias': args.chord_bias,
                        }

        if args.restart:
            with open(os.path.join(args.restart_dir, 'args.json'), 'rb') as f:
                saved_model_kwargs = json.load(f)
            for k, v in saved_model_kwargs.items():
                model_kwargs[k] = v
            self.model = self.model_class(**model_kwargs)
            with open(os.path.join(args.restart_dir, 'model.pt'), 'rb') as f:
                self.model.load_state_dict(torch.load(f), strict=False)
            update_dropout_partial = partial(update_dropout, args=self.args)
            self.model.apply(update_dropout_partial)
            self.model.apply(update_dropout_partial)
        else:
            self.model = self.model_class(**model_kwargs)
            weights_init_partial = partial(weights_init, args=self.args)
            self.model.apply(weights_init_partial)
            self.model.encode_pitch.apply(
                weights_init_partial)  # ensure embedding init is not overridden by out_layer in case of weight sharing
            self.model.encode_duration.apply(
                weights_init_partial)  # ensure embedding init is not overridden by out_layer in case of weight sharing
            if self.model.offset:
                self.model.encode_offset.apply(
                    weights_init_partial)  # ensure embedding init is not overridden by out_layer in case of weight sharing

        with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
            del model_kwargs['converter']
            json.dump(model_kwargs, f)
        shutil.copy(os.path.join(args.data_pkl, 'converter_and_duration.pkl'),
                    os.path.join(args.work_dir, 'converter_and_duration.pkl'))
        shutil.copy(os.path.join(args.config),
                    os.path.join(args.work_dir, os.path.basename(args.config)))

        args.n_all_param = sum([p.nelement() for p in self.model.parameters()])
        args.n_nonemb_param = sum([p.nelement() for p in self.model.layers.parameters()])

        self.para_model = self.model.to(self.device)

        self.logging('=' * 100)
        for k, v in args.__dict__.items():
            self.logging('    - {} : {}'.format(k, v))
        self.logging('=' * 100)
        self.logging('#params = {}'.format(args.n_all_param))
        self.logging('#non emb params = {}'.format(args.n_nonemb_param))

    def init_optimizer(self):
        args = self.args
        model = self.model
        if args.optim.lower() == 'sgd':
            if args.sample_softmax > 0:
                dense_params = []
                for param in model.parameters():
                    dense_params.append(param)
                optimizer = optim.SGD(dense_params, lr=args.lr, momentum=args.mom)
            else:
                optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
        elif args.optim.lower() == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr)
        elif args.optim.lower() == 'adagrad':
            optimizer = optim.Adagrad(model.parameters(), lr=args.lr)
        elif args.optim.lower() == 'ranger':
            from jazz_rnn.utils.ranger import Ranger
            optimizer = Ranger(model.parameters(), lr=args.lr)

        #### scheduler
        if args.scheduler == 'cosine':
            # here we do not set eta_min to lr_min to be backward compatible
            # because in previous versions eta_min is default to 0
            # rather than the default value of lr_min 1e-6
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                             args.max_step,
                                                             eta_min=args.eta_min)  # should use eta_min arg
        elif args.scheduler == 'inv_sqrt':
            # originally used for Transformer (in Attention is all you need)
            def lr_lambda(step):
                # return a multiplier instead of a learning rate
                if step == 0 and args.warmup_step == 0:
                    return 1.
                else:
                    return 1. / (step ** 0.5) if step > args.warmup_step \
                        else step / (args.warmup_step ** 1.5)

            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        elif args.scheduler == 'dev_perf':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             factor=args.decay_rate, patience=args.patience,
                                                             min_lr=args.lr_min)
        elif args.scheduler == 'constant':
            pass

        if args.restart and type(self) is Trainer:
            if os.path.exists(os.path.join(args.restart_dir, 'optimizer.pt')):
                with open(os.path.join(args.restart_dir, 'optimizer.pt'), 'rb') as f:
                    opt_state_dict = torch.load(f)
                    optimizer.load_state_dict(opt_state_dict)
            else:
                print('Optimizer was not saved. Start from scratch.')

        self.optimizer = optimizer
        self.scheduler = scheduler

    def evaluate(self, eval_iter):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        # If the model does not use memory at all, make the ext_len longer.
        # Otherwise, make the mem_len longer and keep the ext_len the same.
        if self.args.mem_len == 0:
            self.model.reset_length(self.args.eval_tgt_len,
                                    self.args.ext_len + self.args.tgt_len - self.args.eval_tgt_len, self.args.mem_len)
        else:
            self.model.reset_length(self.args.eval_tgt_len,
                                    self.args.ext_len, self.args.mem_len + self.args.tgt_len - self.args.eval_tgt_len)

        # total_losses_dict = {'nll': 0., 'loss': 0., 'p_entropy': 0., 'd_entropy': 0., 'total_entropy': 0.}
        total_losses_dict = {k: 0. for k in self.meters_list}
        # Evaluation
        total_len, total_loss = 0, 0.
        with torch.no_grad():
            mems = tuple()
            for i, (data, target, seq_len) in enumerate(eval_iter):
                if 0 < self.args.max_eval_steps <= i:
                    break
                prediction, ret, loss_dict, _ = self.model(data, target, *mems)
                loss, mems = ret[0], ret[1:]
                loss = loss.mean()
                for k, v in loss_dict.items():
                    total_losses_dict[k] += seq_len * float(v)
                total_loss += seq_len * loss.float().item()
                total_len += seq_len

        # Switch back to the training mode
        self.model.reset_length(self.args.tgt_len, self.args.ext_len, self.args.mem_len)
        self.model.train()
        for k, v in total_losses_dict.items():
            total_losses_dict[k] = total_losses_dict[k] / total_len

        return total_loss / total_len, total_losses_dict

    def train(self, epoch):
        # Turn on training mode which enables dropout.
        self.model.train()
        mems = tuple()
        train_iter = self.tr_iter.get_varlen_iter() if self.args.varlen else self.tr_iter

        meters = {k: AverageMeter(k, ':.4e') for k in self.meters_list}

        log_start_time = time.time()
        eval_start_time = time.time()
        for batch, (data, target, seq_len) in enumerate(train_iter):
            self.model.zero_grad()
            prediction, ret, loss_dict, _ = self.para_model(data, target, *mems)
            loss, mems = ret[0], ret[1:]
            loss = loss.float().mean().type_as(loss)
            loss.backward()

            for k in self.meters_list:
                meters[k].update(float(loss_dict[k]), data.shape[1])

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip)

            self.optimizer.step()

            # step-wise learning rate annealing
            self.train_step += 1
            if self.args.scheduler in ['cosine', 'constant', 'dev_perf']:
                # linear warmup stage
                if self.train_step < self.args.warmup_step:
                    curr_lr = self.args.lr * self.train_step / self.args.warmup_step
                    self.optimizer.param_groups[0]['lr'] = curr_lr
                else:
                    if self.args.scheduler == 'cosine':
                        self.scheduler.step(self.train_step)
            elif self.args.scheduler == 'inv_sqrt':
                self.scheduler.step(self.train_step)

            if self.train_step % self.args.log_interval == 0 or self.train_step == 1:
                cur_meters = {k: v.avg for k, v in meters.items()}
                elapsed = time.time() - log_start_time

                meters_str = ' | '.join(['{}: {:5.2f}'.format(k, v) for k, v in cur_meters.items()])
                log_str = '| epoch {:3d} step {:>8d} | {:>6d} batches | lr {:.3g} ' \
                          '| ms/batch {:5.2f} | {}'.format(
                    epoch, self.train_step, self.train_step, self.optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / self.args.log_interval, meters_str)

                self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], global_step=self.train_step)
                self.writer.add_scalar('train_loss', cur_meters['loss'], global_step=self.train_step)
                # writer.flush()
                log_str += ' | ppl {:9.3f}'.format(math.exp(cur_meters['loss']))
                self.logging(log_str)
                for k, v in meters.items():
                    v.reset()
                log_start_time = time.time()

            if self.train_step % self.args.eval_interval == 0 or self.train_step == 1:
                val_loss, val_losses_dict = self.evaluate(self.va_iter)
                self.logging('-' * 100)
                log_str = '| Eval {:3d} at step {:>8d} | time: {:5.2f}s ' \
                          '| valid loss {:5.2f}'.format(
                    self.train_step // self.args.eval_interval, self.train_step,
                    (time.time() - eval_start_time), val_loss)
                log_str += ' | valid nll {:9.3f}'.format(val_losses_dict['nll'])
                self.logging(log_str)
                self.logging('-' * 100)
                for (k, v), (kval, vval) in zip(cur_meters.items(), val_losses_dict.items()):
                    self.writer.add_scalars(k, {f'train_{k}': v,
                                                f'val_{k}': vval},
                                            global_step=self.train_step)
                # Save the model if the validation loss is the best we've seen so far.
                with open(os.path.join(self.args.work_dir, 'model.pt'), 'wb') as f:
                    torch.save(self.model.state_dict(), f)
                with open(os.path.join(self.args.work_dir, 'optimizer.pt'), 'wb') as f:
                    torch.save(self.optimizer.state_dict(), f)
                if not self.best_val_loss or val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    shutil.copy(os.path.join(self.args.work_dir, 'model.pt'),
                                os.path.join(self.args.work_dir, 'model_best.pt'))

                if not self.best_val_p_acc or val_losses_dict['p_top1'] < self.best_val_p_acc:
                    self.best_val_p_acc = val_losses_dict['p_top1']
                    shutil.copy(os.path.join(self.args.work_dir, 'model.pt'),
                                os.path.join(self.args.work_dir, 'model_best_p_acc.pt'))
                if not self.best_val_d_acc or val_losses_dict['d_top1'] < self.best_val_d_acc:
                    self.best_val_d_acc = val_losses_dict['d_top1']
                    shutil.copy(os.path.join(self.args.work_dir, 'model.pt'),
                                os.path.join(self.args.work_dir, 'model_best_d_acc.pt'))

                # dev-performance based learning rate annealing
                if self.args.scheduler == 'dev_perf':
                    self.scheduler.step(val_loss)

                # Resume-aware: persist counters + best_val so a fresh process
                # can continue where it stopped. model.pt remains a pure
                # state_dict (don't bundle metadata into it — inference reads it).
                self._train_time_sec = self._resumed_train_time_sec + (time.time() - self._train_start_time)
                _save_train_state(
                    self.args.work_dir,
                    self.train_step,
                    self.best_val_loss,
                    self.best_val_p_acc,
                    self.best_val_d_acc,
                    self._train_time_sec,
                )
                _append_epoch_csv(
                    self.args.work_dir,
                    {
                        "step": self.train_step,
                        "elapsed_sec": round(self._train_time_sec, 2),
                        "lr": self.optimizer.param_groups[0]["lr"],
                        "val_loss": float(val_loss),
                        "val_nll": float(val_losses_dict.get("nll", 0.0)),
                        "val_p_top1": float(val_losses_dict.get("p_top1", 0.0)),
                        "val_d_top1": float(val_losses_dict.get("d_top1", 0.0)),
                        "val_p_entropy": float(val_losses_dict.get("p_entropy", 0.0)),
                        "val_d_entropy": float(val_losses_dict.get("d_entropy", 0.0)),
                    },
                )

                eval_start_time = time.time()

            if self.train_step == self.args.max_step:
                break

    def main(self):
        # Resume-aware initialization: restore counters from train_state.json
        # if --restart, otherwise start fresh.
        resumed_state = _load_train_state(self.args.restart_dir) if self.args.restart else None
        if resumed_state is not None:
            self.train_step = resumed_state["train_step"]
            self.best_val_loss = resumed_state["best_val_loss"]
            self.best_val_p_acc = resumed_state["best_val_p_acc"]
            self.best_val_d_acc = resumed_state["best_val_d_acc"]
            self._resumed_train_time_sec = resumed_state.get("train_time_sec", 0.0)
            self.logging(
                '[resume] restored train_state: step={}, best_val_loss={}, train_time_sec={:.1f}'.format(
                    self.train_step, self.best_val_loss, self._resumed_train_time_sec,
                )
            )
        else:
            self.train_step = 0
            self.best_val_loss = None
            self.best_val_p_acc = 0
            self.best_val_d_acc = 0
            self._resumed_train_time_sec = 0.0

        self._train_start_time = time.time()
        self._train_time_sec = self._resumed_train_time_sec

        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in itertools.count(start=1):
                self.train(epoch)
                if self.train_step == self.args.max_step:
                    self.logging('-' * 100)
                    self.logging('End of training')
                    break
            self.writer.close()
        except KeyboardInterrupt:
            self.logging('-' * 100)
            self.logging('Exiting from training early')
            with open(os.path.join(self.args.work_dir, 'model_latest.pt'), 'wb') as f:
                torch.save(self.model.state_dict(), f)
            with open(os.path.join(self.args.work_dir, 'optimizer_latest.pt'), 'wb') as f:
                torch.save(self.optimizer.state_dict(), f)
            self.writer.close()
            return

        # Training completed cleanly: load model_best.pt, run a final
        # evaluation on val_iter, and write summary.json. val.pkl contains
        # split.json[eval]=43 (used for best-checkpoint selection during
        # training, never for gradient steps). The canonical test=40
        # (split.json[test]) is held out from this whole script — it is
        # evaluated separately by evaluate_canonical.py against model_best.pt
        # and produces final_test_* fields appended to summary.json.
        best_path = os.path.join(self.args.work_dir, 'model_best.pt')
        if os.path.isfile(best_path):
            self.model.load_state_dict(torch.load(best_path), strict=False)
            self.logging('[summary] loaded model_best.pt for final eval')
        else:
            self.logging('[summary] model_best.pt not found — using last weights')

        final_val_loss, final_val_dict = self.evaluate(self.va_iter)
        self._train_time_sec = self._resumed_train_time_sec + (time.time() - self._train_start_time)

        summary = {
            'config': {k: v for k, v in self.args.__dict__.items() if not k.startswith('_')},
            'n_all_param': int(self.args.n_all_param),
            'n_nonemb_param': int(self.args.n_nonemb_param),
            'train_time_sec': round(self._train_time_sec, 2),
            'completed_step': int(self.train_step),
            'max_step': int(self.args.max_step),
            'best_val_loss': float(self.best_val_loss) if self.best_val_loss is not None else None,
            'best_val_ppl': math.exp(self.best_val_loss) if self.best_val_loss is not None else None,
            'final_val_loss': float(final_val_loss),
            'final_val_ppl': math.exp(final_val_loss),
            'final_val_metrics': {k: float(v) for k, v in final_val_dict.items()},
            'note': 'val.pkl = split.json[eval]=43 (used only for best-checkpoint '
                    'selection); canonical test=40 is held out and evaluated '
                    'separately by evaluate_canonical.py (final_test_* fields '
                    'are appended to this summary by that script)',
        }
        with open(os.path.join(self.args.work_dir, SUMMARY_FILE), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        self.logging(
            '[summary] wrote {} (best_val_ppl={:.3f}, final_val_ppl={:.3f})'.format(
                SUMMARY_FILE, summary['best_val_ppl'], summary['final_val_ppl'],
            )
        )


if __name__ == '__main__':
    trainer = Trainer(sys.argv[1:])
    trainer.main()
