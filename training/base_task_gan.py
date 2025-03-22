import logging
import os
import pathlib
import sys
from typing import Dict

import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch.utils.data
from lightning.pytorch.utilities.rank_zero import rank_zero_debug, rank_zero_info, rank_zero_only
from torch import nn
# torch.nn.utils.weight_norm
from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric

import utils
# from utils.indexed_datasets import IndexedDataset
from utils.training_utils import (
    DsBatchSampler, DsEvalBatchSampler,
    get_latest_checkpoint_path
)

matplotlib.use('Agg')

torch.multiprocessing.set_sharing_strategy(os.getenv('TORCH_SHARE_STRATEGY', 'file_system'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')


class BaseDataset(Dataset):
    """
        Base class for datasets.
        1. *sizes*:
            clipped length if "max_frames" is set;
        2. *num_frames*:
            unclipped length.

        Subclasses should define:
        1. *collate*:
            take the longest data, pad other data to the same length;
        2. *__getitem__*:
            the index function.
    """

    def __init__(self, config: dict, data_dir, prefix, allow_aug=False):
        super().__init__()
        self.config = config
        self.prefix = prefix
        self.data_dir = data_dir if isinstance(data_dir, pathlib.Path) else pathlib.Path(data_dir)
        self.sizes = np.load(self.data_dir / f'{self.prefix}.lengths')
        self.indexed_ds = IndexedDataset(self.data_dir, self.prefix)
        self.allow_aug = allow_aug

    @property
    def _sizes(self):
        return self.sizes

    def __getitem__(self, index):
        return self.indexed_ds[index]

    def __len__(self):
        return len(self._sizes)

    def num_frames(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return self._sizes[index]

    def collater(self, samples):
        return {
            'size': len(samples)
        }





class GanBaseTask(pl.LightningModule):
    """
        Base class for training tasks.
        1. *load_ckpt*:
            load checkpoint;
        2. *training_step*:
            record and log the loss;
        3. *optimizer_step*:
            run backwards step;
        4. *start*:
            load training configs, backup code, log to tensorboard, start training;
        5. *configure_ddp* and *init_ddp_connection*:
            start parallel training.

        Subclasses should define:
        1. *build_model*, *build_optimizer*, *build_scheduler*:
            how to build the model, the optimizer and the training scheduler;
        2. *_training_step*:
            one training step of the model;
        3. *on_validation_end* and *_on_validation_end*:
            postprocess the validation output.
    """

    def __init__(self, config: dict, *args, **kwargs):
        # dataset configs
        super().__init__(*args, **kwargs)
        self.dataset_cls = None
        self.config = config
        # self.max_batch_frames = self.config['max_batch_frames']
        # self.max_batch_size = self.config['max_batch_size']
        # self.max_val_batch_frames = self.config['max_val_batch_frames']
        # self.max_val_batch_size = self.config['max_val_batch_size']

        # self.accumulate_grad_batches = self.config['accumulate_grad_batches']
        self.clip_grad_norm = self.config['clip_grad_norm']

        self.training_sampler = None
        self.model = None
        self.generator = None
        self.discriminator = None
        self.skip_immediate_validation = False
        self.skip_immediate_ckpt_save = False

        self.valid_losses: Dict[str, Metric] = {
            'total_loss': MeanMetric()
        }
        self.valid_metric_names = set()
        self.mix_loss = None

        self.automatic_optimization = False
        self.skip_immediate_validations = 0

        self.aux_step = self.config.get('aux_step')
        self.train_dataset = None
        self.valid_dataset = None

    ###########

    # Training, validation and testing
    ###########
    def setup(self, stage):
        self.model = self.build_model()
        self.unfreeze_all_params()
        if self.config['freezing_enabled']:
            self.freeze_params()
        if self.config['finetune_enabled'] and get_latest_checkpoint_path(
                pathlib.Path(self.config['work_dir'])) is None:
            self.load_finetune_ckpt(self.load_pre_train_model())
        self.print_arch()
        self.build_losses_and_metrics()
        self.build_dataset()
        # self.train_dataset = self.dataset_cls(
        #     config=self.config, data_dir=self.config['binary_data_dir'],
        #     prefix=self.config['train_set_name'], allow_aug=True
        # )
        # self.valid_dataset = self.dataset_cls(
        #     config=self.config, data_dir=self.config['binary_data_dir'],
        #     prefix=self.config['valid_set_name'], allow_aug=False
        # )

    def build_dataset(self):
        raise NotImplementedError()

    def get_need_freeze_state_dict_key(self, model_state_dict) -> list:
        key_list = []
        for i in self.config['frozen_params']:
            for j in model_state_dict:
                if j.startswith(i):
                    key_list.append(j)
        return list(set(key_list))

    def freeze_params(self) -> None:
        model_state_dict = self.state_dict().keys()
        freeze_key = self.get_need_freeze_state_dict_key(model_state_dict=model_state_dict)

        for i in freeze_key:
            params = self.get_parameter(i)

            params.requires_grad = False

    def unfreeze_all_params(self) -> None:
        for i in self.parameters():
            i.requires_grad = True

    def load_finetune_ckpt(
            self, state_dict
    ) -> None:

        adapt_shapes = self.config['finetune_strict_shapes']
        if not adapt_shapes:
            cur_model_state_dict = self.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print('| Unmatched keys: ', key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        self.load_state_dict(state_dict, strict=False)

    def load_pre_train_model(self):

        pre_train_ckpt_path = self.config.get('finetune_ckpt_path')
        blacklist = self.config.get('finetune_ignored_params')
        if blacklist is None:
            blacklist = []
        # if whitelist is  None:
        #     raise RuntimeError("")

        if pre_train_ckpt_path is not None:
            ckpt = torch.load(pre_train_ckpt_path)

            state_dict = {}
            for i in ckpt['state_dict']:
                # if 'diffusion' in i:
                # if i in rrrr:
                #     continue
                skip = False
                for b in blacklist:
                    if i.startswith(b):
                        skip = True
                        break

                if skip:
                    continue

                state_dict[i] = ckpt['state_dict'][i]
            return state_dict
        else:
            raise RuntimeError("")

    def build_model(self):
        raise NotImplementedError()

    @rank_zero_only
    def print_arch(self):
        utils.print_arch(self)

    def build_losses_and_metrics(self):
        raise NotImplementedError()

    def register_metric(self, name: str, metric: Metric):
        assert isinstance(metric, Metric)
        setattr(self, name, metric)
        self.valid_metric_names.add(name)

    # def run_model(self, sample, infer=False):
    #     """
    #     steps:
    #         1. run the full model
    #         2. calculate losses if not infer
    #     """
    #     raise NotImplementedError()

    def Gforward(self, sample, infer=False):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        raise NotImplementedError()

    def Dforward(self, Goutput):
        """
        steps:
            1. run the full model
            2. calculate losses if not infer
        """
        raise NotImplementedError()

    # def on_train_epoch_start(self):
    #     if self.training_sampler is not None:
    #         self.training_sampler.set_epoch(self.current_epoch)

    def _training_step(self, sample, batch_idx):
        """
        :return: total loss: torch.Tensor, loss_log: dict, other_log: dict

        """
        aux_only = False
        if self.aux_step is not None:
            if self.aux_step > self.global_step:
                aux_only = True

        log_diet = {}
        opt_g, opt_d = self.optimizers()
        Goutput = self.Gforward(sample=sample)
        if not aux_only:
            Dfake = self.Dforward(Goutput=Goutput['audio'].detach())
            Dtrue = self.Dforward(Goutput=sample['audio'])
            Dloss, Dlog = self.mix_loss.Dloss(Dfake=Dfake, Dtrue=Dtrue)
            log_diet.update(Dlog)
            # if self.clip_grad_norm is not None:
            #     self.manual_backward(Dloss/self.clip_grad_norm)
            # else:
            opt_d.zero_grad()
            self.manual_backward(Dloss)
            if self.clip_grad_norm is not None:
                self.clip_gradients(opt_d, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
            opt_d.step()
            opt_d.zero_grad()
        if not aux_only:
            GDfake = self.Dforward(Goutput=Goutput['audio'])
            GDtrue = self.Dforward(Goutput=sample['audio'])
            GDloss, GDlog = self.mix_loss.GDloss(GDfake=GDfake,GDtrue=GDtrue)
            log_diet.update(GDlog)
        Auxloss, Auxlog = self.mix_loss.Auxloss(Goutput=Goutput, sample=sample)

        log_diet.update(Auxlog)
        if not aux_only:
            Gloss=GDloss + Auxloss
        else:
            Gloss=Auxloss

        # if self.clip_grad_norm is not None:
        #     self.manual_backward(Gloss / self.clip_grad_norm)
        # else:
        #     self.manual_backward(Gloss)
        # if (batch_idx + 1) % self.accumulate_grad_batches == 0:
        opt_g.zero_grad()
        self.manual_backward(Gloss)
        if self.clip_grad_norm is not None:
            self.clip_gradients(opt_g, gradient_clip_val=self.clip_grad_norm, gradient_clip_algorithm="norm")
        opt_g.step()



        return log_diet

    def training_step(self, sample, batch_idx, ):  # todo
        log_outputs = self._training_step(sample, batch_idx)

        # logs to progress bar
        self.log_dict({'loss':sum(log_outputs.values())}, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # self.log('lr', self.lr_schedulers().get_last_lr()[0], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # logs to tensorboard
        if self.global_step % self.config['log_interval'] == 0:
            tb_log = {f'training/{k}': v for k, v in log_outputs.items()}
            # tb_log['training/lr'] = self.lr_schedulers().get_last_lr()[0]
            self.logger.log_metrics(tb_log, step=self.global_step)
        #
        # return total_loss

    # def on_before_optimizer_step(self, *args, **kwargs):
    #     self.log_dict(grad_norm(self, norm_type=2))

    def _on_validation_start(self):
        pass

    def on_validation_start(self):
        self._on_validation_start()
        for metric in self.valid_losses.values():
            metric.to(self.device)
            metric.reset()

    def _validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:
        :return: loss_log: dict, weight: int
        """
        raise NotImplementedError()

    def validation_step(self, sample, batch_idx):
        """

        :param sample:
        :param batch_idx:

        """

        # if self.skip_immediate_validations == 0 and self.global_step != 0:
        #     self.skip_immediate_validation = True
        #     self.skip_immediate_validations = 1
        # if self.global_step == 0:
        #     self.skip_immediate_validations = 1

        if self.skip_immediate_validation:
            rank_zero_debug(f"Skip validation {batch_idx}")
            return {}
        with torch.autocast(self.device.type, enabled=False):
            losses, weight = self._validation_step(sample, batch_idx)
        losses = {
            'total_loss': sum(losses.values()),
            **losses
        }
        for k, v in losses.items():
            if k not in self.valid_losses:
                self.valid_losses[k] = MeanMetric().to(self.device)
            self.valid_losses[k].update(v, weight=weight)  # weight=1
        return losses

    def on_validation_epoch_end(self):
        if self.skip_immediate_validation:
            self.skip_immediate_validation = False
            self.skip_immediate_ckpt_save = True
            return
        loss_vals = {k: v.compute() for k, v in self.valid_losses.items()}
        self.log('val_loss', loss_vals['total_loss'], on_epoch=True, prog_bar=True, logger=False, sync_dist=True)
        self.logger.log_metrics({f'validation/{k}': v for k, v in loss_vals.items()}, step=self.global_step)
        for metric in self.valid_losses.values():
            metric.reset()
        metric_vals = {k: getattr(self, k).compute() for k in self.valid_metric_names}
        self.logger.log_metrics({f'metrics/{k}': v for k, v in metric_vals.items()}, step=self.global_step)
        for metric_name in self.valid_metric_names:
            getattr(self, metric_name).reset()

    # noinspection PyMethodMayBeStatic
    def build_scheduler(self, optimizer):
        from utils import build_lr_scheduler_from_config

        scheduler_args = self.config['lr_scheduler_args']
        assert scheduler_args['scheduler_cls'] != ''
        scheduler = build_lr_scheduler_from_config(optimizer, scheduler_args)
        return scheduler

    # noinspection PyMethodMayBeStatic
    def build_optimizer(self, model, optimizer_args):
        from utils import build_object_from_class_name

        assert optimizer_args['optimizer_cls'] != ''
        if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
            optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])

        if isinstance(model, nn.ModuleList):
            parameterslist = []
            for i in model:
                parameterslist = parameterslist + list(i.parameters())
            optimizer = build_object_from_class_name(
                optimizer_args['optimizer_cls'],
                torch.optim.Optimizer,
                parameterslist,
                **optimizer_args
            )
        elif isinstance(model, nn.ModuleDict):
            parameterslist = []
            for i in model:
                # parameterslist = parameterslist + list(model[i].parameters())
                parameterslist.append({'params': model[i].parameters()})
            optimizer = build_object_from_class_name(
                optimizer_args['optimizer_cls'],
                torch.optim.Optimizer,
                parameterslist,
                **optimizer_args
            )
        elif isinstance(model, nn.Module):

            optimizer = build_object_from_class_name(
                optimizer_args['optimizer_cls'],
                torch.optim.Optimizer,
                model.parameters(),
                **optimizer_args
            )
        else:
            raise RuntimeError("")

        return optimizer

    def configure_optimizers(self):
        optG = self.build_optimizer(self.generator, optimizer_args=self.config['generater_optimizer_args'])
        optD = self.build_optimizer(self.discriminator, optimizer_args=self.config['discriminate_optimizer_args'])

        return [optG, optD]
        # scheduler = self.build_scheduler(optm)
        # if scheduler is None:
        #     return optm
        # return {
        #     "optimizer": optm,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "step",
        #         "frequency": 1
        #     }
        # }

    def train_dataloader(self):
        # self.training_sampler = DsBatchSampler(
        #     self.train_dataset,
        #     max_batch_frames=self.max_batch_frames,
        #     max_batch_size=self.max_batch_size,
        #     num_replicas=(self.trainer.distributed_sampler_kwargs or {}).get('num_replicas', 1),
        #     rank=(self.trainer.distributed_sampler_kwargs or {}).get('rank', 0),
        #     sort_by_similar_size=self.config['sort_by_len'],
        #     required_batch_count_multiple=self.config['accumulate_grad_batches'],
        #     frame_count_grid=self.config['sampler_frame_count_grid'],
        #     shuffle_sample=True,
        #     shuffle_batch=False,
        #     seed=self.config['seed']
        # )
        return torch.utils.data.DataLoader(self.train_dataset,
                                           collate_fn=self.train_dataset.collater,
                                           batch_size=self.config['batch_size'],
                                           # batch_sampler=self.training_sampler,
                                           num_workers=self.config['ds_workers'],
                                           prefetch_factor=self.config['dataloader_prefetch_factor'],
                                           pin_memory=True,
                                           persistent_workers=True)

    def val_dataloader(self):
        # sampler = DsEvalBatchSampler(
        #     self.valid_dataset,
        #     max_batch_frames=self.max_val_batch_frames,
        #     max_batch_size=self.max_val_batch_size,
        #     rank=(self.trainer.distributed_sampler_kwargs or {}).get('rank', 0),
        #     batch_by_size=False
        # )
        return torch.utils.data.DataLoader(self.valid_dataset,
                                           collate_fn=self.valid_dataset.collater,
                                           batch_size=1,
                                           # batch_sampler=sampler,
                                           num_workers=self.config['ds_workers'],
                                           prefetch_factor=self.config['dataloader_prefetch_factor'],
                                           shuffle=False)

    def test_dataloader(self):
        return self.val_dataloader()

    def on_test_start(self):
        self.on_validation_start()

    def test_step(self, sample, batch_idx):
        return self.validation_step(sample, batch_idx)

    def on_test_end(self):
        return self.on_validation_end()

    def on_save_checkpoint(self, checkpoint):
        pass
        # checkpoint['trainer_stage'] = self.trainer.state.stage.value

    # def on_load_checkpoint(self, checkpoint):
    #     # from lightning.pytorch.trainer.states import RunningStage
    #     from utils import simulate_lr_scheduler
    #     # if checkpoint.get('trainer_stage', '') == RunningStage.VALIDATING.value:
    #     #     self.skip_immediate_validation = True
    #
    #     optimizer_args = self.config['optimizer_args']
    #     scheduler_args = self.config['lr_scheduler_args']
    #
    #     if 'beta1' in optimizer_args and 'beta2' in optimizer_args and 'betas' not in optimizer_args:
    #         optimizer_args['betas'] = (optimizer_args['beta1'], optimizer_args['beta2'])
    #
    #     if checkpoint.get('optimizer_states', None):
    #         opt_states = checkpoint['optimizer_states']
    #         assert len(opt_states) == 1  # only support one optimizer
    #         opt_state = opt_states[0]
    #         for param_group in opt_state['param_groups']:
    #             for k, v in optimizer_args.items():
    #                 if k in param_group and param_group[k] != v:
    #                     if 'lr_schedulers' in checkpoint and checkpoint['lr_schedulers'] and k == 'lr':
    #                         continue
    #                     rank_zero_info(f'| Overriding optimizer parameter {k} from checkpoint: {param_group[k]} -> {v}')
    #                     param_group[k] = v
    #             if 'initial_lr' in param_group and param_group['initial_lr'] != optimizer_args['lr']:
    #                 rank_zero_info(
    #                     f'| Overriding optimizer parameter initial_lr from checkpoint: {param_group["initial_lr"]} -> {optimizer_args["lr"]}'
    #                 )
    #                 param_group['initial_lr'] = optimizer_args['lr']
    #
    #     if checkpoint.get('lr_schedulers', None):
    #         assert checkpoint.get('optimizer_states', False)
    #         assert len(checkpoint['lr_schedulers']) == 1  # only support one scheduler
    #         checkpoint['lr_schedulers'][0] = simulate_lr_scheduler(
    #             optimizer_args, scheduler_args,
    #             step_count=checkpoint['global_step'],
    #             num_param_groups=len(checkpoint['optimizer_states'][0]['param_groups'])
    #         )
    #         for param_group, new_lr in zip(
    #                 checkpoint['optimizer_states'][0]['param_groups'],
    #                 checkpoint['lr_schedulers'][0]['_last_lr'],
    #         ):
    #             if param_group['lr'] != new_lr:
    #                 rank_zero_info(
    #                     f'| Overriding optimizer parameter lr from checkpoint: {param_group["lr"]} -> {new_lr}')
    #                 param_group['lr'] = new_lr
