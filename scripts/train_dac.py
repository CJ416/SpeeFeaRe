import os
import sys
import yaml
import warnings
from dataclasses import dataclass
from pathlib import Path
from transformers import WhisperTokenizer

import numpy as np

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator

from audiotools import AudioSignal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))
from speefeare import DAC, Discriminator, TextAudioSpeakerLoader, TextAudioCollate
from speefeare.utils import HParams
from speefeare.models.DAC.vq import loss as dac_losses

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False

np.set_printoptions(threshold=sys.maxsize)
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val
def get_grad_norm(model):
    total_norm = 0
    for name,p in model.named_parameters():
        try:
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            else:
                # print(name)
                continue
        except Exception as e:
            print(e)
            print(name)
    total_norm = total_norm ** (1. / 2) 
    return total_norm
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def cycle(dl):
    while True:
        for data in dl:
            yield data
def clip_grad_value_(parameters, clip_value, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    if clip_value is not None:
        clip_value = float(clip_value)

    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
        if clip_value is not None:
            p.grad.data.clamp_(min=-clip_value, max=clip_value)
    total_norm = total_norm ** (1.0 / norm_type)
    return total_norm

class Trainer(object):
    def __init__(self, cfg_path):
        super().__init__()
        self.accelerator = Accelerator()
        with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)

        hps = HParams(**config)
        self.hps = hps
        

        dataset = TextAudioSpeakerLoader(self.hps)
        collate_fn = TextAudioCollate()
        self.dataloader = DataLoader(
            dataset,
            batch_size=hps.train.batch_size,
            num_workers=hps.train.num_workers,
            shuffle=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            drop_last=True,
        )
        self.train_steps = self.hps.train.train_steps
        self.val_freq = self.hps.train.val_freq

        self.vq = DAC(**self.hps.architecture)
        self.discriminator = Discriminator(**self.hps.discriminator)

        # Initialize optimizers
        self.vq_optimizer = AdamW(self.vq.parameters(), self.hps.train.learning_rate, betas=self.hps.train.betas, eps=self.hps.train.eps)
        self.discriminator_optimizer = AdamW(self.discriminator.parameters(), self.hps.train.learning_rate, betas=self.hps.train.betas, eps=self.hps.train.eps)  # discriminator 优化器

        # Initialize schedulers 
        self.scheduler_vq = torch.optim.lr_scheduler.ExponentialLR(self.vq_optimizer, gamma=self.hps.train.lr_decay, last_epoch=-1)
        self.scheduler_discriminator = torch.optim.lr_scheduler.ExponentialLR(self.discriminator_optimizer, gamma=self.hps.train.lr_decay, last_epoch=-1)  # discriminator 调度器

        # Prepare model and optimizers for training accelerator
        self.vq, self.vq_optimizer, self.discriminator, \
        self.discriminator_optimizer, self.dataloader = self.accelerator.prepare(
            self.vq, self.vq_optimizer,
            self.discriminator, self.discriminator_optimizer,
            self.dataloader
        )

        self.dataloader = cycle(self.dataloader)

        self.step = 0
        self.epoch = 1
        self.gradient_accumulate_every = hps.train.gradient_accumulate_every
        self.logs_folder = Path(hps.train.logs_folder)

    def load(self, model_paths):
        accelerator = self.accelerator
        device = accelerator.device

        if model_paths['vq'] is not None:
            # 加载 VQ
            vq_data = torch.load(model_paths['vq'], map_location=device)
            vq_state_dict = vq_data['vq']
            # vq_opt_state_dict = vq_data['vq_opt']

            vq = accelerator.unwrap_model(self.vq)
            current_vq_dict = vq.state_dict()
            vq_state_dict = {
                k: v if v.size() == current_vq_dict[k].size() else current_vq_dict[k]
                for k, v in zip(current_vq_dict.keys(), vq_state_dict.values())
            }
            vq.load_state_dict(vq_state_dict, strict=False)

        if model_paths['discriminator'] is not None:
            # 加载 Discriminator
            discriminator_data = torch.load(model_paths['discriminator'], map_location=device)
            discriminator_state_dict = discriminator_data['discriminator']
            # discriminator_opt_state_dict = discriminator_data['discriminator_opt']

            discriminator = accelerator.unwrap_model(self.discriminator)
            current_discriminator_dict = discriminator.state_dict()
            discriminator_state_dict = {
                k: v if v.size() == current_discriminator_dict[k].size() else current_discriminator_dict[k]
                for k, v in zip(current_discriminator_dict.keys(), discriminator_state_dict.values())
            }
            discriminator.load_state_dict(discriminator_state_dict, strict=False)

    def train_one_step(self, pbar, waveform_loss, stft_loss, mel_loss, gan_loss): 
        
        accelerator = self.accelerator
        device = accelerator.device
        batch = next(self.dataloader)
        lambdas = self.hps.lambdas
        self.vq.train()
        self.discriminator.train()
        output = {}

        signal = AudioSignal(batch['wav'], self.hps.data.sampling_rate)
        
        with accelerator.autocast():
            out = self.vq(signal.audio_data, self.hps.data.sampling_rate)
            recons = AudioSignal(out['audio'], self.hps.data.sampling_rate)
            commitment_loss = out['vq/commitment_loss']
            codebook_loss = out['vq/codebook_loss']
            signal = signal.to(device)
            recons = recons.to(device)
            output['adv/disc_loss'] = gan_loss.discriminator_loss(recons, signal)

        self.discriminator_optimizer.zero_grad()
        accelerator.backward(output['adv/disc_loss'])
        output['other/grad_norm_discriminator'] = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 10.0)
        self.discriminator_optimizer.step()
        self.scheduler_discriminator.step()

        vq = self.accelerator.unwrap_model(self.vq)
        import pdb
        pdb.set_trace()
        with accelerator.autocast():
            signal = signal.to(device)
            recons = recons.to(device)
            signal = signal[:, :, :recons.shape[-1]]
            output["stft/loss"] = stft_loss(recons, signal)
            output['mel/loss'] = mel_loss(recons, signal)
            output['waveform_loss'] = waveform_loss(recons, signal)
            output['adv/gen_loss'], output['adv/feat_loss'] = gan_loss.generator_loss(recons, signal)
            output['vq/commitment_loss'] = commitment_loss
            output['vq/codebook_loss'] = codebook_loss
            output['loss'] = sum([v * output[k] for k, v in lambdas.items() if k in output])

        self.vq_optimizer.zero_grad()
        accelerator.backward(output['loss'])
        output['other/grad_norm'] = torch.nn.utils.clip_grad_norm_(self.vq.parameters(), 10.0)
        self.vq_optimizer.step()
        self.scheduler_vq.step()

        output['other/learning_rate'] = self.vq_optimizer.param_groups[0]['lr']
        pbar.set_description(f'G_loss: {output["loss"]:.4f} D_loss: {output["adv/disc_loss"]:.4f} lr: {output["other/learning_rate"]:.5f}')
        if self.step % self.val_freq == 0 and self.accelerator.is_main_process:
            eval_model = self.accelerator.unwrap_model(self.vq)
            eval_model.eval()
            with torch.no_grad():
                wav_eval = eval_model(
                    batch['raw_wav'][:, :, :48000], self.hps.data.sampling_rate
                )['audio']
            eval_model.train()
            milestone = self.step // self.val_freq
            torchaudio.save(str(self.logs_folder / f'sample-{milestone}.wav'), wav_eval[0].detach().cpu(), self.hps.data.sampling_rate)
            torchaudio.save(str(self.logs_folder / f'original-{milestone}.wav'), batch['raw_wav'][:, :, :48000][0].cpu(), self.hps.data.sampling_rate)
        return {k: v for k, v in sorted(output.items())}
    
        
    def train(self):
        
        accelerator = self.accelerator
        device = accelerator.device

        waveform_loss = dac_losses.L1Loss()
        stft_loss = dac_losses.MultiScaleSTFTLoss(**self.hps.MultiScaleSTFTLoss)
        mel_loss = dac_losses.MelSpectrogramLoss(**self.hps.MelSpectrogramLoss)
        gan_loss = dac_losses.GANLoss(self.discriminator)

        #-------------------------#
        #---Distill module Loss---#        
        #-------------------------#

        with tqdm(initial=self.step, total=self.train_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_steps:

                self.train_one_step(pbar, waveform_loss, stft_loss, mel_loss, gan_loss)

                if accelerator.is_main_process and self.step % self.hps.train.save_freq == 0:
                    save_path = f"{self.hps.train.checkpoint_dir}/dac_step_{self.step}.pt"
                    accelerator.wait_for_everyone()
                    unwrapped_vq = accelerator.unwrap_model(self.vq)
                    torch.save(
                        {
                            "vq": unwrapped_vq.state_dict(),
                            "vq_optimizer": self.vq_optimizer.state_dict(),
                            "discriminator": accelerator.unwrap_model(self.discriminator).state_dict(),
                            "discriminator_optimizer": self.discriminator_optimizer.state_dict(),
                            "step": self.step,
                            "epoch": self.epoch,
                        },
                        save_path,
                    )
                    print(f"Saved checkpoint to: {save_path}")
                self.step += 1
                pbar.update(1)
        accelerator.print("Training complete.")

if __name__ == "__main__":
    config_path = '../speefeare/models/DAC/config/dac25hz_16.yaml'
    trainer = Trainer(cfg_path=config_path)
   
    trainer.train()

    

                


        









