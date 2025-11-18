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
from speefeare import DAC, TextAudioSpeakerLoader, TextAudioCollate
from speefeare.utils import HParams
from speefeare.models.DAC.vq import loss as dac_losses


def compute_codebook_utilization(model: DAC, dataloader: DataLoader, device: torch.device):
    model.eval()
    codebook_size = model.quantizer.codebook_size
    usage_counts = torch.zeros(codebook_size, dtype=torch.int32)
    import pdb; pdb.set_trace()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating codebook utilization"):
            audio = batch['wav'].to(device)  # [B, 1, T]
            latents, codes, _, _, _ = model.encode(audio)  # [B, D, T']

            for b in range(codes.size(0)):
                unique_indices = torch.unique(codes[b])
                usage_counts[unique_indices] += 1

    utilized_codebooks = (usage_counts > 0).sum().item()
    utilization_rate = utilized_codebooks / codebook_size

    return utilization_rate, utilized_codebooks, codebook_size


if __name__ == "__main__":

    config_path = '../speefeare/models/DAC/config/dac25hz_16.yaml'

    # Load model config
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    hps = HParams(**config_dict)

    # Initialize model
    model = DAC(**hps.architecture)
    checkpoint = None

    # Load checkpoint
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # Prepare dataset and dataloader
    dataset = TextAudioSpeakerLoader(hps)
    collate_fn = TextAudioCollate()
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Compute codebook utilization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    utilization_rate, utilized_codebooks, total_codebooks = compute_codebook_utilization(model, dataloader, device)

    print(f"Codebook Utilization Rate: {utilization_rate*100:.2f}% ({utilized_codebooks}/{total_codebooks})")