import os
import random
import numpy as np
import torch
import torchaudio
import torch.utils.data
import torchaudio.functional as F

from torch.utils.data import Dataset, DataLoader
from transformers import WhisperTokenizer
from audiotools import AudioSignal
from audiotools import STFTParams
import yaml
import glob
import json
from tqdm import tqdm
import h5py

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

from speefeare.utils import mel_spectrogram_torch
from speefeare.utils import HParams

def read_jsonl(path):
    with open(path, 'r') as f:
        json_str = f.read()
    
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list

def read_hdf5(path):
    with h5py.File(path, 'r') as hdf:
        paths = hdf['path'][:]
        texts = hdf['text'][:]
    return paths, texts

class TextAudioSpeakerLoader(Dataset):
    '''
        1) loads audio, text, speaker_id
        2) normalizes text and converts them to sequences of IDs
        3) computes spectrograms from audio files.
    '''
    def __init__(self, hparams, all_in_mem: bool = False, vol_aug: bool = True):
        super().__init__()
        self.tokenizer = WhisperTokenizer.from_pretrained(hparams.tokenizer_path)
        self.audiopaths_and_text = []
        
        self.train_paths = hparams.data.training_files

        if self.train_paths.endswith('.jsonl'):
            self.audiopaths_and_text = read_jsonl(self.train_paths)
        else:
            self.hdf = h5py.File(self.train_paths, 'r')
            self.audiopaths = self.hdf['paths']
            self.texts = self.hdf['texts']
            self.length = len(self.texts)
            self.indices = list(range(self.length))
            random.shuffle(self.indices)
        
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.n_mel_channels = hparams.data.n_mel_channels
        self.mel_fmin = hparams.data.mel_fmin
        self.mel_fmax = hparams.data.mel_fmax
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.sampling_rate = hparams.data.sampling_rate
        self.segment_size = hparams.data.segment_size
        self.stft_params = STFTParams(
            window_length=self.win_length,
            hop_length=self.hop_length,
            match_stride=True,
        )

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)

        self.all_in_mem = all_in_mem
    
    def get_audio(self, path_and_text):
        try:
            audiopath, text = path_and_text['path'], path_and_text['text']
            raw_text = text
            raw_text = text.replace(',','，')
            if raw_text[-1] not in '。！？“':
                raw_text = raw_text+'。'
            text_ids = self.tokenizer.encode(raw_text)
            text_ids = torch.Tensor(text_ids)
            wav, sr = torchaudio.load(audiopath)
            if wav.shape[0] > 1:
                wav = wav[0].unsqueeze(0)
            wav = F.resample(wav, sr, self.sampling_rate)
            audio_norm = wav
            mel = mel_spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.n_mel_channels,
                self.sampling_rate,
                self.hop_length,
                self.win_length,
                self.mel_fmin,
                self.mel_fmax,
            )
            mel = torch.squeeze(mel, 0)

            return audio_norm, text_ids, raw_text, audiopath, mel

        except Exception as e:
            print('error: ', e)
            return None, None, None, None, None
    
    def random_slice(self, audio_norm, text_ids, raw_text, audiopath, mel):
        l = min(mel.shape[1], audio_norm.shape[-1] // self.hop_length // 8 * 8)
        audio_norm = audio_norm[:, :l * self.hop_length]
        mel = mel[:, :l]
        raw_wav, raw_mel = audio_norm, mel
        if audio_norm.shape[-1] > self.segment_size * self.hop_length:
            start = random.randint(0, mel.shape[1] - self.segment_size)
            end = start + self.segment_size
            mel = mel[:, start:end]
            audio_norm = audio_norm[:, start * self.hop_length : end * self.hop_length]

        else:
            return None
        return audio_norm, raw_wav, mel, raw_mel, text_ids, raw_text, audiopath, 
    
    def __getitem__(self, index):
        try:
            if self.train_paths.endswith(".jsonl"):
                ret = self.random_slice(*self.get_audio(self.audiopaths_and_text[index]))
            else:
                actual_index = self.indices[index]
                data = {
                    'path': self.audiopaths[actual_index].decode('utf-8'),
                    'text': self.texts[actual_index].decode('utf-8')
                }
                ret = self.random_slice(*self.get_audio(data))
        except Exception as e:
            if self.train_paths.endswith(".jsonl"):
                print(self.audiopaths_and_text[index])
            else:
                print({'path':self.audiopaths[index].decode('utf-8'),'text':self.texts[index].decode('utf-8')})
            print('eror: ', e)
            return None
        return ret

    def __len__(self):
        if self.train_paths.endswith('.jsonl'):
            return len(self.audiopaths_and_text)
        else:
            return len(self.texts)
    


class TextAudioCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if len(batch) == 0:
            return None
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].shape[1] for x in batch]),
            dim=0, descending=True
        )
        # 1. Record max length of each field
        max_wav_len = max([x[0].size(1) for x in batch])
        max_raw_wav_len = max([x[1].size(1) for x in batch])
        max_mel_len = max([x[2].size(1) for x in batch])
        max_raw_mel_len = max([x[3].size(1) for x in batch])
        max_text_len = max([len(x[4]) for x in batch])
        # 2. Prepare padded tensors container
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        raw_wav_padded = torch.FloatTensor(len(batch), 1, max_raw_wav_len)
        text_padded = torch.LongTensor(len(batch), max_text_len)
        mel_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_mel_len)
        raw_mel_padded = torch.FloatTensor(len(batch), batch[0][2].shape[0], max_raw_mel_len)
        # 3. Prepare lengths
        text_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        raw_wav_lengths = torch.LongTensor(len(batch))
        mel_lengths = torch.LongTensor(len(batch))
        raw_mel_lengths = torch.LongTensor(len(batch))
        # 4. Initialize padded tensors
        wav_padded.zero_()
        text_padded.zero_()
        raw_wav_padded.zero_()
        mel_padded.zero_()
        raw_mel_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]
            wav = row[0]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
                       
            raw_wav = row[1]
            raw_wav_padded[i, :, :raw_wav.size(1)] = raw_wav
            raw_wav_lengths[i] = raw_wav.size(1)
            
            mel = row[2]
            mel_padded[i, :, :mel.size(1)] = mel
            mel_lengths[i] =mel.size(1)
            
            raw_mel = row[3]
            raw_mel_padded[i, :, :raw_mel.size(1)] = raw_mel
            raw_mel_lengths[i] = raw_mel.size(1)
            
            text = row[4]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)
        
        raw_text = [x[5] for x in batch]
        audiopath = [x[6] for x in batch]

        return {
            "wav":wav_padded,
            "wav_length":wav_lengths,
            "raw_wav":raw_wav_padded,
            "raw_wav_length":raw_wav_lengths,
            "mel":mel_padded,
            "mel_length":mel_lengths,
            "raw_mel":raw_mel_padded,
            "raw_mel_length":raw_mel_lengths,
            "text":text_padded,
            "text_length":text_lengths,
        }

if __name__ == "__main__":
    cfg_path = './models/DAC/config/dac25hz_16.yaml'
    with open(cfg_path, 'r') as f:
            config = yaml.safe_load(f)

    hps = HParams(**config)
    dataset = TextAudioSpeakerLoader(hps)
    import pdb
    pdb.set_trace()
    dataset.__getitem__(0)
    



        
            




