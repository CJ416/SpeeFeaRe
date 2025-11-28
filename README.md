# SpeeFeaRe: Speech Feature Representation

<img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">

<img alt="PRs Welcome" src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg">

English | 中文

---

<a name="english"></a>

## 🇬🇧 English Version

**SpeeFeaRe** (Speech Feature Representation) is a repository dedicated to tracking, analyzing, and reproducing cutting-edge methods in speech representation learning.

This repository aims to map the evolutionary trajectory from Neural Audio Codecs to next-generation joint semantic-acoustic representations, with a special focus on **LLM-based TTS**, **Omni-models**, and **Autoregressive (AR) generation** scenarios.

### 📖 Motivation

With the advent of Omni-models like GPT-4o, speech is no longer just an object of signal processing but has become a "First-class Citizen" for Large Language Models (LLMs). To enable LLMs to understand and generate speech as fluently as text, we need efficient **Speech Representations**.

Current speech representation research is undergoing a paradigm shift from "high-fidelity reconstruction" to "semantic disentanglement and fusion." This repository documents this exciting technological evolution.

### 🗺️ Evolution Roadmap

We categorize the development of speech representations into the following key stages:

#### 1. The Rise of Discrete Representations: Reconstruction & Compression

Early Neural Audio Codecs aimed to discretize continuous audio into tokens for autoregressive modeling by LLMs.

- **Key Tech**: Residual Vector Quantization (RVQ), GAN Discriminators.
- **Representative Works**:
  - **SoundStream**: The pioneer of reconstruction-based RVQ, laying the foundation for modern codecs.
  - **EnCodec**: Introduced Transformers and streaming processing, further improving compression rates and quality.
  - **DAC**: Fully convolution-based encoder-quantizer-decoder architecture, GAN training paradigm with improved tricks, including L2 normalize, factorized codes and diverse discriminators.

#### 2. Semantic Enhancement: Distillation & Disentanglement

Purely acoustic reconstruction tokens lack semantic information, leading to content inaccuracies in LLM speech generation. Researchers began injecting semantic information from Self-Supervised Learning (SSL) models (e.g., HuBERT/W2V-BERT) into codecs.

- **Key Tech**: Semantic Distillation, Mutual Information Disentanglement.
- **Representative Works**:
  - **SpeechTokenizer**: Attempts to disentangle semantic and acoustic tokens via HuBERT distillation.
  - **MiMicodec**: Further optimizes semantic-acoustic alignment.
  - **Xcodec**: Large-scale semantic distillation at the architectural level to enhance token language understanding.
  - **WavTokenizer**: Extreme compression rates with SOTA reconstruction quality.

#### 3. The Paradigm Shift: Discrete vs. Continuous

While discretization (Quantization) fits the Cross-Entropy Loss of LLMs, it inevitably leads to **information loss** (especially in prosody and emotion). Consequently, continuous features are regaining attention.

> **💡 Deep Dive: Challenges of Continuous Feature Modeling in AR** Continuous features (like Mel-spectrograms or Latent vectors) are typically used in Non-AR or Diffusion (DiT) architectures. Modeling them directly in Autoregressive (AR) architectures faces significant challenges:
> 
> 1. **Error Accumulation**: In AR generation, $x_t$ depends on $x_{t-1}$. Discrete tokens can "reset" errors via codebook lookup, whereas minute prediction errors in continuous features amplify exponentially over sequence length, leading to generation collapse.
> 2. **Variance Collapse / Over-smoothing**: Traditional regression losses (like MSE) tend to predict the mean of the distribution. Due to the high variance of speech data, models tend to output "averaged," smooth trajectories, resulting in muffled speech lacking detail and high-frequency information.
> 3. **Difficulty in Density Estimation**: The discrete space is finite (Softmax), while the continuous space is infinite. Accurately modeling complex multi-modal distributions in an AR framework is notoriously difficult.

#### 4. Next-Gen Frontiers: Continuous Autoregressive Modeling

To combine the contextual capabilities of AR with the high fidelity of continuous features, recent works are exploring new paths, combining Diffusion Loss or Flow Matching to address the challenges above.

- **Representative Works**:
  - **VoxCPM**: Exploring continuous features in language modeling.
  - **DiTAR**: Diffusion for Autoregressive generation.
  - **CosyVoice**: Combining the strengths of discrete and continuous representations for high-quality zero-shot generation.

### 📚 Paper List & Analysis

*(Note: Links point to detailed analysis documents within the repo)*

| Category            | Paper               | Key Idea                                 | Code/Analysis                   |
| ------------------- | ------------------- | ---------------------------------------- | ------------------------------- |
| **Pioneer**         | **SoundStream**     | End-to-end neural audio codec with RVQ   | [Analysis](./assets/pioneer.md) |
| **Pioneer**         | **EnCodec**         | High fidelity neural audio compression   | [Analysis](./assets/semantic.md) |
| **Semantic Fusion** | **SpeechTokenizer** | Unified speech tokenizer for speech LLMs | Analysis                        |
| **Semantic Fusion** | **Mimi**            | Split RVQ with Semantic Distillation     | Analysis                        |
| **Semantic Fusion** | **Xcodec**          | Large-scale semantic distillation        | Analysis                        |
