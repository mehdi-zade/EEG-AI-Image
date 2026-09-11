# Open Source Implementations & Codebases Hub

This directory indexes verified open-source code repositories, model checkpoints, and reproduction guides for state-of-the-art EEG-to-image decoding models.

---

## 1. Quick Reference Matrix

| Model | Venue | Primary Framework | Official GitHub Repository | Model Checkpoints | Primary Dataset |
|---|---|---|---|---|---|
| **GWIT** (Guess What I Think) | ICASSP 2025 | PyTorch / Diffusers | [LuigiSigillo/GWIT](https://github.com/LuigiSigillo/GWIT) | [Hugging Face](https://huggingface.co/datasets/luigi-s/EEG_Image_CVPR_ALL_subj) | EEGCVPR40 |
| **BrainVis** | ICASSP 2025 | PyTorch / Accelerate | [RomGai/BrainVis](https://github.com/RomGai/BrainVis) | GitHub Releases | THINGS-EEG / EEGCVPR40 |
| **ATM / Guided Diffusion** | NeurIPS 2024 | PyTorch / Diffusers | [dongyangli-del/EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode) | Hugging Face | THINGS-EEG2 |
| **DreamDiffusion** | arXiv 2023 | PyTorch / Timm | [bbaaii/DreamDiffusion](https://github.com/bbaaii/DreamDiffusion) | Baidu Netdisk / Hugging Face | MOABB + EEGCVPR40 |
| **EEGStyleGAN-ADA** | arXiv 2023/2024 | PyTorch / StyleGAN2 | [prajwalsingh/EEGStyleGAN-ADA](https://github.com/prajwalsingh/EEGStyleGAN-ADA) | Google Drive | EEGCVPR40 / ThoughtViz |
| **ViEEG** | ICML 2024 | PyTorch / OpenCLIP | [OpenReview / GitHub](https://openreview.net/forum?id=ViEEG) | Available on request | THINGS-EEG |
| **BReAD** | SIGIR 2025 | PyTorch / FAISS | [zhouyujia.cn / GitHub](https://github.com/zhouyujia/BReAD) | Hugging Face | THINGS-EEG / MEG |
| **MinD-Vis** (fMRI Reference) | ICML 2023 | PyTorch / LDM | [Mind-Vis/Mind-Vis](https://github.com/Mind-Vis/Mind-Vis) | Hugging Face | GOD / BOLD5000 |
| **MindEye2** (fMRI Reference) | ICML 2024 | PyTorch / SDXL | [MedARC-AI/MindEye2](https://github.com/MedARC-AI/MindEye2) | Hugging Face Hub | NSD (7T fMRI) |

---

## 2. Deep Dive: Key EEG Codebases & Execution Setup

### 2.1 GWIT (Guess What I Think) — Sigillo & Lopez et al.
*   **Repository**: [https://github.com/LuigiSigillo/GWIT](https://github.com/LuigiSigillo/GWIT)
*   **Core Architecture**: Latent Diffusion Model conditioned via a 1D-CNN ControlNet adapter.
*   **Installation & Setup**:
    ```bash
    git clone https://github.com/LuigiSigillo/GWIT.git
    cd GWIT
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install diffusers transformers accelerate einops
    ```
*   **Dataset Acquisition**:
    ```python
    from datasets import load_dataset
    dataset = load_dataset("luigi-s/EEG_Image_CVPR_ALL_subj")
    ```

---

### 2.2 ATM / Guided Diffusion (Dongyang Li et al., NeurIPS 2024)
*   **Repository**: [https://github.com/dongyangli-del/EEG_Image_decode](https://github.com/dongyangli-del/EEG_Image_decode)
*   **Core Concept**: Adaptive Thinking Mapper (ATM) extracts low-frequency and high-frequency EEG features and generates a dual conditioning signal:
    1. A low-level "blurry image" prior guiding early diffusion timesteps.
    2. A high-level CLIP embedding guiding cross-attention layers.
*   **Dataset Format**: Preprocessed THINGS-EEG2 numpy arrays (`[num_trials, 64_channels, 250_timesteps]`).

---

### 2.3 BrainVis (Fu et al., ICASSP 2025)
*   **Repository**: [https://github.com/RomGai/BrainVis](https://github.com/RomGai/BrainVis)
*   **Core Innovation**: Cascaded diffusion models combined with time-frequency EEG representations (wavelet transform) and semantic interpolation in CLIP space.
*   **Data Efficiency**: Demonstrates superior reconstruction quality using only ~10% of the training dataset.

---

### 2.4 DreamDiffusion (Bai et al., 2023)
*   **Repository**: [https://github.com/bbaaii/DreamDiffusion](https://github.com/bbaaii/DreamDiffusion)
*   **Workflow**:
    1. **Pretraining**: ViT-Large encoder trained via Masked Signal Modeling (75% token mask) on MOABB dataset.
    2. **Fine-tuning**: Frozen Stable Diffusion v1.5 with trainable cross-attention heads conditioned on EEG embeddings.
    3. **Alignment**: Contrastive CLIP loss mapping EEG tokens to image/text semantic representations.
