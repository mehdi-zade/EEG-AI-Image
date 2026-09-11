# Comprehensive Guide to Datasets for EEG Visual Decoding & Reconstruction

## 1. Overview Matrix

| Dataset Name | Modality | Channels | Subjects | Classes / Concepts | Total Images | Paradigm | Public Availability | Primary Use-Cases |
|---|---|---|---|---|---|---|---|---|
| **THINGS-EEG** (Grootswagers 2022) | EEG | 64 | 50 | 1,854 concepts | 22,248 | Rapid Serial Visual Presentation (RSVP, 100 ms) | [OpenNeuro (ds003825)](https://openneuro.org/datasets/ds003825) / [Figshare](https://figshare.com) | Modern gold standard for Zero-Shot & LDM reconstruction |
| **THINGS-EEG2** (Gifford 2022) | EEG | 64 | 10 | 1,854 concepts | 22,248 | RSVP (100 ms) with high repetition | [OpenNeuro (ds004148)](https://openneuro.org/datasets/ds004148) / [OSF](https://osf.io/anvqy/) | SOTA benchmark for ATM, ViEEG, Saliency-guided models |
| **EEGCVPR40 / EEG-ImageNet** (Spampinato 2017) | EEG | 128 | 6 | 40 ImageNet classes | 2,000 (50/class) | Block design (500 ms display, 1000 ms ISI) | [Hugging Face](https://huggingface.co/datasets/luigi-s/EEG_Image_CVPR_ALL_subj) / [GitHub](https://github.com/perceivelab/eeg_visual_classification) | Historical baseline; used by GWIT, EEGStyleGAN, BrainDecoder |
| **ThoughtViz** (Tirupattur 2018) | EEG | 14 (Emotiv Epoc) | 23 | 10 object classes + digits + characters | Variable (32,000 epochs) | Visual stimulus & imagination | [Google Drive](https://drive.google.com/file/d/1atP9CsjWIT-hg3fX--fcC1hg0uvg9bEH/view) | Low-density consumer-grade headset evaluation |
| **Kaneshiro (Stanford 2015)** | EEG | 128 | 10 | 72 natural object exemplars (12 categories) | 72 | Rapid RSVP (500 ms) | [Stanford SDR](https://purl.stanford.edu/bq914sc3730) | Temporal dynamics of categorization |
| **MOABB Benchmark** (NeuroTechX) | EEG | 30–128 | 400+ | Heterogeneous (motor imagery, P300, visual) | >120,000 epochs | Cross-study pooled | [moabb.neurotechx.com](https://neurotechx.github.io/moabb/) | Self-supervised masked signal pretraining (DreamDiffusion) |
| **Natural Scenes Dataset (NSD)** | 7T fMRI | N/A | 8 | COCO natural scenes | 73,000 trials (10,000 images) | Long presentation (3s) | [naturalscenesdataset.org](https://naturalscenesdataset.org) | fMRI cross-modal baseline (MinD-Vis, MindEye 1 & 2) |

---

## 2. THINGS-EEG & THINGS-EEG2 (The Modern Gold Standard)

### 2.1 Experimental Architecture
Developed as part of the international **THINGS Initiative** (Hebart et al.):
- **Stimuli**: 1,854 concrete object concepts based on American English nouns (animals, tools, foods, vehicles, instruments).
- **Images**: 22,248 high-quality photographic images with naturalistic backgrounds.
- **Presentation Design**: Rapid Serial Visual Presentation (RSVP) at 10 Hz (100 ms display duration), preventing eye saccades and cognitive task rehearsal artifacts.
- **Randomization**: Complete cross-class randomization across trials. Completely immune to block-design leakage!

### 2.2 Preprocessing Standard for Modern Diffusion Models
Most recent models (ATM, ViEEG, BReAD) process THINGS-EEG as follows:
```python
# Standard THINGS-EEG preprocessing pipeline
import mne

def preprocess_things_eeg(raw):
    # 1. Bandpass filter to isolate visual ERP frequencies
    raw.filter(l_freq=0.1, h_freq=100.0, fir_design='firwin')
    # 2. Notch filter for line noise (50/60 Hz)
    raw.notch_filter(freqs=[50, 60])
    # 3. Epoching: -200 ms baseline to 800 ms post-stimulus onset
    epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.8, baseline=(-0.2, 0))
    # 4. Resample to 250 Hz (or 200 Hz) -> [Channels: 64, Timesteps: 250]
    epochs.resample(250)
    return epochs
```

---

## 3. EEGCVPR40 (Historical ImageNet Benchmark)

### 3.1 Dataset Structure
- **Equipment**: 128-channel BrainVision actiCAP.
- **Subjects**: 6 healthy human participants.
- **Stimuli**: 40 distinct ImageNet synsets (dog, cat, elephant, butterfly, airliner, sports car, guitar, etc.), 50 images per synset. Total: 2,000 unique images, ~12,000 total EEG segments across subjects.
- **Signal Specs**: Sampled at 1,000 Hz, epoched into 500 ms windows (500 time points).

### 3.2 Where to Download
- Pre-packaged `.pth` / `.npy` files hosted on Hugging Face by Luigi Sigillo (author of GWIT):
  - [Hugging Face: luigi-s/EEG_Image_CVPR_ALL_subj](https://huggingface.co/datasets/luigi-s/EEG_Image_CVPR_ALL_subj)
- Original Perceive Lab repository:
  - [GitHub: perceivelab/eeg_visual_classification](https://github.com/perceivelab/eeg_visual_classification)

---

## 4. ThoughtViz (Consumer-Grade Benchmark)

### 4.1 Dataset Structure
- **Device**: 14-channel Emotiv EPOC headset (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4).
- **Tasks**:
  - 10 ImageNet classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).
  - 10 handwritten digits (MNIST-style).
  - 26 uppercase English characters.
- **Participants**: 23 subjects.
- **Value**: Demonstrates feasibility and degradation when moving from high-density clinical caps (64–128 channels) to low-density wearable systems.

---

## 5. Selection Criteria for New Research

| Question | Recommended Dataset |
|---|---|
| Need to publish at top-tier venues (NeurIPS, ICML, CVPR, ICLR)? | **THINGS-EEG / THINGS-EEG2** |
| Reproducing or benchmarking against GWIT, BrainDecoder, or EEGStyleGAN? | **EEGCVPR40** (with cross-subject/leave-session splits) |
| Exploring wearable, consumer, or VR neurotechnology? | **ThoughtViz** or downsampled THINGS-EEG |
| Pretraining large-scale foundational EEG encoders via self-supervision? | **MOABB + THINGS-EEG** |
