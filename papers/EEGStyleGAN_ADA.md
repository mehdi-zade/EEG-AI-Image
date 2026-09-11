# Learning Robust Deep Visual Representations from EEG Brain Recordings (EEGStyleGAN-ADA & EEGClip)

- **Authors**: Prajwal Singh, et al.
- **Preprint**: [arXiv:2310.16532](https://arxiv.org/abs/2310.16532)
- **Code**: [https://github.com/prajwalsingh/EEGStyleGAN-ADA](https://github.com/prajwalsingh/EEGStyleGAN-ADA)
- **Key Models**: EEGStyleGAN-ADA, EEGClip, Triplet Loss LSTM Encoder
- **Datasets**: EEGCVPR40, ThoughtViz, Object

---

## 1. Overview & Contributions

This paper represents the pinnacle of the **Generative Adversarial Network (GAN)** lineage in EEG image synthesis before the complete pivot of the field toward Latent Diffusion Models. 

The authors address two fundamental shortcomings in prior GAN approaches (such as Brain2Image and ThoughtViz):
1. **Poor Feature Generalization**: Supervised encoders overfit to training distributions and fail on unseen classes.
2. **Discriminator Overfitting on Small Datasets**: Small EEG datasets cause the discriminator in GANs to memorize real images, resulting in training instability and mode collapse.

---

## 2. Technical Methodology

### 2.1 Metric-Learning EEG Feature Extraction (Triplet Loss)
Rather than simple supervised classification or regression, the EEG encoder is trained using **semi-hard triplet loss**:
$$\mathcal{L}_{\text{triplet}} = \max(0, \|f_\theta(x_a) - f_\theta(x_p)\|_2^2 - \|f_\theta(x_a) - f_\theta(x_n)\|_2^2 + \delta)$$
where $x_a$ is an anchor EEG trial, $x_p$ is a positive trial from the same visual concept, and $x_n$ is a negative trial from a different concept.

**Semi-hard triplets** satisfy:
$$\|f_\theta(x_a) - f_\theta(x_p)\|_2 < \|f_\theta(x_a) - f_\theta(x_n)\|_2 < \|f_\theta(x_a) - f_\theta(x_p)\|_2 + \delta$$
This forces the LSTM encoder to learn discriminative clustering in latent space without collapsing all representations.

### 2.2 Adaptive Discriminator Augmentation (EEGStyleGAN-ADA)
To prevent the discriminator from overfitting to the limited image counts in EEGCVPR40 (50 images per class):
- Incorporates Karras et al.'s **Adaptive Discriminator Augmentation (ADA)**, applying differentiable stochastic geometric transforms (translations, scaling, color jitter) dynamically based on discriminator divergence.
- The generator receives the EEG feature vector combined with isotropic Gaussian noise $\mathcal{N}(0, I)$ to generate diverse photorealistic samples.

### 2.3 Joint Space Learning via EEGClip
The authors introduce EEGClip:
- Freezes a pretrained ResNet-50 image encoder.
- Optimizes a multi-layer bidirectional LSTM using contrastive symmetric cross-entropy loss over batch diagonal pairs.
- Yields state-of-the-art EEG-to-image zero-shot retrieval accuracy.

---

## 3. Results & Evaluation

- **Inception Score Improvement**: Achieved 62.9% improvement on EEGCVPR40 and 36.13% improvement on ThoughtViz over previous GAN baselines.
- **Unseen Class Generalization**: Trained on 34 classes and evaluated zero-shot on 6 held-out classes (dog, cat, fish, canoe, golf, pool), significantly outperforming earlier methods in SVM and kNN linear probe accuracy.
