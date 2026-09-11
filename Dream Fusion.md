# DreamDiffusion: Generating Images from Brain EEG Signals

> [!NOTE]
> **Name Clarification**: This document analyzes **DreamDiffusion** (Bai et al., Tencent ARC Lab & Tsinghua University, 2023) for EEG-to-image reconstruction, which builds on principles from MinD-Vis. (It is distinct from Google's text-to-3D *DreamFusion*).
> An expanded and fully updated analysis is available at [papers/DreamDiffusion.md](papers/DreamDiffusion.md).

![image](https://github.com/user-attachments/assets/831e6f53-ac12-4a81-8a58-3f55d2200fee)

- **Masked signal pre-training**: Utilizes masked signal modeling to train a robust EEG encoder.
- **Fine-tuning with limited EEG-image pairs**: Uses pre-trained Stable Diffusion (SD) with EEG conditional features.
- **Aligning EEG, text, and image spaces**: Uses CLIP encoders to align these spaces for better image generation.

### Masked Signal Pre-training for EEG Representation
  - Approximately 120,000 EEG data samples from over 400 subjects with channel ranges from 30 to 128 on the [MOABB](https://neurotechx.github.io/moabb/) platform.
  - EEG tasks include looking at objects, motor imagery, and watching videos.
  - EEG data is uniformly padded to 128 channels by replicating missing channel values.
  - EEG signals are filtered in the frequency range of 5-95 Hz and truncated to a length of 512.
  - Pre-training model for EEG is based on ViT-Large architecture.
  - The method leverages the **temporal characteristics** of EEG signals by dividing them into tokens in the time domain.
  - Every four adjacent time steps are grouped into a token.
  - A percentage of these tokens are **randomly masked**, and a one-dimensional convolutional layer is used to create embeddings.
  - The EEG mask ratio set to 75%.
  - Each token is transformed into a 1024-dimensional embedding via a projection layer.
  - An **asymmetric architecture** (similar to MAE) predicts missing tokens, training the EEG encoder to understand contextual EEG information.
  - Through reconstructing the masked signals, the pre-trained EEG encoder learns a deep understanding of EEG data across different people and various brain activities.
  - Masked signal modeling is used with MSE (Mean Squared Error) as the loss function, computed only on masked patches. 
  - Reconstruction is performed on all 128 channels as a whole, rather than on a per-channel basis.
  - The decoder is discarded after pre-training.
  - The encoder is pre-trained for 500 epochs and fine-tuned with Stable Diffusion for another 300 epochs.
  - ![image](https://github.com/user-attachments/assets/90c686b7-fa68-4941-9e66-5020b88ad757)

### Fine-tuning with Stable Diffusion:
   - The pre-trained EEG encoder provides conditional features for **Stable Diffusion** via a **cross-attention mechanism**.
   - **Stable Diffusion (SD)** works in the latent space: pixel space image $x$ is encoded to a latent representation $z = E(x)$.
   - EEG embeddings are introduced into **U-Net** with a cross-attention mechanism.
   - The cross-attention involves projection matrices $W_Q$, $W_K$, and $W_V$, which incorporate the EEG embeddings into U-Net layers.
   - During fine-tuning, only the **EEG encoder and cross-attention heads** are optimized while keeping the rest of the SD model fixed.
   - The fine-tuning loss function for Stable Diffusion is:
     $L_{SD} = E_{x, \epsilon \sim N(0, 1), t} \left[\||\epsilon - \epsilon_{\theta}(x_t, t, \tau_{\theta}(y))\||_2^2\right]$

### Aligning EEG, Text, and Image Spaces using CLIP:
The pre-trained Stable Diffusion model is specifically trained for text-to-image generation; however, the EEG signal has its own characteristics, and its latent space is quite different from that of text and image. Therefore, directly fine-tuning the Stable Diffusion model end-to-end using limited EEG-image paired data is unlikely to accurately align the EEG features with existing text embedding in pre-trained SD.
In conclusion,the aim is to fine-tune the EEG representation obtained from pre-training to make it more suitable for generating images  
- Its is proposed to employ additional CLIP supervision to assist in the alignment of EEG, text, and image space. Specifically:
- The EEG features obtained from the pre-trained encoder are transformed into embeddings with the same dimension as those of CLIP through a projection layer.
- The loss function is used to minimize the distance between the EEG embeddings and the image embeddings obtained from the CLIP image encoder.
- The CLIP model is fixed during the fine-tuning process.
- The loss function is defined as follows:
   $L_{clip} = 1 - \frac{E_I(I) \cdot h(\tau_{\theta}(y))}{|E_I(I)||h(\tau_{\theta}(y))|}$
   where $h$ is a projection layer and $E_I$ is the CLIP image encoder.
- This loss encourages EEG features to align with image features in the unified CLIP space, improving the compatibility of EEG with Stable Diffusion for image generation.

![image](https://github.com/user-attachments/assets/1d51b121-f769-49fb-9d7c-500cdd3fa4d9)

### Ablation Studies
![image](https://github.com/user-attachments/assets/919120f8-82bc-461d-94a5-cb62c54736a8)
1. **Evaluation Task**:
   - A 50-way top-1 accuracy classification task is used to evaluate the model.
   - A pre-trained ImageNet1K classifier is used to determine the semantic correctness of generated images.
   - Both ground-truth and generated images are inputted into the classifier.
   - A generated image is deemed correct if its top-1 classification matches the ground-truth classification in the 50 selected classes.

2. **Role of Pre-training**:
   - Two models are compared: one with a **full pre-trained model**, and another with a **shallow EEG encoding layer** (only two layers to avoid overfitting).
   - Models are trained with and without **CLIP supervision**.
   - Results show that **accuracy decreases without pre-training**.

3. **Mask Ratios**:
   - Results from **Model 5-7 in Table 1** show that excessively high or low mask ratios reduce performance.
   - The best overall accuracy is achieved with a **mask ratio of 0.75**.
   - Unlike natural language processing, where low mask ratios are common, a **high mask ratio** performs better for EEG data in this study.

4. **CLIP Aligning**:
- The alignment of EEG representations with images is achieved through the **CLIP encoder**.
- **Experiments 13-14** (Table 1) show that **performance decreases** significantly when **CLIP supervision** is not used.
- Even without pre-training, using CLIP to align EEG features still produces reasonable results, emphasizing the importance of **CLIP supervision** in the method.
- ![image](https://github.com/user-attachments/assets/2d030973-8dcd-4fbd-91f6-2fe0ee20df8d)


### Limitations 
Currently, EEG data only provide coarse-grained information at the category level in experimental results. 
This may result in some categories being mapped to other categories with similar shapes or colors. 
The authors assume this may be due to the fact that the human brain considers shape and color as two important factors when recognizing objects.
![image](https://github.com/user-attachments/assets/4b45d56c-a2c3-41d4-a608-01a5a6e5e084)
