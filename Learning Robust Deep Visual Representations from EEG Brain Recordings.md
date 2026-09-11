# Learning Robust Deep Visual Representations from EEG Brain Recordings

> [!TIP]
> A structured summary and comparison of this paper is also indexed at [papers/EEGStyleGAN_ADA.md](papers/EEGStyleGAN_ADA.md).

Original Code: https://github.com/prajwalsingh/EEGStyleGAN-ADA

### Summary
proposes a two-stage method for learning robust deep visual representations from EEG brain recordings. The first stage involves obtaining EEG-derived features, and the second stage uses these features for image generation and classification. The authors claim that their method is generalizable across three different datasets and achieves state-of-the-art performance in image synthesis from EEG signals.


![image](https://github.com/user-attachments/assets/50cbe076-05a6-41ea-ba02-b9e01cb68cf8)

This figure shows sample images generated from EEG signals using a technique called EEGStyleGAN-ADA. Each image is generated with different EEG signals across different classes from the EEGCVPR40 dataset.

### Feature Extraction

**What is feature extraction?**
Feature extraction is a process of selecting and transforming raw data into a more meaningful representation that can be used for machine learning tasks.

**Why is feature extraction important?**
Feature extraction is crucial because it helps in reducing the dimensionality of the data, removing noise, while retaining important information.

**What are the methods used for feature extraction?**
There are two types of methods used for feature extraction:

1. **Supervised methods**: These methods use labeled data to train a model to extract features. The model learns to extract features that are relevant for a specific task, such as classification.
2. **Self-supervised/metric-based learning methods**: These methods do not use labeled data. Instead, they use techniques such as regression, KL-divergence, and metric learning to extract features.

**What is the issue with supervised methods?**
The issue with supervised methods is that they require the test data distribution to overlap with the train data distribution, which is not always the case with EEG datasets.

**What is the solution?**
The solution is to use self-supervised/metric-based learning methods, which can overcome the issue of non-overlapping data distributions.

**What is the triplet loss function?**
The triplet loss function is a type of metric learning-based method used for feature learning.
Training networks using triplet loss helps them learn discriminative features, which leads to better k-means accuracy.
It is defined as:

$min_θ E [||f_θ(x^a) - f_θ(x^p)||^2 _2 - ||f_θ(x^a) - f_θ(x^n)||^2 _2 + δ]$

Where $f_θ$ is the encoder, $x^a$ is an anchor, $x^p$ is a positive sample, $x^n$ is a negative sample, and $δ$ is the margin distance between the positive and negative samples.

**What is the purpose of semi-hard triplets?**
Semi-hard triplets prevent the encoder network from generating similar representations for all the data and enforce the learning of discriminative features.
The semi-hard triplets have the following property: 
$\left\| f_{\theta}(x_a) - f_{\theta}(x_p) \right\| < \left\| f_{\theta}(x_a) - f_{\theta}(x_n) \right\| < \left( \left\| f_{\theta}(x_a) - f_{\theta}(x_p) \right\| + \delta \right)$

![image](https://github.com/user-attachments/assets/b0f0b4b7-4ede-439b-b5cf-13e520fef6ca)

Figure 5 shows the results of clustering EEG signals using two different architectures: LSTM (Long Short-Term Memory) and CNN (Convolutional Neural Network). The clustering is done using a technique called triplet loss, which is a type of metric learning approach.

The performance of the CNN architecture degrades as the timestep size of the EEG signal decreases. This suggests that the CNN architecture is not as effective at handling shorter EEG signals.

Previous methods have tailored their architectures to specific datasets, whereas the approach described in this article uses a more generalizable architecture that can be applied across different datasets.

### Showing the generalizability
1. **Unseen Data**
  The network is trained on 34 classes from EEGCVPR40 [39] dataset and tested on the remaining 6
  classes, which are a dog, cat, fish, canoe, golf, and pool.
  Compared to [ 26], a pre-trained image network is not required. 
  As shown in Table 1, our method performs better and has a higher SVM [12 ] and kNN [25] score.
  ![image](https://github.com/user-attachments/assets/fb3b4b57-dafc-47a2-83b1-8dea4c049b22)
  
  We have also shown t-SNE [42] plot for all the 6 unseen classes learned features Fig.6.
  ![image](https://github.com/user-attachments/assets/595133d5-2513-4906-81de-1cc77dabf126)

2. **Image to Image Translation**
   Instead of using an EEG signal from EEGCVPR40 [ 39 ] dataset, its equivalent images are used, which are transformed into EEG representation space, and later, the image is reconstructed with approximation using a pre-trained generative network.
  The EEG features generated from unseen images can reconstruct the images with high fidelity.
  ![image](https://github.com/user-attachments/assets/102c7473-14e4-44de-b706-324d04b06da0)


### Image Generation with EEG based conditioning
![image](https://github.com/user-attachments/assets/f9720a41-11c0-4b8f-afa8-45cbb40d2f32)

EEGStyleGAN-ADA network which is a tailored version of StyleGAN-ADA. StyleGAN-ADA uses adaptive discriminator augmentation, which helps the discriminator learn with limited data by augmenting real images at training time.

Inputs:
  - feature vector obtained from a pre-trained LSTM network with triplet loss.
  - noise vector sampled from iso-tropic Gaussian distribution

Training:
  - employ the ’cifar’ hyperparameters, leveraging data available from the EEGCVPR40 [39 ] and ThoughtViz [22 , 41 ] datasets.

### Image Generation with class based conditioning
Class-based conditioning involves using one-hot encoded class labels (such as categories or predefined labels) rather than actual EEG signals as input. This method is commonly used in conditional generative models but faces challenges when applied to datasets with imbalanced classes (like the EEGCVPR40 dataset, which contains a limited number of images per class). The ablation study involving class-based conditioning serves to show how the model performs when not using the EEG data itself, but just the class information. It also demonstrates the limitations of current state-of-the-art methods, like NoisyTwins, when generating images with class labels on complex, imbalanced datasets.

**Why separate these two types of conditioning?**
  Show the advantage of directly using EEG data for more personalized and specific image generation.
  Compare this with class-based conditioning to assess whether using class labels alone can achieve the same results, and to highlight the challenges and shortcomings of that approach (especially in the context of EEG data).
  The result suggests that the photorealistic images we are synthesizing with the proposed EEGStyleGAN-ADA are the best among all the GAN state-of-the-art methods for EEG-based image generation.
![image](https://github.com/user-attachments/assets/b18fce1e-386d-417b-b783-f169affe42ca)


### EEG CLIP: Joint Space Learning for EEG and Image

**Previous Methods**
1. Use a pre-trained image encoder to generate a representation for images equivalent to an EEG signal and train an EEG encoder network to regress the image feature vector.
2. train the EEG encoder in a contrastive setting with a triplet loss instead of regressing the image feature vector. utilize the CLIP based method.

**This Article's Method**
Use a pre-trained ResNet50 as an image encoder and a multilayer LSTM network as an EEG feature encoder. 
During training, freeze the weight of ResNet50 and only update the weights of the LSTM network.
Use CLIP-based loss for training the complete pipeline. 
![image](https://github.com/user-attachments/assets/f535031b-5f09-48c9-b379-92b3d7b5468f)
As shown in Fig.4, each EEG-image pair is treated as a positive sample (diagonal elements) and the rest as a negative sample (non-diagonal elements).  
Uses a pre-trained image encoder for the EEG-based image retrieval tasks.

**Investigating the impact of batch size and the number of training epochs on learning joint representations**
![image](https://github.com/user-attachments/assets/7dbccb62-63b1-4f73-8e5f-7939425fedb4)

The proposed EEGClip framework achieved superior performance when trained with a **batch size of 64** and for **2048 epochs**.
This configuration yielded the highest recall rates for different values of K. 
To provide a visual representation of the effectiveness of EEGClip, we present image retrieval results from EEG in Fig.8. 
![image](https://github.com/user-attachments/assets/c577b6e2-352f-416b-8ed7-abfb0652046d)

These results demonstrate the ability of our framework to retrieve relevant images based on EEG
input accurately.


### Evaluation Merics:
- Inception score: a commonly used metric in generative models.
- Frechet Inception Distance (FID)
- Kernel Inception Distance (KID)
- Mean Reciprocal Rank (MRR) assesses the effectiveness of the retrieval model in accurately ranking the unique visually-cued instances
- Mean Average Precision (mAP) evaluates the retrieval model’s ability to capture all relevant visual cues. These relevant visual cues correspond to images that belong to the same semantic class as the correct match.


### Datasets
| Name | Number of classes | Images per class | EEG channels | No. of participants | EEG-Image pairs | Parent dataset
|------------|------------|-------------|---------------|-------------|---------------|--------|
| [EEGCVPR40 ](https://arxiv.org/pdf/2211.06956)   | 40  | 50 | 128 | N/A | 11800 | ImageNet |
| [ThoughtViz](https://web.njit.edu)   | 10    | 50 | 14 | 23 | N/A | ImageNet |
| [Object](https://www.crcv.ucf.edu)   | 6    | 12 | 128 | 10 | N/A | Non |

### Discussion
In this paper, we:
1. addressed the problem of EEG-to-image reconstruction
2. presented a comprehensive method to extract visual information from EEG data,
3. synthesize images using extracted EEG features
4. jointly train an EEG-Image model for tasks such as EEG-based image retrieval.
5. We conducted experiments and ablation studies on three different datasets: EEGCVPR40 [ 39 ], ThoughtViz [22 , 41 ], and Object [17], using various architectures and loss functions.
6. We first discussed different strategies for feature extraction from EEG data, including supervised and self-supervised methods.
7. We compared supervised classification methods with self-supervised/metric-based learning approaches and found that the latter yielded more discriminative and generalizable features, particularly using triplet loss.
8. We demonstrated the same through improved k-means accuracy, t-SNE visualizations, and zero-shot classification.
9. We explored the generation of images from EEG features using the GAN framework. For this, we proposed EEGStyleGAN-ADA, which incorporated EEG features and noise vectors to synthesize diverse and high-fidelity images.
10. Our method outperformed previous EEG-to-image synthesis networks, with 62.9% and 36.13% inception score improvement on the EEGCVPR40 [ 39 ] dataset and Thoughtviz [ 22, 41 ] dataset, which is better than state-of-the-art performance using GAN.
11. We have shown that achieving the photorealistic effect is not trivial with the help of class-based conditioning 4.3
12. We investigated joint representation space learning for EEG and image using the proposed EEGClip framework. We achieved significant improvements in joint representation learning by freezing the weights of a pretrained image encoder and training an EEG feature encoder using CLIP-based loss.
13. Upon examining the effects of batch size and epoch numbers, we observed a direct correlation between increased batch size and enhanced performance, peaking at a batch size of 64 and 2048 epochs, yielding scores of 79% and 95% for top@1 for EEG and image, respectively. However, extending the epoch count beyond this point showed no significant improvement.
14. EEGClip has shown 5.96% improvement over the previous state-of-the-art joint representation learning method.

### Limitations
1. Although we have used the same architecture for EEG feature extraction across all datasets, it is still an open problem to achieve SOTA performance using a single architecture.
2. In EEG-based image synthesis, we outperform the previous methods. Still, the quality of images in a limited dataset regime can be improved with better GAN training strategies, and we can further utilize the same for EEG-based image reconstruction
