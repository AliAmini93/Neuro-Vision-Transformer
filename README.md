# Neuro Vision Transformer (Neuro ViT)

## Overview

Neuro ViT is an innovative approach to EEG signal classification, leveraging the power of the Vision Transformer (ViT) model to distinguish between individuals with ADHD and healthy controls. This project is at the forefront of applying transformer models, typically used in image processing, to the realm of EEG data analysis.

### Key Features
- **Custom EEG Data Processing**: EEG signals, with an input size of (19,512) representing 19 electrodes and 512 time-steps, are preprocessed and divided into patches, analogous to image patch processing in ViT models.
- **Tailored ViT Model for EEG**: The model is adapted to handle the unique structure of EEG data. Key adaptations include:
  - **Patch Extraction Layer**: Custom layer for transforming EEG data into a format suitable for ViT processing.
  - **Multi-Head Self-Attention Mechanism**: Utilizes 8 heads in each attention layer to capture various aspects of EEG signals.
  - **Transformer Architecture**: Comprises 6 transformer layers, each with multi-head attention, layer normalization, and MLP layers, fine-tuned for EEG data.
- **EEG Data Classification**: Focused on classifying between ADHD and healthy control subjects, demonstrating the potential of ViT in medical diagnostics.

### Dataset
The EEG dataset used in this project consists of signals recorded at 128Hz, covering 4-second intervals across 19 electrodes. This rich dataset provides a comprehensive view of brain activity, enabling the deep learning model to discern patterns indicative of ADHD.

This project not only demonstrates the adaptability of ViT models to new domains but also opens up new avenues in the application of deep learning for medical diagnostics, particularly in neuroimaging and mental health.

