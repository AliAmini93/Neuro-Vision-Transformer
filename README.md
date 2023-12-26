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

### Dataset Overview
The EEG dataset used in this project is derived from the original dataset available at [EEG Data of ADHD and Control Children](https://ieee-dataport.org/open-access/eeg-data-adhd-control-children). This dataset includes EEG recordings from 61 children with ADHD and 60 healthy controls, aged 7-12. 

#### Participant Profile
- **ADHD Group**: Diagnosed according to DSM-IV criteria, under Ritalin treatment for up to 6 months.
- **Control Group**: Children with no history of psychiatric disorders, epilepsy, or high-risk behaviors.

#### Recording Methodology
- **Standard**: 10-20 system across 19 channels (Fz, Cz, Pz, C3, T3, C4, T4, Fp1, Fp2, F3, F4, F7, F8, P3, P4, T5, T6, O1, O2).
- **Frequency**: 128 Hz sampling.
- **Protocol**: Based on visual attention tasks with varying durations, dependent on each child’s response time.

### Dataset Adaptation for Neuro ViT
The dataset used in this project is a modified version of the original, specifically adapted for the Neuro ViT model. 

#### Preprocessing Details
- **Sample Structure**: Non-overlapping 4-second samples of raw continuous EEG signals.
- **Participants**: Includes both ADHD and control participants from the original dataset.
- **Objective**: Ensures a consistent and focused analysis of EEG data for the ViT model.

#### Significance
This preprocessing approach provides a comprehensive view of brain activity, enabling the deep learning model to discern patterns indicative of ADHD, thus demonstrating the adaptability of ViT models to new domains in medical diagnostics.



