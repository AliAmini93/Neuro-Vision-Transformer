# Neuro Vision Transformer (NeuroViT)

## Overview

NeuroViT is an innovative approach to EEG signal classification, leveraging the power of the Vision Transformer (ViT) model to distinguish between individuals with ADHD and healthy controls. This project is at the forefront of applying transformer models, typically used in image processing, to the realm of EEG data analysis.

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

### Dataset Adaptation for NeuroViT
The dataset used in this project is a modified version of the original, specifically adapted for the NeuroViT model. 

#### Preprocessing Details
- **Sample Structure**: Non-overlapping 4-second samples of raw continuous EEG signals.
- **Participants**: Includes both ADHD and control participants from the original dataset.
- **Objective**: Ensures a consistent and focused analysis of EEG data for the ViT model.

#### Significance
This preprocessing approach provides a comprehensive view of brain activity, enabling the deep learning model to discern patterns indicative of ADHD, thus demonstrating the adaptability of ViT models to new domains in medical diagnostics.

### Data Loading and Preprocessing

In the initial phase of our analysis, we focus on preparing the EEG dataset for the NeuroViT model. This involves several key steps:

- **Data Loading**: We load the EEG feature data (`X.npy`) and the corresponding labels (`Y.npy`), ensuring that we have a clear understanding of the dataset's structure.
- **Segment Processing**: Information about EEG signal segments per subject is loaded and processed. This step is crucial for understanding the distribution of data across different participants.
- **Grouping**: The data is grouped based on individual subjects to maintain consistency in analysis.
- **Data Splitting**: We employ stratified K-Fold cross-validation to split the dataset into training and testing sets. This ensures a representative distribution across different sets and is vital for the model's generalizability.
- **Data Reshaping and Label Processing**: The feature data is reshaped to fit the model's input requirements, and labels are processed into a categorical format, suitable for classification tasks.

This preprocessing pipeline is essential to ensure that the EEG data is in the appropriate format for effective analysis using the NeuroViT model.

### Model Configuration and Hyperparameters

Setting up the right configuration and hyperparameters is crucial for the optimal performance of the NeuroViT model. The following parameters were carefully selected to tailor the model specifically for EEG signal classification:

- **Learning Rate**: Set at 0.001, this parameter controls the rate at which our model updates its weights during training.
- **Weight Decay**: A factor of 0.0001 is used for regularization, helping to prevent overfitting.
- **Batch Size and Epochs**: We train the model in batches of 256 samples over 100 epochs to ensure comprehensive learning.
- **EEG Signal Specifications**: The EEG data is characterized by 128 time steps and 19 electrodes. This information is pivotal in structuring the input layer of our model.
- **ViT-Specific Parameters**:
  - **Number of Patches**: The EEG signals are divided into 4 patches, aligning with the ViT model’s mechanism.
  - **Projection Dimension**: Set to 10, determining the size of the dense layers in the transformer.
  - **Multi-Head Attention**: The model uses 8 heads in its attention layers to effectively capture different aspects of the EEG signals.
  - **Transformer Layers**: Comprising 6 layers, each layer is equipped with self-attention and feed-forward networks.
  - **MLP Head Units**: The final classifier consists of dense layers with sizes 50 and 10, fine-tuning the model's output.

### Custom Patch Extraction Layer

An essential step in adapting the Vision Transformer model for EEG data analysis is the creation of a custom layer for patch extraction. This layer is a pivotal part of the NeuroViT model, enabling the transformation of EEG signals into a format that the transformer can process effectively.

- **Custom Layer 'Patches'**: This layer is designed to handle EEG data which is fundamentally different from natural images typically used in ViT models.
- **Functionality**:
  - **Patch Extraction**: The layer extracts patches from the EEG signals based on the number of electrodes and time steps. This mimics the patch extraction process in standard ViT models but is tailored to EEG data characteristics.
  - **Batch Processing**: The layer is capable of processing multiple samples in a batch, maintaining efficiency in data handling.
  - **Reshaping Patches**: After extraction, the patches are reshaped to align with the model's input requirements.

This custom layer is integral to the model’s ability to understand and interpret EEG data.

### Multilayer Perceptron (MLP) in the EEG Transformer Model

A key component in the architecture of the NeuroViT model is the Multilayer Perceptron (MLP). This component plays a crucial role in classifying the EEG data.

- **Functionality of MLP**: The MLP is a sequence of layers in our model that works on the features extracted by the transformer. It helps in further refining and interpreting these features for classification purposes.
- **Structure**:
  - **Dense Layers with GELU Activation**: The MLP consists of dense layers, each followed by GELU (Gaussian Error Linear Unit) activation. This choice of activation function is crucial for capturing the non-linear relationships in the data.
  - **Dropout for Regularization**: To prevent overfitting, dropout layers are incorporated. The dropout rate is a hyperparameter that can be tuned based on the model's performance.

The design and implementation of the MLP are tailored to complement the transformer architecture, ensuring that the NeuroViT model is robust and effective for EEG signal classification tasks.

### Patch Encoding with Positional Information

A crucial element in adapting the Vision Transformer for EEG data is the PatchEncoder layer, which imbues positional information into the EEG patches.

- **Purpose of PatchEncoder**: This custom layer encodes each EEG patch with information about its position, a key aspect in the transformer model that captures the sequential nature of the data.
- **Components**:
  - **Projection**: Each patch is projected to a specified dimension, creating a dense representation that is suitable for the transformer.
  - **Positional Embedding**: In addition to projection, positional embeddings are added to the patches. This step is crucial in maintaining the order and context of the patches within the EEG sequence.
- **Operation**: The layer takes in the patches and applies both projection and positional embedding, effectively preparing them for processing by the transformer layers.

The implementation of the PatchEncoder is vital for the model to understand the spatial relationship between different parts of the EEG signals.

### Construction of the Vision Transformer (ViT) Model

The core of our project is the custom-built Vision Transformer (ViT) model, designed specifically for classifying EEG signals into ADHD and control categories.

- **Model Architecture**:
  - **Input Layer**: The model begins with an input layer designed to accommodate the shape of EEG data.
  - **Patch Creation and Encoding**: EEG signals are first converted into patches and then encoded with positional information.
  - **Transformer Layers**: The model includes several transformer layers, each consisting of layer normalization, multi-head attention mechanisms, and skip connections, followed by another layer normalization and an MLP. These layers are instrumental in processing the EEG data through self-attention mechanisms.
  - **Output Representation**: The output of the transformer layers is flattened and passed through dropout layers for regularization.
  - **Classification Layer**: Finally, an MLP is used for feature extraction, and a dense layer with softmax activation performs the classification.

### Training and Evaluation of the NeuroViT Model

The training and evaluation phase is crucial in determining the effectiveness of the NeuroViT model in classifying EEG data for ADHD and control groups.

- **Model Compilation**:
  - The model is compiled with an AdamW optimizer, incorporating both the specified learning rate and weight decay for better training dynamics.
  - Binary Cross entropy is used as the loss function.
  - Accuracy is tracked as the performance metric.

- **Checkpointing**:
  - Model checkpoints are created to save the best version of the model based on validation accuracy. This approach is essential to capture the most effective state of the model during training.

- **Model Training**:
  - The model is trained, using a batch size of 32 and iterating for 100 epochs.
  - Validation data is used to monitor the model's performance on unseen data.

- **Model Evaluation**:
  - After training, the best model weights are loaded, and the model is evaluated on the test dataset.
  - The test accuracy is reported.



### Dataset Availability
The modified EEG dataset utilized in this project is available for researchers and developers interested in EEG signal classification and analysis. For access to this modified dataset and further details, please [contact us](ali.amini.scholar@gmail.com).


### Contributions
Your contributions will help make this project more robust and beneficial for the community. We look forward to collaborating with you!

---

*This project is part of an ongoing effort to apply advanced machine learning techniques for meaningful real-world applications, particularly in the field of healthcare and neuroimaging.*

