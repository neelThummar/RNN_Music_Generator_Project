# RNN Music Generation with LSTM (TensorFlow)
This project implements a character-level Recurrent Neural Network (RNN) using LSTM cells to generate original music in ABC notation, trained on an Irish folk music corpus. The model learns sequential musical structure and generates novel compositions that are converted into playable audio waveforms.

## Overview
- Task: Generative sequence modeling for music
- Input format: ABC music notation
- Output: Generated musical sequences synthesized into audio
- Framework: TensorFlow 2.x
- Tracking: Comet ML experiment tracking

## Model Architecture
- Embedding Layer: 256-dimensional character embeddings
- Recurrent Layer:
    Single stateful LSTM
    2048 hidden units
    Glorot uniform initialization
- Output Layer: Fully connected Dense layer projecting to vocabulary size
- Loss Function: Sparse categorical cross-entropy

## Training Configuration
- Training iterations: 5,000
- Batch size: 32
- Sequence length: 300 characters
- Learning rate: 0.005
- Checkpointing: Periodic weight saving during training
- Experiment Tracking: Hyperparameters and loss logged using Comet ML

## Dataset
- Irish folk music dataset provided by the MIT 6.S191 Introduction to Deep Learning course
- Encoded in ABC notation
- Dataset consists of multiple musical pieces concatenated into a single training corpus
- Vocabulary built from unique characters in the corpus

## Music Generation
- Generates 1000-character musical sequences
- Converts generated ABC notation into audio waveforms
- Produces multiple playable outputs demonstrating learned musical structure

## How to Run
1. Install dependencies from `requirements.txt`
2. Run the notebook
3. Listen to generated `.wav` songs

4. ## Credits
Dataset and template from [MIT Deep Learning](http://introtodeeplearning.com/)
