
# BERT-Based Sentiment Analysis on IMDb Dataset

## Project Overview
This implementation of a BERT (Bidirectional Encoder Representations from Transformers) based model to perform sentiment analysis on the IMDb movie reviews dataset. The goal is to classify reviews into positive or negative sentiment categories.

## Requirements
- Python 3.x
- PyTorch
- Transformers (by Hugging Face)
- NumPy
- scikit-learn
- tqdm

## Installation
To install the required libraries, run:
```bash
pip install torch transformers numpy sklearn tqdm
```

## Dataset
The IMDb dataset contains movie reviews along with sentiment labels (positive or negative). The dataset is preprocessed to remove HTML tags and non-alphanumeric characters, and split into training and validation sets.

## Model
The project uses the `bert-base-uncased` model from the Transformers library. This pre-trained model is fine-tuned on the IMDb dataset for the sentiment analysis task.

## Training
The model is trained using the following steps:
1. Tokenization and Encoding: Reviews are tokenized and encoded using the BERT tokenizer.
2. DataLoader Creation: DataLoaders for training and validation sets are created with appropriate batch sizes.
3. Model Setup: BERT for Sequence Classification is used with binary labels.
4. Optimizer and Scheduler: AdamW optimizer with a learning rate scheduler is used.
5. Training Loop: The model is trained and validated over multiple epochs, saving the model state after each epoch.

## Evaluation
The model is evaluated using the following metrics:
- F1 Score
- AUC (Area Under Curve)
- Confusion Matrix

## Usage
To train and evaluate the model, follow the steps implemented in the attached notebook file.

## Results
The results include the model's performance metrics on the validation set. The metrics provide insights into the model's accuracy and its ability to generalize.

## License
[MIT License](LICENSE)
