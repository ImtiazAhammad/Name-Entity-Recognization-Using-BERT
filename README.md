# Named Entity Recognition (NER) Pipeline with BERT

This repository contains a complete pipeline for performing Named Entity Recognition (NER) using the BERT language model. The pipeline is implemented using the Hugging Face Transformers library and includes data preprocessing, model training, evaluation, and inference.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Training and Evaluation](#training-and-evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Inference](#inference)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to fine-tune BERT for the NER task using the CoNLL-2003 dataset. The solution aligns input tokens with NER labels and handles subword tokenization challenges.

## Dataset
We use the [CoNLL-2003 dataset](https://huggingface.co/datasets/conll2003) for NER, which contains labeled examples for named entities such as persons, locations, and organizations.

## Model Architecture
- **Pretrained Model:** BERT (bert-base-uncased)
- **Fine-tuning:** BERT is fine-tuned for sequence classification with NER labels.

## Installation
1. Clone the repository and navigate to the project directory.
```bash
git clone "https://github.com/ImtiazAhammad/Name-Entity-Recognization-Using-BERT.git"
cd <project-directory>
```

2. Install the required packages:
```bash
pip install transformers datasets tokenizers seqeval ipywidgets
```

## Usage
To train and evaluate the NER model, follow these steps:

### 1. Preprocess and Align Labels
Tokenize the dataset and align NER tags with the tokens.

### 2. Model Training
Run the following command to start training:
```python
trainer.train()
```

### 3. Save and Load the Model
```python
model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")
```

## Training and Evaluation
Run the training process using the Hugging Face Trainer API and monitor performance metrics such as precision, recall, F1 score, and accuracy.

## Saving and Loading the Model
After training, you can save and reload the fine-tuned model:
```python
model.save_pretrained("ner_model")
tokenizer.save_pretrained("tokenizer")
```

## Results
The fine-tuned BERT model achieves competitive performance on the NER task with accurate detection of named entities.

## Contributing
Contributions are welcome! Please submit a pull request or raise an issue for improvements.

## License
This project is licensed under the MIT License.

