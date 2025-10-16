# TP 1 NLP — Fine-tuning BERT for Text Classification with PyTorch

## Description

This project consists of fine-tuning a pre-trained **BERT** model for **binary text classification** (positive/negative sentiment analysis) using **PyTorch** and **Hugging Face Transformers**.
The dataset used is **SST-2 (Stanford Sentiment Treebank)**, a benchmark dataset for sentiment analysis.

---

## Objective

Implement a fine-tuned BERT model to classify text into one of two classes:

* Positive sentiment
* Negative sentiment

The model will be evaluated using standard metrics for classification.

---

## Tasks Overview

1. **Data Loading and Exploration**

   * Use the **SST-2 dataset** from the Hugging Face `datasets` library.
   * Explore the dataset: label distribution, sample texts, etc.

2. **Preprocessing**

   * Tokenize the text using `BertTokenizer` (for example, `bert-base-uncased`).
   * Create attention masks and input IDs.

3. **Model Setup**

   * Load a pre-trained **BERT** model with a classification head (`BertForSequenceClassification`).
   * Use `num_labels=2` for binary classification.

4. **Training**

   * Use the **Adam optimizer** and a **learning rate scheduler**.
   * Fine-tune the model for approximately **10 epochs** (adjust depending on validation performance).
   * Use GPU acceleration if available (`torch.cuda.is_available()`).

5. **Evaluation**

   * Make predictions on the test set.
   * Compute evaluation metrics:

     * Accuracy
     * F1-score
     * Confusion matrix

---

## Requirements

Install the following dependencies before running the code:

```bash
pip install torch torchvision torchaudio
pip install transformers datasets scikit-learn matplotlib
```

---

## How to Run

1. Clone or download this project:

   ```bash
   git clone https://github.com/yourusername/tp1-nlp-bert.git
   cd tp1-nlp-bert
   ```

2. Run the training script:

   ```bash
   python train.py
   ```

3. After training, evaluate the model:

   ```bash
   python evaluate.py
   ```

---

## Expected Output

After training, the following results should be obtained:

* Training loss curve
* Validation accuracy and F1-score
* Confusion matrix visualization
* A fine-tuned BERT model capable of sentiment classification

---

## Tips

* Experiment with different learning rates, batch sizes, and number of epochs.
* Consider using the `Trainer` API from the `transformers` library for simplicity.
* Save your trained model for later use:

  ```python
  model.save_pretrained("fine_tuned_bert")
  tokenizer.save_pretrained("fine_tuned_bert")
  ```

---
