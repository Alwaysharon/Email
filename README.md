# E-mail Message Classification Based on Machine Learning Algorithms

This repository contains the code for my thesis project **"E-mail message classification based on machine learning algorithms"**.  
The goal is to automatically classify enterprise e-mails into several business-oriented categories using traditional machine learning models.


## 1. Problem Statement

Modern organisations exchange a huge number of e-mails every day.  
Manually sorting messages into folders such as **HR, Finance, IT, Legal, Marketing, Operations** is time-consuming and error-prone.

This project formulates e-mail sorting as a **multi-class text classification** problem:

> Given the raw text of an e-mail, predict to which business category it belongs.

---

## 2. Dataset

### 2.1 Source

- Raw data: **Enron e-mail dataset** (CSV version, ~1.43 GB).
- Only the message body (and subject) is used; meta-fields like `From`, `To`, `Cc` etc. are removed. 

### 2.2 Cleaning & Weak Labelling

Raw Enron e-mails are noisy and unlabeled.  
I use `prep_enron_multi_cls.py` to:

1. Extract `subject + body` from each message.
2. Remove HTML tags, URLs, e-mail addresses, digits and punctuation.
3. Normalise whitespace and lowercase all text.
4. Apply **keyword-based weak labelling rules** to assign one of six categories:

- `HR`
- `Finance`
- `IT`
- `Marketing`
- `Legal`
- `Operations` 


### 2.3 Final Dataset Used in Experiments

After cleaning and filtering, the final dataset used for training & evaluation contains:

3,744 e-mails

6 balanced categories (each 624 samples):

data/email_dataset.csv   # columns: email_text, category
| Category   | Samples |
| ---------- | ------- |
| Operations | 624     |
| HR         | 624     |
| Finance    | 624     |
| Legal      | 624     |
| Marketing  | 624     |
| IT         | 624     |

The file format is:

email_text,category

"please review the quarterly budget report ...",Finance

"system maintenance will be performed tonight ...",IT

## 3. Methods
### 3.1 Pre-processing

All e-mails undergo the following pre-processing steps (see train_model.py):

Lowercasing.

Removing all non-alphabetic characters.

Normalising whitespace.

Tokenisation (NLTK word_tokenize with fall-back to simple split).

Stop-word removal (NLTK English stopwords with a small manual fallback list).

Dropping very short tokens (length ≤ 2).

The cleaned text is stored in a processed_text column and used as model input.

### 3.2 Feature Extraction

TF-IDF vectorisation using TfidfVectorizer from scikit-learn:

max_features=5000

ngram_range=(1, 2) (unigrams + bigrams)

min_df=2, max_df=0.95

stop_words='english' (scikit-learn’s built-in list as backup).

### 3.3 Models

Three classical machine learning classifiers are trained using a scikit-learn Pipeline:

Multinomial Naive Bayes (alpha=0.1)

Random Forest (n_estimators=100, n_jobs=-1)

Linear SVM (kernel='linear', probability=True)

The data is split into train/test sets with:

test_size = 0.2

stratify = category

random_state = 42

## 4. Evaluation Metrics & Results
### 4.1 Metrics

For each model I report:

Accuracy

Precision, Recall, F1-score per class

Macro-averaged precision / recall / F1

Weighted-averaged precision / recall / F1

Confusion matrix (saved as PNG for the best model)

All metrics are computed using classification_report and confusion_matrix from scikit-learn.

### 4.2 Overall Results (Test Set, 20%)
| Model         | Accuracy   | Macro Precision | Macro Recall | Macro F1 |
| ------------- | ---------- | --------------- | ------------ | -------- |
| Naive Bayes   | 0.7677     | 0.78            | 0.77         | 0.77     |
| Random Forest | **0.8852** | **0.89**        | **0.89**     | **0.88** |
| Linear SVM    | 0.8304     | 0.83            | 0.83         | 0.83     |
Random Forest achieves the best overall performance with 88.5% accuracy.

SVM also performs strongly, outperforming Naive Bayes on all macro metrics.

Naive Bayes is the simplest model but still reaches ~77% accuracy.

## 5. Project Structure
```text
email-classifier/
├─ app.py                     # Flask web application for online e-mail classification 
├─ train_model.py             # Data loading, preprocessing, model training & saving 
├─ model_evaluation.py        # Additional evaluation tools: confusion matrices, comparison, custom tests 
├─ prep_enron_multi_cls.py    # Script to create labeled dataset from raw Enron CSV via weak supervision 
├─ prep_enron_to_email_dataset.py # Alternative helper to adapt existing labeled CSVs to the project format 
├─ data/
│   └─ email_dataset.csv      # Cleaned, labeled Enron subset (not included in repo due to size/privacy)
├─ models/
│   ├─ naive_bayes_model.pkl
│   ├─ random_forest_model.pkl
│   └─ svm_model.pkl          # Trained pipelines (TF-IDF + classifier)
├─ templates/
│   └─ index.html             # Front-end page for the web demo
├─ static/                    # (Optional) CSS/JS/Images for the web UI
├─ requirements.txt           # Python dependencies (or see setup_environment.py)
└─ setup_environment.py       # Helper script to install packages & download NLTK data 
```
### 6. Web Application

The Flask app (app.py) exposes:

GET / — main HTML page (templates/index.html)

POST /classify — classify a single e-mail ({"email_text": "...", "model": "random_forest"})

POST /batch_classify — classify a list of e-mails

GET /health — simple health check, lists loaded models

Under the hood it:

Loads pre-trained models from models/*.pkl.

Applies the same text pre-processing as in training.

Returns the predicted category and full probability distribution for each class.

## 7. How to Run
### 7.1 Requirements

Python 3.8+

pip

Install dependencies (either):

pip install -r requirements.txt


or run the helper script:

python setup_environment.py

### 7.2 Prepare the Dataset

If you already have data/email_dataset.csv, you can skip this step.

Otherwise:

Download the Enron e-mail CSV (emails.csv) into data/.

Run:

python prep_enron_multi_cls.py --input data/emails.csv --output data/email_dataset.csv --sample 4000 --balance


This will create a balanced multi-class dataset with weak labels.

### 7.3 Train Models
python train_model.py


This will:

Load data/email_dataset.csv

Preprocess the text

Train Naive Bayes, Random Forest, and SVM models

Evaluate them on a held-out test set

Save the trained models to models/

### 7.4 Start the Web Demo
python app.py


Open your browser at:

http://127.0.0.1:5000

Type or paste an e-mail and get its predicted category and class probabilities.

