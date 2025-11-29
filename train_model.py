import pandas as pd
import numpy as np
import re
import nltk
import sklearn.model_selection
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.svm
import sklearn.metrics as metrics
import sklearn.pipeline
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ssl

os.makedirs('models', exist_ok=True)

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_resources():
    """Download required NLTK data with comprehensive error handling"""
    required_data = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('tokenizers/punkt', 'punkt'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    for path, name in required_data:
        try:
            nltk.data.find(path)
            print(f"✓ {name} already available")
        except LookupError:
            print(f"Downloading {name}...")
            try:
                nltk.download(name, quiet=True)
                print(f"✓ {name} downloaded successfully")
            except Exception as e:
                print(f"✗ Failed to download {name}: {e}")
                if name == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                        print("✓ Downloaded punkt as fallback")
                    except:
                        pass

download_nltk_resources()

class EmailModelTrainer:
    def __init__(self):
        self.models = {}
        self.vectorizer = None
        try:
            self.stop_words = set(nltk.corpus.stopwords.words('english'))
        except LookupError:
            print("Stopwords not available, using basic set")
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def clean_text(self, text):
        """Clean and preprocess email text"""
        if pd.isna(text) or text == '':
            return ''
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        try:
            tokens = nltk.tokenize.word_tokenize(text)
        except LookupError:
            tokens = text.split()
        tokens = list(filter(lambda word: word not in self.stop_words and len(word) > 2, tokens))
        return ' '.join(tokens)
    
    def load_and_prepare_data(self, filepath):
        """Load and preprocess the email dataset"""
        print("Loading dataset...")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset file not found: {filepath}")
        df = pd.read_csv(filepath)
        required_columns = ['email_text', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        print(f"Dataset shape: {df.shape}")
        print(f"Category distribution:\n{df['category'].value_counts()}")
        initial_rows = len(df)
        df = df.dropna(subset=['email_text', 'category'])
        print(f"Removed {initial_rows - len(df)} rows with missing data")
        print("Preprocessing text data...")
        df['processed_text'] = df['email_text'].apply(self.clean_text)
        df = df[df['processed_text'].str.len() > 0]
        print(f"Final dataset shape after preprocessing: {df.shape}")
        return df
    
    def train_all_models(self, df):
        """Train multiple classification models"""
        features = df['processed_text']
        labels = df['category']
        if len(features) == 0:
            raise ValueError("No data available for training after preprocessing")
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=labels
        )
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        self.vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        models_to_train = {
            'naive_bayes': sklearn.naive_bayes.MultinomialNB(alpha=0.1),
            'random_forest': sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'svm': sklearn.svm.SVC(kernel='linear', random_state=42, probability=True)
        }
        best_model_name = None
        best_accuracy = 0.0
        results = {}
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            pipeline = sklearn.pipeline.Pipeline([
                ('tfidf', self.vectorizer),
                ('classifier', model)
            ])
            try:
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                accuracy = metrics.accuracy_score(y_test, y_pred)
                print(f"{name} accuracy: {accuracy:.4f}")
                self.models[name] = pipeline
                results[name] = {
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'model': pipeline
                }
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model_name = name
                print(f"\nClassification Report for {name}:")
                print(metrics.classification_report(y_test, y_pred))
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
        if not best_model_name:
            raise ValueError("No models trained successfully")
        print(f"\nBest model: {best_model_name} with accuracy: {best_accuracy:.4f}")
        self.display_confusion_matrix(y_test, results[best_model_name]['predictions'], best_model_name)
        return X_test, y_test, best_model_name
    
    def display_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        try:
            cm = metrics.confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sorted(set(y_true)),
                       yticklabels=sorted(set(y_true)))
            plt.title(f'Confusion Matrix - {model_name}')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.tight_layout()
            plot_path = f'models/confusion_matrix_{model_name}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {plot_path}")
            plt.show()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def save_trained_models(self):
        """Save trained models and vectorizer"""
        print("\nSaving models...")
        if not self.models:
            print("No models to save")
            return
        try:
            for name, model in self.models.items():
                model_path = f'models/{name}_model.pkl'
                joblib.dump(model, model_path)
                print(f"✓ Saved {name} model to {model_path}")
            print("All models saved successfully!")
        except Exception as e:
            print(f"Error saving models: {e}")
    
    def predict_category_for_email(self, email_text, model_name='naive_bayes'):
        """Predict category for a single email"""
        if model_name not in self.models:
            available_models = list(self.models.keys())
            if not available_models:
                raise ValueError("No trained models available")
            model_name = available_models[0]
            print(f"Model {model_name} not found. Using {available_models[0]} instead.")
        try:
            processed_text = self.clean_text(email_text)
            if not processed_text:
                return "unknown", {"unknown": 1.0}
            prediction = self.models[model_name].predict([processed_text])[0]
            probability = self.models[model_name].predict_proba([processed_text])[0]
            classes = self.models[model_name].classes_
            prob_dict = dict(zip(classes, probability))
            return prediction, prob_dict
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "error", {"error": 1.0}

def main():
    try:
        trainer = EmailModelTrainer()
        dataset_path = 'data/emails.csv'
        if not os.path.exists('data'):
            os.makedirs('data')
            print(f"Created data directory. Please place your email dataset at: {dataset_path}")
            return
        if not os.path.exists(dataset_path):
            print(f"Dataset not found at: {dataset_path}")
            print("Please ensure your dataset is in the correct location with columns: 'email_text' and 'category'")
            return
        df = trainer.load_and_prepare_data(dataset_path)
        X_test, y_test, best_model_name = trainer.train_all_models(df)
        trainer.save_trained_models()
        print("\n" + "="*50)
        print("Testing with sample emails:")
        print("="*50)
        test_emails = [
            "Please review the quarterly budget report and approve the marketing expenses.",
            "System maintenance will be performed tonight. Email services may be unavailable.",
            "New employee orientation scheduled for Monday morning in the conference room.",
            "Contract terms need legal review before we can proceed with the vendor.",
            "Inventory levels are low. Please reorder office supplies as soon as possible."
        ]
        for email in test_emails:
            prediction, probabilities = trainer.predict_category_for_email(email, best_model_name)
            print(f"\nEmail: {email[:60]}...")
            print(f"Predicted Category: {prediction}")
            if prediction != "error":
                print(f"Confidence: {probabilities[prediction]:.3f}")
                print(f"All probabilities: {dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))}")
        print(f"\nTraining completed successfully!")
        print(f"Best model: {best_model_name}")
        print(f"Models saved in: ./models/")
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
