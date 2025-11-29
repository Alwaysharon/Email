import pandas as pd
import numpy as np
import joblib
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sns

class ClassifierEvaluator:
    def __init__(self):
        self.models = {}
        self.load_trained_models()
    
    def load_trained_models(self):
        """Load all trained models"""
        model_files = {
            'naive_bayes': 'models/naive_bayes_model.pkl',
            'random_forest': 'models/random_forest_model.pkl',
            'svm': 'models/svm_model.pkl'
        }
        for name, filepath in model_files.items():
            try:
                self.models[name] = joblib.load(filepath)
                print(f"✓ Loaded {name} model")
            except FileNotFoundError:
                print(f"✗ Model file {filepath} not found")
    
    def evaluate_all_models(self, test_data_path):
        """Evaluate all models on test data"""
        df = pd.read_csv(test_data_path)
        X_test = df['email_text']
        y_test = df['category']
        print(f"Evaluating models on {len(df)} test samples...")
        print("=" * 60)
        results = {}
        for model_name, model in self.models.items():
            print(f"\n📊 Evaluating {model_name.upper()} Model")
            print("-" * 40)
            y_pred = model.predict(X_test)
            accuracy = metrics.accuracy_score(y_test, y_pred)
            results[model_name] = {
                'accuracy': accuracy,
                'predictions': y_pred,
                'classification_report': metrics.classification_report(y_test, y_pred, output_dict=True)
            }
            print(f"Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(metrics.classification_report(y_test, y_pred))
            self.display_confusion_matrix(y_test, y_pred, model_name)
        self.compare_model_performance(results)
        return results
    
    def display_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix for a model"""
        try:
            cm = metrics.confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=sorted(set(y_true)),
                       yticklabels=sorted(set(y_true)))
            plt.title(f'Confusion Matrix - {model_name.upper()}')
            plt.xlabel('Predicted Category')
            plt.ylabel('True Category')
            plt.tight_layout()
            plt.savefig(f'models/confusion_matrix_{model_name}.png', dpi=300, bbox_inches='tight')
            plt.show()
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
    
    def compare_model_performance(self, results):
        """Compare model performance"""
        print("\n" + "=" * 60)
        print("🏆 MODEL COMPARISON")
        print("=" * 60)
        comparison_data = [
            {
                'Model': model_name.upper(),
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision (avg)': f"{result['classification_report']['weighted avg']['precision']:.4f}",
                'Recall (avg)': f"{result['classification_report']['weighted avg']['recall']:.4f}",
                'F1-Score (avg)': f"{result['classification_report']['weighted avg']['f1-score']:.4f}"
            }
            for model_name, result in results.items()
        ]
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        print(f"\n🥇 Best performing model: {best_model_name.upper()}")
        print(f"   Accuracy: {best_accuracy:.4f}")
        self.plot_accuracy_comparison(results)
    
    def plot_accuracy_comparison(self, results):
        """Plot model accuracy comparison"""
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        plt.figure(figsize=(10, 6))
        bars = plt.bar([m.upper() for m in models], accuracies, color=['#4f46e5', '#10b981', '#f59e0b'])
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                     f'{accuracy:.3f}', ha='center', va='bottom', fontweight='bold')
        plt.title('Model Accuracy Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Accuracy', fontsize=12)
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('models/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_misclassified_emails(self, test_data_path, model_name='naive_bayes'):
        """Analyze misclassified emails"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        df = pd.read_csv(test_data_path)
        X_test = df['email_text']
        y_test = df['category']
        y_pred = self.models[model_name].predict(X_test)
        misclassified = df[y_test != y_pred].copy()
        misclassified['predicted'] = y_pred[y_test != y_pred]
        print(f"\n🔍 MISCLASSIFICATION ANALYSIS - {model_name.upper()}")
        print("=" * 60)
        print(f"Total misclassified: {len(misclassified)} out of {len(df)} ({len(misclassified)/len(df)*100:.1f}%)")
        if len(misclassified) > 0:
            print("\nMisclassification by True Category:")
            misclass_by_category = misclassified.groupby('category').size()
            for category, count in misclass_by_category.items():
                total_in_category = (y_test == category).sum()
                error_rate = count / total_in_category * 100
                print(f"  {category}: {count}/{total_in_category} ({error_rate:.1f}%)")
            print("\nSample Misclassifications:")
            print("-" * 40)
            for idx, row in misclassified.head(5).iterrows():
                print(f"True: {row['category']} | Predicted: {row['predicted']}")
                print(f"Email: {row['email_text'][:100]}...")
                print("-" * 40)
    
    def evaluate_custom_emails(self, custom_emails, model_name='naive_bayes'):
        """Test model on custom email examples"""
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        print(f"\n🧪 TESTING CUSTOM EMAILS - {model_name.upper()}")
        print("=" * 60)
        for i, email in enumerate(custom_emails, 1):
            prediction = self.models[model_name].predict([email])[0]
            probabilities = self.models[model_name].predict_proba([email])[0]
            classes = self.models[model_name].classes_
            max_prob = probabilities.max()
            print(f"\n📧 Email {i}:")
            print(f"Text: {email[:80]}...")
            print(f"Predicted Category: {prediction}")
            print(f"Confidence: {max_prob:.3f}")
            prob_dict = dict(zip(classes, probabilities))
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            print("Top 3 probabilities:")
            for category, prob in sorted_probs:
                print(f"  {category}: {prob:.3f}")

def main():
    evaluator = ClassifierEvaluator()
    if not evaluator.models:
        print("No models found. Please train models first using train_model.py")
        return
    try:
        results = evaluator.evaluate_all_models('data/email_dataset.csv')
        evaluator.analyze_misclassified_emails('data/email_dataset.csv', 'naive_bayes')
        custom_test_emails = [
            "Please approve the budget for the new marketing campaign targeting millennials.",
            "Server downtime scheduled for maintenance this weekend from 2-6 AM.",
            "Congratulations on your promotion! HR will contact you about salary adjustments.",
            "Legal review required for the new vendor contract before we can proceed.",
            "Warehouse inventory shows we're running low on packaging materials.",
            "Performance review meeting scheduled for next Tuesday at 3 PM."
        ]
        evaluator.evaluate_custom_emails(custom_test_emails, 'naive_bayes')
    except FileNotFoundError:
        print("Test data not found. Please run data_generator.py first to create the dataset.")

if __name__ == "__main__":
    main()
