import os
import re
import flask
import joblib
import nltk

app = flask.Flask(__name__)

class EmailClassifierService:
    def __init__(self):
        self.models = {}
        self.load_trained_models()
        self.stop_words = set(nltk.corpus.stopwords.words('english'))
    
    def load_trained_models(self):
        """Load pre-trained models"""
        model_files = {
            'naive_bayes': 'models/naive_bayes_model.pkl',
            'random_forest': 'models/random_forest_model.pkl',
            'svm': 'models/svm_model.pkl'
        }
        for name, filepath in model_files.items():
            if os.path.exists(filepath):
                self.models[name] = joblib.load(filepath)
                print(f"Loaded {name} model")
            else:
                print(f"Model file {filepath} not found")
    
    def clean_text(self, text):
        """Clean and preprocess email text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Tokenize and remove stopwords
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        return ' '.join(tokens)
    
    def predict_category(self, email_text, model_name='naive_bayes'):
        """Classify email and return prediction with probabilities"""
        if model_name not in self.models:
            return None, None
        processed_text = self.clean_text(email_text)
        try:
            prediction = self.models[model_name].predict([processed_text])[0]
            probabilities = self.models[model_name].predict_proba([processed_text])[0]
            classes = self.models[model_name].classes_
            prob_dict = dict(zip(classes, probabilities))
            sorted_probs = dict(sorted(prob_dict.items(), key=lambda x: x[1], reverse=True))
            return prediction, sorted_probs
        except Exception as e:
            print(f"Error in classification: {e}")
            return None, None

# Initialize classifier
classifier = EmailClassifierService()

@app.route('/')
def index():
    """Main page"""
    return flask.render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_email_endpoint():
    """API endpoint for email classification"""
    try:
        data = flask.request.get_json()
        email_text = data.get('email_text', '')
        model_name = data.get('model', 'naive_bayes')
        if not email_text.strip():
            return flask.jsonify({'error': 'Email text is required'}), 400
        prediction, probabilities = classifier.predict_category(email_text, model_name)
        if prediction is None:
            return flask.jsonify({'error': 'Classification failed'}), 500
        return flask.jsonify({
            'prediction': prediction,
            'probabilities': probabilities,
            'confidence': probabilities[prediction] if probabilities else 0
        })
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500

@app.route('/batch_classify', methods=['POST'])
def batch_classify_emails():
    """API endpoint for batch email classification"""
    try:
        data = flask.request.get_json()
        emails = data.get('emails', [])
        model_name = data.get('model', 'naive_bayes')
        if not emails:
            return flask.jsonify({'error': 'No emails provided'}), 400
        results = []
        i = 0
        while i < len(emails):
            email_text = emails[i]
            prediction, probabilities = classifier.predict_category(email_text, model_name)
            results.append({
                'index': i,
                'email': email_text[:100] + '...' if len(email_text) > 100 else email_text,
                'prediction': prediction,
                'confidence': probabilities[prediction] if probabilities else 0
            })
            i += 1
        return flask.jsonify({'results': results})
    except Exception as e:
        return flask.jsonify({'error': str(e)}), 500

@app.route('/health')
def check_health():
    """Health check endpoint"""
    return flask.jsonify({
        'status': 'healthy',
        'models_loaded': list(classifier.models.keys())
    })

if __name__ == '__main__':
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
