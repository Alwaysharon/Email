import os
import subprocess
import sys
import nltk
import ssl

def install_packages():
    """Install required Python packages"""
    requirements = [
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.0.0',
        'nltk>=3.6',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'joblib>=1.0.0'
    ]
    print("Installing Python packages...")
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")

def configure_nltk():
    """Setup NLTK data with comprehensive error handling"""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    nltk_resources = [
        'punkt_tab',
        'punkt',
        'stopwords',
        'wordnet',
        'omw-1.4'
    ]
    print("\nDownloading NLTK data...")
    for data in nltk_resources:
        try:
            nltk.download(data, quiet=True)
            print(f"✓ Downloaded {data}")
        except Exception as e:
            print(f"✗ Failed to download {data}: {e}")

def ensure_directories_exist():
    """Create necessary directories"""
    directories = ['data', 'models']
    print("\nCreating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created {directory}/ directory")

def create_sample_data():
    """Create a sample dataset for testing"""
    import pandas as pd
    sample_data = {
        'email_text': [
            "Please review the quarterly budget report and approve the marketing expenses for next quarter.",
            "System maintenance will be performed tonight from 11 PM to 3 AM. Email services may be unavailable.",
            "New employee orientation scheduled for Monday morning in the main conference room at 9 AM.",
            "Contract terms need legal review before we can proceed with the vendor partnership agreement.",
            "Inventory levels are critically low. Please reorder office supplies as soon as possible.",
            "The marketing campaign results exceeded our expectations with 25% increase in conversions.",
            "IT support ticket #12345 has been resolved. Your computer should now be working properly.",
            "HR policy update: remote work guidelines have been revised effective immediately.",
            "Legal compliance training is mandatory for all employees. Please complete by Friday.",
            "Supply chain disruption may affect delivery schedules. Please plan accordingly."
        ],
        'category': [
            'finance', 'it', 'hr', 'legal', 'operations',
            'marketing', 'it', 'hr', 'legal', 'operations'
        ]
    }
    df = pd.DataFrame(sample_data)
    sample_path = 'data/email_dataset.csv'
    if not os.path.exists(sample_path):
        df.to_csv(sample_path, index=False)
        print(f"✓ Created sample dataset at {sample_path}")
    else:
        print(f"✓ Dataset already exists at {sample_path}")

def main():
    """Main setup function"""
    print("Setting up Email Classifier environment...")
    print("=" * 50)
    install_packages()
    configure_nltk()
    ensure_directories_exist()
    create_sample_data()
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("You can now run: python train-model.py")

if __name__ == "__main__":
    main()
