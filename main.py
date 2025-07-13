import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()
DATA_PATH = os.getenv('DATA_PATH', './data')

# Generate synthetic data if not present
os.makedirs(DATA_PATH, exist_ok=True)
data_file = os.path.join(DATA_PATH, 'documents.csv')
if not os.path.exists(data_file):
    samples = [
        # Invoice samples
        ("Invoice for services rendered. Total due: $500.", "invoice"),
        ("Payment due upon receipt. Invoice #12345.", "invoice"),
        ("Invoice: Consulting services, 10 hours at $50/hr.", "invoice"),
        ("Invoice date: 2023-01-15. Amount: $1200.", "invoice"),
        ("Please find attached the invoice for your recent purchase.", "invoice"),
        ("Invoice #67890. Payment terms: Net 30 days.", "invoice"),
        ("This is a reminder that your invoice is overdue.", "invoice"),
        ("Invoice for web development project. Total: $2000.", "invoice"),
        ("Invoice issued for graphic design services.", "invoice"),
        ("Final invoice for completed work.", "invoice"),
        # Resume samples
        ("Curriculum Vitae: John Doe. Experience in software engineering.", "resume"),
        ("Skills: Python, Machine Learning. Education: BSc Computer Science.", "resume"),
        ("Professional Experience: Data Analyst at XYZ Corp.", "resume"),
        ("Education: MBA, Harvard Business School.", "resume"),
        ("Work History: Software Developer, 5 years.", "resume"),
        ("Certifications: AWS Certified Solutions Architect.", "resume"),
        ("Languages: English, Spanish, French.", "resume"),
        ("Objective: Seeking a challenging position in IT.", "resume"),
        ("References available upon request.", "resume"),
        ("Summary: Experienced project manager with PMP certification.", "resume"),
        # Report samples
        ("Quarterly report on company performance. Revenue increased by 10%.", "report"),
        ("Annual report: Financial summary and future outlook.", "report"),
        ("Research report: Analysis of market trends in 2022.", "report"),
        ("Monthly report: Sales figures and analysis.", "report"),
        ("Project report: Milestones achieved and next steps.", "report"),
        ("Incident report: Summary of events and actions taken.", "report"),
        ("Technical report: System architecture and implementation.", "report"),
        ("Audit report: Findings and recommendations.", "report"),
        ("Progress report: Tasks completed and pending.", "report"),
        ("Evaluation report: Assessment of project outcomes.", "report"),
    ]
    df = pd.DataFrame(samples, columns=["text", "label"])
    df.to_csv(data_file, index=False)
    logging.info(f"Synthetic data generated at {data_file}")
else:
    logging.info(f"Data file found at {data_file}")

# 1. Load data
logging.info("Loading data...")
df = pd.read_csv(data_file)

# 2. Preprocess data
logging.info("Preprocessing data...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

# 3. Train/test split
logging.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
logging.info("Training model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# 5. Evaluate
logging.info("Evaluating model...")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
plt.close()
logging.info('Confusion matrix saved as confusion_matrix.png')

# Print final accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Final Accuracy: {accuracy*100:.2f}%')
logging.info(f'Final Accuracy: {accuracy*100:.2f}%')

# Test prediction with test_samples.csv
import csv

test_file = 'test_samples.csv'
if os.path.exists(test_file):
    print('\nTest predictions on test_samples.csv:')
    with open(test_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            test_text = row['text']
            test_vec = vectorizer.transform([test_text])
            pred_label = model.predict(test_vec)[0]
            print(f"Text: {test_text}\nPredicted document type: {pred_label}\n")
            logging.info(f"Test sample classified as: {pred_label}")
else:
    print("No test_samples.csv file found. Skipping test predictions.")
