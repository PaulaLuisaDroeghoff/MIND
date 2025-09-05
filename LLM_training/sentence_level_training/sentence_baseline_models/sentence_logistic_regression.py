# LOGISTIC REGRESSION BASELINE VERSION - Simplified

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report, \
    accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import time

# File path - same as your BERT code
file_path = '/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/data/mentalmanip_detailed_sentencelevel.xlsx'

# Check if file exists before attempting to load
if not os.path.exists(file_path):
    print(f"Warning: File not found at {file_path}")
    print("Current working directory:", os.getcwd())
    print("Looking for file in parent directory...")
    parent_file_path = os.path.join('../../../Copy', 'data', 'mentalmanip_detailed_sentencelevel.xlsx')
    if os.path.exists(parent_file_path):
        print(f"Found file at {parent_file_path}")
        file_path = parent_file_path
    else:
        print("File not found in parent directory either.")

# Load data from Excel
try:
    import openpyxl

    print("openpyxl is installed")

    xls = pd.ExcelFile(file_path)
    available_sheets = xls.sheet_names
    print(f"Available sheets in Excel file: {available_sheets}")

    sheet_to_use = available_sheets[0]
    if 'Sheet1' in available_sheets:
        sheet_to_use = 'Sheet1'

    data = pd.read_excel(file_path, sheet_name=sheet_to_use)
    print(f"Successfully loaded {len(data)} rows from Excel sheet '{sheet_to_use}'")

    print("\nFirst few rows of the Excel data:")
    print(data.head())
    print("\nColumn names in Excel file:")
    print(data.columns.tolist())

except Exception as e:
    print(f"Error loading Excel file: {e}")
    raise Exception("Failed to load Excel file. See suggestions above.")


# Use the same preprocessing function from your BERT code
def preprocess_data(df):
    """Clean and prepare the sentence-level data from the Excel format"""
    print("\nChecking columns in the Excel file...")

    # Find sentence column
    possible_sentence_cols = ['Sentence', 'sentence', 'text', 'dialog']
    sentence_col = None
    for col in possible_sentence_cols:
        if col in df.columns:
            sentence_col = col
            print(f"Found sentence column: '{sentence_col}'")
            break

    # Find manipulative column
    possible_manip_cols = ['Manipulative', 'manipulative', 'manipulation_1', 'manipulation']
    manipulative_col = None
    for col in possible_manip_cols:
        if col in df.columns:
            manipulative_col = col
            print(f"Found manipulative column: '{manipulative_col}'")
            break

    # Auto-detect columns if not found
    if not sentence_col:
        for col in df.columns:
            if df[col].dtype == 'object' and not col.startswith('Unnamed'):
                test_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                if len(test_val) > 5:
                    sentence_col = col
                    print(f"Using column '{sentence_col}' as sentence text")
                    break

    if not manipulative_col:
        for col in df.columns:
            if col != sentence_col:
                try:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) <= 2 and all(
                            isinstance(x, (int, float, np.integer, np.floating)) or x in ('0', '1') for x in
                            unique_vals):
                        manipulative_col = col
                        print(f"Using column '{manipulative_col}' as manipulative label")
                        break
                except:
                    continue

    if not sentence_col or not manipulative_col:
        raise ValueError("Could not find suitable sentence or label columns!")

    # Clean text data
    print(f"Cleaning sentence text from column '{sentence_col}'")
    df[sentence_col] = df[sentence_col].fillna("").astype(str).apply(lambda x: x.strip())

    # Convert labels to binary
    print(f"Converting manipulative column '{manipulative_col}' to binary labels")
    try:
        df['label'] = df[manipulative_col].astype(int)
    except:
        if df[manipulative_col].dtype == 'object':
            mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 1, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
            df['label'] = df[manipulative_col].map(mapping)
            if df['label'].isna().any():
                try:
                    df['label'] = df[manipulative_col].astype(float).astype(int)
                except:
                    df['label'] = (df[manipulative_col].replace('', np.nan)
                                   .fillna(0)
                                   .astype(float) > 0).astype(int)
        else:
            df['label'] = (df[manipulative_col] > 0).astype(int)

    # Create clean DataFrame
    clean_df = pd.DataFrame({
        'sentence': df[sentence_col],
        'label': df['label']
    })

    print(f"Preprocessing complete. Dataset contains {len(clean_df)} rows with sentence text and binary labels.")
    return clean_df


# Preprocess the data
data = preprocess_data(data)

# Print dataset statistics
print(f"\n===== DATASET STATISTICS =====")
print(f"Total examples: {len(data)}")
print(f"Manipulative examples: {sum(data['label'] == 1)}")
print(f"Non-Manipulative examples: {sum(data['label'] == 0)}")

# Extract features and labels
sentences = data['sentence'].tolist()
labels = data['label'].tolist()

# Check class imbalance
class_counts = data['label'].value_counts()
print("Class distribution:")
print(class_counts)

# Split into train, validation and test sets (same as BERT)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    sentences, labels, test_size=0.3, random_state=42, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"\n===== DATA SPLITS =====")
print(f"Train size: {len(train_texts)}")
print(f"Validation size: {len(val_texts)}")
print(f"Test size: {len(test_texts)}")

# Create logistic regression pipelines
print("\n===== CREATING MODELS =====")

pipelines = {
    'LogReg_Balanced': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'  # Handle class imbalance
        ))
    ]),

    'LogReg_SMOTE': ImbPipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english'
        )),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1000
        ))
    ])
}

# Train and evaluate each pipeline
results = {}

for name, pipeline in pipelines.items():
    print(f"\n===== TRAINING {name} =====")

    # Time the training
    start_time = time.time()
    pipeline.fit(train_texts, train_labels)
    training_time = time.time() - start_time

    print(f"Training completed in {training_time:.2f} seconds")

    # Test on test set
    test_predictions = pipeline.predict(test_texts)
    test_probabilities = pipeline.predict_proba(test_texts)[:, 1]

    # Calculate test metrics
    test_accuracy = accuracy_score(test_labels, test_predictions)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_predictions, average='macro'
    )

    print(f"\n--- Test Results for {name} ---")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"Macro F1: {test_f1:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")

    # Confusion Matrix
    print(f"\nConfusion Matrix:")
    conf_matrix = confusion_matrix(test_labels, test_predictions)
    print(conf_matrix)

    # Calculate additional metrics
    if conf_matrix.shape == (2, 2):
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
        balanced_acc = (sensitivity + specificity) / 2

        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Sensitivity (TPR): {sensitivity:.4f}")
        print(f"Specificity (TNR): {specificity:.4f}")

    print(f"\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions,
                                target_names=["Non-Manipulative", "Manipulative"]))

    # Threshold analysis
    print(f"\n--- Threshold Analysis for {name} ---")
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
    threshold_results = []

    for threshold in thresholds:
        preds = [1 if prob > threshold else 0 for prob in test_probabilities]
        acc = accuracy_score(test_labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(test_labels, preds, average='macro')

        threshold_results.append({
            'threshold': threshold,
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        })

        print(f"Threshold {threshold:.2f}: Acc={acc:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

    # Find best threshold
    best = max(threshold_results, key=lambda x: x['f1'])
    print(f"\nBest threshold: {best['threshold']:.2f} with F1={best['f1']:.4f}")

    # Store results
    results[name] = {
        'pipeline': pipeline,
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'best_threshold': best['threshold'],
        'best_threshold_f1': best['f1'],
        'training_time': training_time,
        'predictions': test_predictions,
        'probabilities': test_probabilities
    }

# Find and display best performing model
best_model_name = max(results.keys(), key=lambda x: results[x]['test_f1'])
best_model = results[best_model_name]

print(f"\n{'=' * 50}")
print(f"BEST MODEL: {best_model_name}")
print(f"{'=' * 50}")
print(f"Test Accuracy: {best_model['test_accuracy']:.4f}")
print(f"Test F1 Score: {best_model['test_f1']:.4f}")
print(f"Test Precision: {best_model['test_precision']:.4f}")
print(f"Test Recall: {best_model['test_recall']:.4f}")
print(f"Best Threshold: {best_model['best_threshold']:.2f}")
print(f"F1 at Best Threshold: {best_model['best_threshold_f1']:.4f}")
print(f"Training Time: {best_model['training_time']:.2f} seconds")


# Define prediction function
def predict_manipulation(text, threshold=None):
    """Predict manipulation using the best logistic regression model"""
    if threshold is None:
        threshold = best_model['best_threshold']

    # Get probability
    prob = best_model['pipeline'].predict_proba([text])[0][1]

    # Use threshold for prediction
    prediction = 1 if prob > threshold else 0

    label = "Manipulative" if prediction == 1 else "Non-Manipulative"
    confidence = prob if prediction == 1 else 1 - prob

    return {
        "label": label,
        "confidence": confidence,
        "raw_manipulative_prob": prob
    }


# Test on sample sentences
test_samples = [
    "I could always get us a bottle.",
    "No. I'll take you home.",
    "Tell 'em your father gave it to you.",
    "Joey, what's more important, the kids' clothes or your sexual potency.",
    "I swear to God, our father, that when you change into one of the undead, I will kill you."
]

print(f"\n{'=' * 60}")
print(f"TESTING ON SAMPLE SENTENCES ({best_model_name})")
print(f"{'=' * 60}")

for i, sample in enumerate(test_samples, 1):
    result = predict_manipulation(sample)
    print(f"{i:2d}. Text: '{sample}'")
    print(f"    Prediction: {result['label']}")
    print(f"    Confidence: {result['confidence']:.3f}")
    print(f"    Raw Prob: {result['raw_manipulative_prob']:.3f}")
    print()

# Summary comparison between dialogue_models
print(f"\n{'=' * 50}")
print("MODEL COMPARISON SUMMARY")
print(f"{'=' * 50}")
print(f"{'Model':<20} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Time(s)':<10}")
print("-" * 70)

for name, result in results.items():
    print(f"{name:<20} {result['test_accuracy']:<10.4f} {result['test_f1']:<10.4f} "
          f"{result['test_precision']:<12.4f} {result['test_recall']:<10.4f} {result['training_time']:<10.2f}")

print(f"\nRecommendation: Use {best_model_name} as your baseline for comparison with BERT.")
print(f"If BERT doesn't significantly outperform this (e.g., F1 improvement < 0.05),")
print(f"then the simpler logistic regression model might be sufficient for your task.")

# Optional: Save the best model
try:
    model_path = "best_logistic_model.pkl"
    joblib.dump(best_model['pipeline'], model_path)
    print(f"\nBest model saved as '{model_path}'")
    print(f"Load it later with: model = joblib.load('{model_path}')")
except Exception as e:
    print(f"Could not save model: {e}")

print(f"\n{'=' * 50}")
print("ANALYSIS COMPLETE")
print(f"{'=' * 50}")