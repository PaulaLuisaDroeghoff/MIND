# LOGISTIC REGRESSION DIALOGUE LEVEL VERSION

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

# File path
file_path = '/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/data/dialoguelevel_mentalmanip_detailed.xlsx'

# Check if file exists before attempting to load
if not os.path.exists(file_path):
    print(f"Warning: File not found at {file_path}")
    print("Current working directory:", os.getcwd())
    print("Looking for file in parent directory...")
    parent_file_path = os.path.join('../../../Copy', 'data', 'mentalmanip_detailed_dialoglevel.xlsx')
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


def preprocess_dialogue_data(df):
    """Clean and prepare the dialogue-level data from the Excel format"""
    print("\nChecking columns in the Excel file...")

    # Find dialogue column - common names for dialogue data
    possible_dialogue_cols = ['Dialogue', 'dialogue', 'text', 'conversation', 'Dialog', 'dialog', 'script', 'content']
    dialogue_col = None
    for col in possible_dialogue_cols:
        if col in df.columns:
            dialogue_col = col
            print(f"Found dialogue column: '{dialogue_col}'")
            break

    # Find manipulative column
    possible_manip_cols = ['Manipulative', 'manipulative', 'manipulation_1', 'manipulation', 'label', 'Label']
    manipulative_col = None
    for col in possible_manip_cols:
        if col in df.columns:
            manipulative_col = col
            print(f"Found manipulative column: '{manipulative_col}'")
            break

    # Auto-detect columns if not found
    if not dialogue_col:
        for col in df.columns:
            if df[col].dtype == 'object' and not col.startswith('Unnamed'):
                test_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                # Dialogues are typically longer than sentences
                if len(test_val) > 50:  # Increased threshold for dialogue detection
                    dialogue_col = col
                    print(f"Using column '{dialogue_col}' as dialogue text")
                    break

    if not manipulative_col:
        for col in df.columns:
            if col != dialogue_col:
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

    if not dialogue_col or not manipulative_col:
        raise ValueError("Could not find suitable dialogue or label columns!")

    # Clean dialogue text data
    print(f"Cleaning dialogue text from column '{dialogue_col}'")
    df[dialogue_col] = df[dialogue_col].fillna("").astype(str).apply(lambda x: x.strip())

    # Remove very short dialogues (likely incomplete data)
    initial_count = len(df)
    df = df[df[dialogue_col].str.len() > 10]  # Remove dialogues shorter than 10 characters
    print(f"Removed {initial_count - len(df)} very short dialogues")

    # Convert labels to binary
    print(f"Converting manipulative column '{manipulative_col}' to binary labels")
    try:
        df['label'] = df[manipulative_col].astype(int)
    except:
        if df[manipulative_col].dtype == 'object':
            mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
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
        'dialogue': df[dialogue_col],
        'label': df['label']
    })

    # Print dialogue length statistics
    dialogue_lengths = clean_df['dialogue'].str.len()
    print(f"\nDialogue length statistics:")
    print(f"Mean length: {dialogue_lengths.mean():.1f} characters")
    print(f"Median length: {dialogue_lengths.median():.1f} characters")
    print(f"Min length: {dialogue_lengths.min()} characters")
    print(f"Max length: {dialogue_lengths.max()} characters")

    print(f"Preprocessing complete. Dataset contains {len(clean_df)} dialogues with binary labels.")
    return clean_df


# Preprocess the data
data = preprocess_dialogue_data(data)

# Print dataset statistics
print(f"\n===== DATASET STATISTICS =====")
print(f"Total dialogues: {len(data)}")
print(f"Manipulative dialogues: {sum(data['label'] == 1)}")
print(f"Non-Manipulative dialogues: {sum(data['label'] == 0)}")

# Extract features and labels
dialogues = data['dialogue'].tolist()
labels = data['label'].tolist()

# Check class imbalance
class_counts = data['label'].value_counts()
print("Class distribution:")
print(class_counts)

# Split into train, validation and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    dialogues, labels, test_size=0.3, random_state=42, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"\n===== DATA SPLITS =====")
print(f"Train size: {len(train_texts)}")
print(f"Validation size: {len(val_texts)}")
print(f"Test size: {len(test_texts)}")

# Create logistic regression pipelines optimized for dialogue-level text
print("\n===== CREATING MODELS =====")

pipelines = {
    'LogReg_Balanced_Dialogue': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,  # Increased for longer dialogue text
            ngram_range=(1, 3),  # Include trigrams for better dialogue context
            min_df=2,
            max_df=0.90,  # Slightly lower to handle dialogue-specific patterns
            strip_accents='unicode',
            lowercase=True,
            stop_words='english',
            sublinear_tf=True  # Good for longer documents
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1500,  # Increased iterations for dialogue complexity
            class_weight='balanced',
            C=1.0  # Default regularization
        ))
    ]),

    'LogReg_SMOTE_Dialogue': ImbPipeline([
        ('tfidf', TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.90,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        )),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=1500,
            C=1.0
        ))
    ]),

    'LogReg_Regularized_Dialogue': Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=20000,  # Even more features
            ngram_range=(1, 2),  # Conservative n-grams with more regularization
            min_df=3,  # Higher min_df for dialogue
            max_df=0.85,
            strip_accents='unicode',
            lowercase=True,
            stop_words='english',
            sublinear_tf=True
        )),
        ('classifier', LogisticRegression(
            random_state=42,
            max_iter=2000,
            class_weight='balanced',
            C=0.5  # More regularization for dialogue complexity
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


# Define prediction function for dialogues
def predict_dialogue_manipulation(dialogue_text, threshold=None):
    """Predict manipulation using the best logistic regression model for dialogues"""
    if threshold is None:
        threshold = best_model['best_threshold']

    # Get probability
    prob = best_model['pipeline'].predict_proba([dialogue_text])[0][1]

    # Use threshold for prediction
    prediction = 1 if prob > threshold else 0

    label = "Manipulative" if prediction == 1 else "Non-Manipulative"
    confidence = prob if prediction == 1 else 1 - prob

    return {
        "label": label,
        "confidence": confidence,
        "raw_manipulative_prob": prob
    }


# Test on sample dialogues (create some example dialogues for testing)
test_dialogue_samples = [
    "Person A: You know what would make me really happy? Person B: What? Person A: If you could lend me some money. Person B: I don't know... Person A: Come on, I thought we were friends. Friends help each other out.",

    "Person A: I'm really struggling with this project. Person B: What can I do to help? Person A: Maybe you could review my work when you have time? Person B: Of course, I'd be happy to help.",

    "Person A: You never listen to me! Person B: That's not true, I do listen. Person A: No, you don't! You're always on your phone when I'm talking. Person B: I'm sorry, I'll put it away.",

    "Person A: I need you to do this for me right now. Person B: I'm busy at the moment. Person A: You're always too busy for me. You don't care about our relationship at all. Person B: That's not fair...",

    "Person A: How was your day? Person B: It was good, thanks for asking. How was yours? Person A: Pretty good too. Want to grab dinner together? Person B: Sure, that sounds nice."
]

print(f"\n{'=' * 60}")
print(f"TESTING ON SAMPLE DIALOGUES ({best_model_name})")
print(f"{'=' * 60}")

for i, sample in enumerate(test_dialogue_samples, 1):
    result = predict_dialogue_manipulation(sample)
    print(f"{i:2d}. Dialogue: '{sample[:100]}{'...' if len(sample) > 100 else ''}'")
    print(f"    Prediction: {result['label']}")
    print(f"    Confidence: {result['confidence']:.3f}")
    print(f"    Raw Prob: {result['raw_manipulative_prob']:.3f}")
    print()

# Summary comparison between dialogue_models
print(f"\n{'=' * 50}")
print("MODEL COMPARISON SUMMARY")
print(f"{'=' * 50}")
print(f"{'Model':<30} {'Accuracy':<10} {'F1':<10} {'Precision':<12} {'Recall':<10} {'Time(s)':<10}")
print("-" * 90)

for name, result in results.items():
    print(f"{name:<30} {result['test_accuracy']:<10.4f} {result['test_f1']:<10.4f} "
          f"{result['test_precision']:<12.4f} {result['test_recall']:<10.4f} {result['training_time']:<10.2f}")

print(f"\nRecommendation: Use {best_model_name} as your dialogue-level baseline for comparison with BERT.")
print(f"Note: Dialogue-level classification may show different performance patterns than sentence-level.")

# Optional: Save the best model
try:
    model_path = "best_dialogue_logistic_model.pkl"
    joblib.dump(best_model['pipeline'], model_path)
    print(f"\nBest dialogue model saved as '{model_path}'")
    print(f"Load it later with: model = joblib.load('{model_path}')")
except Exception as e:
    print(f"Could not save model: {e}")

print(f"\n{'=' * 50}")
print("DIALOGUE-LEVEL ANALYSIS COMPLETE")
print(f"{'=' * 50}")