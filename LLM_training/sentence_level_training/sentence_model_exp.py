# SMOTE VERSION

import pandas as pd
import numpy as np
import torch
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
from sklearn.utils import resample
from sentence_experiment_tracker import ExperimentTracker
from codecarbon import EmissionsTracker
from datetime import datetime
from imblearn.over_sampling import SMOTE

# Define function for logging model performance
def log_model_performance(model_name, level, hyperparams, metrics, notes=""):
    """Log the model performance to the experiment tracker."""
    tracker = ExperimentTracker()
    experiment_id = tracker.log_experiment(
        model_name=model_name,
        level=level,
        hyperparams=hyperparams,
        metrics=metrics,
        notes=notes
    )
    print(f"Experiment logged with ID: {experiment_id}")
    return experiment_id


# Check for MPS availability (M1 Mac)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

# File path - accounting for running from LLM_training directory
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
        print("Available files in ../data/:")
        if os.path.exists('../../data'):
            print(os.listdir('../../data'))

# Load data from Excel with better error handling
try:
    # First try to import openpyxl to give a clearer error if it's missing
    try:
        import openpyxl

        print("openpyxl is installed")
    except ImportError:
        print("ERROR: openpyxl is not installed. Please run: pip install openpyxl")
        raise ImportError("Missing dependency: openpyxl. Run 'pip install openpyxl' to fix.")

    # Now try to read the Excel file
    xls = pd.ExcelFile(file_path)
    available_sheets = xls.sheet_names
    print(f"Available sheets in Excel file: {available_sheets}")

    # Use the first sheet by default, or Sheet1 if it exists
    sheet_to_use = available_sheets[0]
    if 'Sheet1' in available_sheets:
        sheet_to_use = 'Sheet1'

    # Read the Excel file
    data = pd.read_excel(file_path, sheet_name=sheet_to_use)
    print(f"Successfully loaded {len(data)} rows from Excel sheet '{sheet_to_use}'")

    # Print the first few rows to understand the structure
    print("\nFirst few rows of the Excel data:")
    print(data.head())

    # Print column names to help with debugging
    print("\nColumn names in Excel file:")
    print(data.columns.tolist())

except Exception as e:
    print(f"Error loading Excel file: {e}")
    print("Please check that the Excel file exists and the path is correct.")

    # Provide helpful instructions for fixing the path
    print("\nTo fix this issue:")
    print("1. Install openpyxl: pip install openpyxl")
    print("2. Check the file path. Your current working directory is:", os.getcwd())
    print("3. The script is looking for the file at:", os.path.abspath(file_path))
    print("4. You may need to adjust the path depending on where you run the script from.")

    raise Exception("Failed to load Excel file. See suggestions above.")


# Preprocess the data for sentence-level data
def preprocess_data(df):
    """Clean and prepare the sentence-level data from the Excel format shown in the screenshot"""
    print("\nChecking columns in the Excel file...")

    # Based on the screenshot, we can identify these key columns
    possible_sentence_cols = ['Sentence', 'sentence', 'text', 'dialog']
    sentence_col = None
    for col in possible_sentence_cols:
        if col in df.columns:
            sentence_col = col
            print(f"Found sentence column: '{sentence_col}'")
            break

    possible_manip_cols = ['Manipulative', 'manipulative', 'manipulation_1', 'manipulation']
    manipulative_col = None
    for col in possible_manip_cols:
        if col in df.columns:
            manipulative_col = col
            print(f"Found manipulative column: '{manipulative_col}'")
            break

    # If we still can't find the columns, look for alternatives
    if not sentence_col:
        for col in df.columns:
            if df[col].dtype == 'object' and not col.startswith('Unnamed'):
                # Check if this column contains text data
                test_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                if len(test_val) > 5:  # Assume text with some length is the sentence
                    sentence_col = col
                    print(f"Using column '{sentence_col}' as sentence text")
                    break

    if not manipulative_col:
        # Look for columns that have binary or numeric values
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

    # Handle case where inner_id or another ID column is present
    id_col = None
    possible_id_cols = ['inner_id', 'Inner_id', 'ID', 'id']
    for col in possible_id_cols:
        if col in df.columns:
            id_col = col
            print(f"Found ID column: '{id_col}'")
            break

    # If we still can't find the columns, we need to stop
    if not sentence_col:
        raise ValueError("Could not find a suitable sentence column in the Excel file!")
    if not manipulative_col:
        raise ValueError("Could not find a suitable manipulative/label column in the Excel file!")

    # Clean text data - add extra error handling
    print(f"Cleaning sentence text from column '{sentence_col}'")
    try:
        # Check for NaN values first and replace with empty string
        df[sentence_col] = df[sentence_col].fillna("")
        df[sentence_col] = df[sentence_col].astype(str).apply(lambda x: x.strip())
    except Exception as e:
        print(f"Warning: Error cleaning sentence text: {e}")
        print("Attempting more basic cleaning...")
        df[sentence_col] = df[sentence_col].fillna("").astype(str)

    # Ensure the manipulative column is binary (0 or 1)
    print(f"Converting manipulative column '{manipulative_col}' to binary labels")
    try:
        df['label'] = df[manipulative_col].astype(int)
    except:
        # Handle text values or conversion issues
        if df[manipulative_col].dtype == 'object':
            mapping = {'yes': 1, 'no': 0, 'true': 1, 'false': 0, 'Yes': 1, 'No': 0, 'True': 1, 'False': 0}
            df['label'] = df[manipulative_col].map(mapping)
            if df['label'].isna().any():
                # If mapping failed, try to convert any numeric strings
                try:
                    df['label'] = df[manipulative_col].astype(float).astype(int)
                except:
                    # Last resort - if value > 0, consider it True
                    print("Warning: Complex label conversion - using threshold > 0")
                    df['label'] = (df[manipulative_col].replace('', np.nan)
                                   .fillna(0)
                                   .astype(float) > 0).astype(int)
        else:
            # Attempt a direct conversion for numeric values
            df['label'] = (df[manipulative_col] > 0).astype(int)

    # Create a clean DataFrame with just the columns we need
    # Use a memory-efficient subset of the data
    clean_df = pd.DataFrame({
        'sentence': df[sentence_col],
        'label': df['label']
    })

    # Add ID column if available (and needed)
    if id_col:
        clean_df['id'] = df[id_col]

    print(f"Preprocessing complete. Dataset contains {len(clean_df)} rows with sentence text and binary labels.")
    return clean_df


# Preprocess the data
data = preprocess_data(data)

# Print dataset statistics
print(f"Total examples: {len(data)}")
print(f"Manipulative examples: {sum(data['label'] == 1)}")
print(f"Non-Manipulative examples: {sum(data['label'] == 0)}")

# Define label mappings
id2label = {0: "Non-Manipulative", 1: "Manipulative"}
label2id = {"Non-Manipulative": 0, "Manipulative": 1}

# Extract features and labels
sentences = data['sentence'].tolist()
labels = data['label'].tolist()

# Check class imbalance
class_counts = data['label'].value_counts()
print("Class distribution:")
print(class_counts)

# Apply class balancing through SMOTE if needed
if class_counts[0] != class_counts[1]:
    print("Applying class balancing through SMOTE (Synthetic Minority Over-sampling)...")


    # Create a feature representation of sentences
    # We'll use a simple approach: sentence length and basic counts
    def extract_features(text):
        # Convert to string if not already
        text = str(text)
        return [
            len(text),  # Length of text
            text.count(' ') + 1,  # Word count (approximate)
            sum(1 for c in text if c.isupper()),  # Count of uppercase chars
            sum(1 for c in text if c in '.,?!;')  # Count of punctuation
        ]


    # Create feature array for SMOTE
    print("Extracting features for SMOTE...")
    X_features = np.array([extract_features(s) for s in sentences])
    y = np.array(labels)

    # Print class distribution before SMOTE
    print(f"Original dataset - Class 0: {sum(y == 0)}, Class 1: {sum(y == 1)}")

    # Apply SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_features, y)

        # Create mapping back to original sentences
        # Keep original sentences where possible, create indices mapping
        indices_map = {}
        for i, features in enumerate(X_features):
            # Use features as a key (converted to tuple for hashability)
            indices_map[tuple(features)] = i

        # Create new sentences list
        new_sentences = []
        for features in X_resampled:
            features_tuple = tuple(features)
            if features_tuple in indices_map:
                # If these features match an original sentence, use that sentence
                new_sentences.append(sentences[indices_map[features_tuple]])
            else:
                # For synthetic examples, find the nearest original sentence as a placeholder
                # In a real system, you might want to generate new text
                distances = [np.linalg.norm(np.array(features) - np.array(f)) for f in X_features]
                closest_idx = np.argmin(distances)
                new_sentences.append(sentences[closest_idx])

        # Update sentences and labels
        sentences = new_sentences
        labels = y_resampled.tolist()

        print(f"Balanced dataset using SMOTE - Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)}")

    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Continuing with original imbalanced dataset and relying on class weights")

# Split into train, validation and test sets
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    sentences, labels, test_size=0.3, random_state=42, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}")

# Define a custom dataset class for sentences
class SentenceDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=96):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Handle potential NaN or None values
        if pd.isna(text):
            text = ""

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Remove batch dimension added by tokenizer when return_tensors="pt"
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        return encoding


# Initialize the tokenizer and model
# Using a smaller model for better training on MPS device
model_name = "distilroberta-base"  # Lighter model that's faster to train
print(f"Using {model_name} model which is more efficient for training")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create datasets with reduced max_length for efficiency
max_length = 64  # Most sentences will fit in this length
train_dataset = SentenceDataset(train_texts, train_labels, tokenizer, max_length=max_length)
val_dataset = SentenceDataset(val_texts, val_labels, tokenizer, max_length=max_length)
test_dataset = SentenceDataset(test_texts, test_labels, tokenizer, max_length=max_length)

# Load the model with more explicit initialization
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification"  # Explicitly set classification type
).to(device)

# Initialize weights properly for better training
# This is crucial for preventing the "predict everything as one class" problem
for name, param in model.named_parameters():
    if 'classifier' in name:  # Only initialize the classification head
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)


# Define enhanced evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro"
    )
    acc = (predictions == labels).mean()
    return {
        "accuracy": acc,
        "macro_f1": f1,
        "precision": precision,
        "recall": recall
    }


# Check transformers version
import transformers
import inspect

print(f"Transformers version: {transformers.__version__}")

# Inspect actual parameters supported by TrainingArguments
training_args_params = inspect.signature(TrainingArguments.__init__).parameters
print(f"Supported TrainingArguments parameters: {list(training_args_params.keys())}")

# Define training arguments with version compatibility check
# Start with basic parameters that work across versions
training_args_dict = {
    "output_dir": "./dialogue_models/sentence_model_exp",  # Different output directory
    "learning_rate": 2e-5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "num_train_epochs": 4,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 50,
    "save_total_limit": 1,  # Only keep the best model
    "dataloader_num_workers": 0,
    "fp16": False,
    "seed": 42
}

# Add newer parameters only if supported
if "evaluation_strategy" in training_args_params:
    print("Using newer-style evaluation parameters")
    training_args_dict.update({
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 100,
        "warmup_ratio": 0.1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "report_to": "none"
    })
elif "eval_strategy" in training_args_params:
    # Your version has 'eval_strategy' instead of 'evaluation_strategy'
    print("Using compatible evaluation parameters for your version")
    training_args_dict.update({
        "eval_strategy": "steps",
        "eval_steps": 100,
        "save_steps": 100,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_macro_f1",
        "greater_is_better": True,
        "warmup_ratio": 0.1
    })
else:
    print("Using older-style evaluation parameters")
    training_args_dict.update({
        "eval_steps": 100,
        "save_steps": 100,
        "warmup_steps": 50
    })

# Create training arguments from the dictionary
training_args = TrainingArguments(**training_args_dict)
print("Training arguments successfully created")


# Simple loss function with label smoothing but no class weights
def compute_loss(outputs, labels):
    logits = outputs.logits
    # Still use label smoothing for better generalization
    label_smoothing = 0.1
    # Use standard cross entropy with label smoothing only
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

# Apply custom loss to model
model.compute_loss = compute_loss

# Define the Trainer with proper error handling and class weights
try:
    # First check if we can use early stopping with the current transformers version
    from transformers import EarlyStoppingCallback

    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    print("Using trainer with early stopping")

except Exception as e:
    print(f"Could not use early stopping callback: {e}")
    # Fallback without early stopping
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Using basic Trainer without early stopping callback")

# Train the model with error handling
carbon_metrics = {
    "total_emissions_kg": 0.0,
    "emissions_rate_kg_per_epoch": 0.0,
    "equivalent_miles_driven": 0.0
}
try:
    print("Starting training...")
    # Initialize carbon tracker
    # Initialize carbon tracker
    carbon_tracker = EmissionsTracker(
        project_name="mental-manipulation-detection",
        experiment_id=f"sentence-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    carbon_tracker.start()
    trainer.train()
    # Stop carbon tracking
    carbon_emissions = carbon_tracker.stop()
    carbon_metrics = {
        "total_emissions_kg": float(f"{carbon_emissions:.10f}"),  # Format with 10 decimal places
        "emissions_rate_kg_per_epoch": float(f"{carbon_emissions / training_args.num_train_epochs:.10f}") if training_args.num_train_epochs > 0 else 0,
        "equivalent_miles_driven": float(f"{carbon_emissions * 2.24:.10f}")
    }
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    print("Try reducing batch size or using a smaller model")

# Save the model with error handling for disk space issues
try:
    print("Saving the model...")
    model_path = "../../../Copy/LLM_training/sentence_level_training/dialogue_models/sentence_model_exp"  # Shorter path name

    # First check if directory exists and has files that can be removed
    if os.path.exists(model_path):
        print(f"Clearing existing files in {model_path} to free space...")
        for file_name in os.listdir(model_path):
            file_path = os.path.join(model_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    # Save with error handling
    try:
        trainer.save_model(model_path)
        print(f"Model successfully saved to {model_path}")
    except OSError as e:
        if "No space left on device" in str(e):
            print("ERROR: No space left on device. Attempting alternative save method...")

            # Try to save just the model weights which is smaller
            import shutil

            os.makedirs("./temp_model", exist_ok=True)
            torch.save(model.state_dict(), "./temp_model/pytorch_model.bin")
            print("Saved model weights only to ./temp_model/pytorch_model.bin")

            # Provide instructions for loading later
            print("\nTo load this model later, use:")
            print("model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base', num_labels=2)")
            print("model.load_state_dict(torch.load('./temp_model/pytorch_model.bin'))")
        else:
            print(f"Error saving model: {e}")
except Exception as e:
    print(f"Unexpected error when saving model: {e}")
    print("Trying alternative save method...")
    try:
        # Save just state_dict as a last resort
        torch.save(model.state_dict(), "model_weights.bin")
        print("Saved model weights only to model_weights.bin")
    except Exception as e2:
        print(f"All save attempts failed: {e2}")
        print("Training was still successful even though saving failed")

# Evaluate on test set
print("Evaluating on test set...")
all_preds = []
all_labels = []
all_probs = []  # Initialize these variables regardless of evaluation method

try:
    results = trainer.evaluate(test_dataset)
    print("Test Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # Store metrics for experiment logging
    accuracy = results["eval_accuracy"]
    f1 = results["eval_macro_f1"]
    precision = results["eval_precision"]
    recall = results["eval_recall"]

    # We still need to collect predictions and probabilities for threshold analysis
    print("Collecting predictions for threshold analysis...")
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Prob of class 1 (Manipulative)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Attempting manual evaluation...")

    # Manual evaluation as fallback
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Prob of class 1 (Manipulative)

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro')

    print(f"Test Results (Manual Evaluation):")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")

# Threshold Analysis
print("\n--- Threshold Analysis ---")
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
results = []

for threshold in thresholds:
    preds = [1 if prob > threshold else 0 for prob in all_probs]
    acc = (np.array(preds) == np.array(all_labels)).mean()
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average='macro')

    results.append({
        'threshold': threshold,
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    })

    print(f"Threshold {threshold:.2f}: Acc={acc:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

# Find best threshold
best = max(results, key=lambda x: x['f1'])
print(f"\nBest threshold: {best['threshold']:.2f} with F1={best['f1']:.4f}")
best_threshold = best['threshold']

# Confusion Matrix and Class-Specific Metrics
print("\n--- Confusion Matrix and Class-Specific Metrics ---")
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Non-Manipulative", "Manipulative"]))

# Calculate balanced metrics
if conf_matrix.shape == (2, 2):  # Ensure it's a 2x2 matrix
    true_pos_rate = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[
        1, 1]) > 0 else 0  # Sensitivity
    true_neg_rate = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[
        0, 1]) > 0 else 0  # Specificity
    balanced_acc = (true_pos_rate + true_neg_rate) / 2

    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
    print(f"Sensitivity (True Positive Rate): {true_pos_rate:.4f}")
    print(f"Specificity (True Negative Rate): {true_neg_rate:.4f}")


# Define helper function for inference with threshold support
def predict_manipulation(text, threshold=None):
    # Use the best threshold from analysis if available, otherwise use 0.5
    if threshold is None:
        threshold = best_threshold if 'best_threshold' in globals() else 0.5

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    manipulative_prob = probabilities[0][1].item()

    # Use threshold for prediction
    prediction = 1 if manipulative_prob > threshold else 0

    label = id2label[prediction]
    confidence = manipulative_prob if prediction == 1 else 1 - manipulative_prob

    return {
        "label": label,
        "confidence": confidence,
        "raw_manipulative_prob": manipulative_prob
    }


# Test on sample sentences from the screenshot and some additional examples
test_samples = [
    "I can almost feel your doubt about what I'm saying.",
    "I don't have no intention whatsoever of hurting you.",
    "I don't believe it.",
    "She's a charming girl, isn't she Mitch?",
    "She was selling birds.",
    "You'd think he could manage to keep her name straight.",
    "Hello, how are you doing today?",
    "You're sure easy to talk to.",
    "I didn't mean that the way it sounded.",
    "You look like you could use some help."
]

print("\nTesting on sample inputs:")
for sample in test_samples:
    result = predict_manipulation(sample, best_threshold)
    print(f"Text: '{sample}'")
    print(
        f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f}, Raw prob: {result['raw_manipulative_prob']:.2f})")
    print("-" * 50)

# ======= EXPERIMENT TRACKING SECTION (MOVED TO THE END) =======
# Now that we have all the variables defined, log the experiment

# Collect hyperparameters
hyperparams = {
    "model_name": model_name,
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "epochs": training_args.num_train_epochs,
    "weight_decay": training_args.weight_decay,
    "max_length": max_length,
    "threshold": best_threshold if 'best_threshold' in locals() else 0.5,
    "label_smoothing": 0.1  # Adding this parameter since we're using it
}

# Collect metrics from evaluation
metrics = {
    "accuracy": accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "balanced_accuracy": balanced_acc if 'balanced_acc' in locals() else None,
    **carbon_metrics
}

# Log the experiment
notes = "Sentence-level model trained with balancing by SMOTE."
experiment_id = log_model_performance(
    model_name=model_name,
    level="sentence",
    hyperparams=hyperparams,
    metrics=metrics,
    notes=notes
)

# You can also save the experiment ID with the model for reference
try:
    with open(os.path.join(model_path, "experiment_id.txt"), "w") as f:
        f.write(str(experiment_id))
    print(f"Experiment ID {experiment_id} saved with model")
except Exception as e:
    print(f"Could not save experiment ID to model path: {e}")
    print(f"However, experiment was logged successfully with ID: {experiment_id}")

print("\n===== EXPERIMENT TRACKING COMPLETE =====")
print(f"You can view this experiment (ID: {experiment_id}) using the experiment tracker's functions.")
print(
    "To view all experiments, run: python -c \"from experiment_tracker import ExperimentTracker; tracker = ExperimentTracker(); print(tracker.get_all_experiments())\"")
print("To visualize results, use tracker.visualize_results() or the comparison_script.py")

# === SENTENCE SELECTION FOR EXPLAINABILITY ===
print("\n" + "=" * 60)
print("SENTENCE SELECTION FOR EXPLAINABILITY METHODS")
print("=" * 60)

# Categorize sentences for selection with detailed analysis
categories = {
    'manip_high_conf_correct': [],  # High confidence correct manipulative predictions
    'manip_med_conf_correct': [],  # Medium confidence correct manipulative predictions
    'manip_low_conf_correct': [],  # Low confidence correct manipulative predictions
    'manip_misclassified': [],  # Misclassified manipulative sentences (false negatives)
    'non_manip_high_conf_correct': [],  # High confidence correct non-manipulative predictions
    'non_manip_med_conf_correct': [],  # Medium confidence correct non-manipulative predictions
    'non_manip_low_conf_correct': [],  # Low confidence correct non-manipulative predictions
    'non_manip_misclassified': []  # Misclassified non-manipulative sentences (false positives)
}

# Enhanced categorization with more granular confidence levels
print("Categorizing sentences based on prediction confidence and correctness...")

for i, (text, true_label, pred_label, prob) in enumerate(zip(test_texts, all_labels, all_preds, all_probs)):
    confidence = prob if pred_label == 1 else (1 - prob)
    correct = (true_label == pred_label)

    # Create detailed sentence info
    sentence_info = {
        'index': i,
        'text': text,
        'true_label': true_label,
        'pred_label': pred_label,
        'manipulative_prob': prob,
        'confidence': confidence,
        'correct': correct,
        'text_length': len(text),
        'word_count': len(text.split()) if text else 0
    }

    if true_label == 1:  # Actually manipulative
        if correct:  # Correctly predicted as manipulative
            if confidence > 0.8:
                categories['manip_high_conf_correct'].append(sentence_info)
            elif confidence > 0.6:
                categories['manip_med_conf_correct'].append(sentence_info)
            else:
                categories['manip_low_conf_correct'].append(sentence_info)
        else:  # Misclassified (false negative)
            categories['manip_misclassified'].append(sentence_info)
    else:  # Actually non-manipulative
        if correct:  # Correctly predicted as non-manipulative
            if confidence > 0.8:
                categories['non_manip_high_conf_correct'].append(sentence_info)
            elif confidence > 0.6:
                categories['non_manip_med_conf_correct'].append(sentence_info)
            else:
                categories['non_manip_low_conf_correct'].append(sentence_info)
        else:  # Misclassified (false positive)
            categories['non_manip_misclassified'].append(sentence_info)

# Print summary statistics
print("\nCategory Statistics:")
print("-" * 40)
for cat_name, items in categories.items():
    print(f"{cat_name.replace('_', ' ').title()}: {len(items)} sentences")


# Enhanced printing function with more details
def print_sentence_category(cat_name, items, max_show=10, show_stats=True):
    """Print category with enhanced details and statistics"""
    print(f"\n{'=' * 20} {cat_name.upper().replace('_', ' ')} {'=' * 20}")
    print(f"Total sentences in category: {len(items)}")

    if not items:
        print("No sentences in this category.")
        return

    # Show statistics if requested
    if show_stats and len(items) > 0:
        confidences = [item['confidence'] for item in items]
        text_lengths = [item['text_length'] for item in items]
        word_counts = [item['word_count'] for item in items]

        print(f"Confidence - Mean: {np.mean(confidences):.3f}, Std: {np.std(confidences):.3f}")
        print(f"Text Length - Mean: {np.mean(text_lengths):.1f}, Std: {np.std(text_lengths):.1f}")
        print(f"Word Count - Mean: {np.mean(word_counts):.1f}, Std: {np.std(word_counts):.1f}")

    # Sort items for display
    if 'misclassified' in cat_name:
        # For misclassified, show most uncertain first (closest to 0.5 probability)
        items_sorted = sorted(items, key=lambda x: abs(0.5 - x['manipulative_prob']))
    else:
        # For correct predictions, show highest confidence first
        items_sorted = sorted(items, key=lambda x: x['confidence'], reverse=True)

    print(f"\nShowing top {min(max_show, len(items))} examples:")
    print("-" * 80)

    for i, item in enumerate(items_sorted[:max_show]):
        print(f"\n{i + 1:2d}. [Index: {item['index']:3d}] "
              f"True: {id2label[item['true_label']]}, "
              f"Pred: {id2label[item['pred_label']]}")
        print(f"    Confidence: {item['confidence']:.3f}, "
              f"Manip Prob: {item['manipulative_prob']:.3f}, "
              f"Length: {item['text_length']} chars, "
              f"Words: {item['word_count']}")
        print(f"    Text: \"{item['text']}\"")


# Print all categories with 10 examples each
print("\n" + "=" * 60)
print("DETAILED SENTENCE EXAMPLES BY CATEGORY")
print("=" * 60)

for cat_name, items in categories.items():
    print_sentence_category(cat_name, items, max_show=10, show_stats=True)

# === RECOMMENDED SELECTION FOR EXPLAINABILITY ===
print("\n" + "=" * 60)
print("RECOMMENDED SENTENCE SELECTION FOR EXPLAINABILITY")
print("=" * 60)


def select_diverse_sentences(categories, num_per_category=5):
    """Select diverse sentences for explainability analysis"""
    selected_sentences = {}

    for cat_name, items in categories.items():
        if not items:
            selected_sentences[cat_name] = []
            continue

        # Sort appropriately for each category
        if 'misclassified' in cat_name:
            # For misclassified, select most uncertain and most confident wrong predictions
            items_sorted = sorted(items, key=lambda x: abs(0.5 - x['manipulative_prob']))
        else:
            # For correct predictions, select highest and medium confidence
            items_sorted = sorted(items, key=lambda x: x['confidence'], reverse=True)

        # Select diverse examples based on confidence and length
        selected = []
        if len(items_sorted) >= num_per_category:
            # Take examples from different confidence ranges and lengths
            selected.append(items_sorted[0])  # Highest confidence/most uncertain

            if num_per_category > 1:
                # Add examples from different parts of the sorted list
                step = len(items_sorted) // num_per_category
                for i in range(1, num_per_category):
                    idx = min(i * step, len(items_sorted) - 1)
                    selected.append(items_sorted[idx])
        else:
            # Take all available
            selected = items_sorted[:num_per_category]

        selected_sentences[cat_name] = selected

    return selected_sentences


# Select recommended sentences
recommended_sentences = select_diverse_sentences(categories, num_per_category=5)

print("RECOMMENDED SENTENCES FOR EXPLAINABILITY ANALYSIS:")
print("=" * 50)

total_selected = 0
for cat_name, sentences in recommended_sentences.items():
    if sentences:
        print(f"\n{cat_name.replace('_', ' ').title()} ({len(sentences)} selected):")
        for i, sentence in enumerate(sentences):
            print(f"  {i + 1}. [ID: {sentence['index']}] Conf: {sentence['confidence']:.3f}")
            print(f"     \"{sentence['text']}\"")
        total_selected += len(sentences)

print(f"\nTotal sentences selected: {total_selected}")
print("\nThese sentences provide a comprehensive view of model behavior across:")
print("- High, medium, and low confidence correct predictions")
print("- Misclassified examples (both false positives and false negatives)")
print("- Both manipulative and non-manipulative sentence types")

# === SAVE SELECTED SENTENCES FOR FURTHER ANALYSIS ===
print("\n" + "=" * 60)
print("SAVING SELECTED SENTENCES")
print("=" * 60)

# Create a comprehensive dataset of selected sentences
selected_for_analysis = []

for cat_name, sentences in recommended_sentences.items():
    for sentence in sentences:
        selected_for_analysis.append({
            'category': cat_name,
            'index': sentence['index'],
            'text': sentence['text'],
            'true_label': id2label[sentence['true_label']],
            'predicted_label': id2label[sentence['pred_label']],
            'confidence': sentence['confidence'],
            'manipulative_probability': sentence['manipulative_prob'],
            'text_length': sentence['text_length'],
            'word_count': sentence['word_count'],
            'correct_prediction': sentence['correct']
        })

# Save to CSV for easy analysis
try:
    import pandas as pd

    df_selected = pd.DataFrame(selected_for_analysis)
    output_file = "./selected_sentences_for_explainability.csv"
    df_selected.to_csv(output_file, index=False)
    print(f"Selected sentences saved to: {output_file}")

    # Print summary
    print(f"\nSummary of saved sentences:")
    print(f"Total sentences: {len(df_selected)}")
    print(f"Categories represented: {df_selected['category'].nunique()}")
    print(f"Correct predictions: {df_selected['correct_prediction'].sum()}")
    print(f"Incorrect predictions: {len(df_selected) - df_selected['correct_prediction'].sum()}")

    # Show some examples from each category
    print(f"\nExample sentences by category:")
    for category in df_selected['category'].unique():
        cat_sentences = df_selected[df_selected['category'] == category]
        if len(cat_sentences) > 0:
            print(f"\n{category.replace('_', ' ').title()}:")
            example = cat_sentences.iloc[0]
            print(f"  \"{example['text']}\" (Conf: {example['confidence']:.3f})")

except Exception as e:
    print(f"Could not save to CSV: {e}")
    print("Selected sentences are still available in the 'selected_for_analysis' variable")

# === ANALYSIS BY SENTENCE CHARACTERISTICS ===
print("\n" + "=" * 60)
print("ANALYSIS BY SENTENCE CHARACTERISTICS")
print("=" * 60)

# Analyze patterns in misclassified sentences
print("MISCLASSIFIED SENTENCE ANALYSIS:")
print("-" * 30)

manip_misclassified = categories['manip_misclassified']
non_manip_misclassified = categories['non_manip_misclassified']

if manip_misclassified:
    print(f"\nManipulative sentences misclassified as non-manipulative ({len(manip_misclassified)}):")
    avg_length = np.mean([s['text_length'] for s in manip_misclassified])
    avg_words = np.mean([s['word_count'] for s in manip_misclassified])
    avg_conf = np.mean([s['confidence'] for s in manip_misclassified])
    print(f"  Average length: {avg_length:.1f} chars, {avg_words:.1f} words")
    print(f"  Average confidence: {avg_conf:.3f}")

    print("\n  Most uncertain examples:")
    sorted_manip_misc = sorted(manip_misclassified, key=lambda x: abs(0.5 - x['manipulative_prob']))[:3]
    for i, sent in enumerate(sorted_manip_misc):
        print(f"    {i + 1}. \"{sent['text']}\" (Prob: {sent['manipulative_prob']:.3f})")

if non_manip_misclassified:
    print(f"\nNon-manipulative sentences misclassified as manipulative ({len(non_manip_misclassified)}):")
    avg_length = np.mean([s['text_length'] for s in non_manip_misclassified])
    avg_words = np.mean([s['word_count'] for s in non_manip_misclassified])
    avg_conf = np.mean([s['confidence'] for s in non_manip_misclassified])
    print(f"  Average length: {avg_length:.1f} chars, {avg_words:.1f} words")
    print(f"  Average confidence: {avg_conf:.3f}")

    print("\n  Most uncertain examples:")
    sorted_non_manip_misc = sorted(non_manip_misclassified, key=lambda x: abs(0.5 - x['manipulative_prob']))[:3]
    for i, sent in enumerate(sorted_non_manip_misc):
        print(f"    {i + 1}. \"{sent['text']}\" (Prob: {sent['manipulative_prob']:.3f})")

print("\n" + "=" * 60)
print("SENTENCE SELECTION COMPLETE")

print("\n===== SENTENCE-LEVEL ANALYSIS COMPLETE =====")
print(f"Selected sentences are available in 'selected_for_analysis' variable")
print(f"All categorized sentences are available in 'categories' variable")