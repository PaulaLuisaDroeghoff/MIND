# SMOTE approach

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
from dialogue_experiment_tracker import ExperimentTracker
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

# File path - CHANGE THIS to point to your dialogue-level data
file_path = '/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/data/dialoguelevel_mentalmanip_detailed.xlsx'

# Check if file exists before attempting to load
if not os.path.exists(file_path):
    print(f"Warning: File not found at {file_path}")
    # If dialogue file doesn't exist, you could use sentence file for testing
    print("Using sentence-level file for testing...")
    file_path = '/data/dialoguelevel_mentalmanip_detailed.xlsx'

# Load data (same as sentence level)
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

# Preprocess the data - ONLY CHANGE: look for 'dialogue' instead of 'sentence'
def preprocess_data(df):
    """Clean and prepare the dialogue-level data - same logic as sentence level"""
    print("\nChecking columns in the Excel file...")

    # MAIN CHANGE: Look for dialogue columns instead of sentence columns
    possible_text_cols = ['Dialogue', 'dialogue', 'Dialog', 'dialog']
    text_col = None
    for col in possible_text_cols:
        if col in df.columns:
            text_col = col
            print(f"Found text column: '{text_col}'")
            break

    possible_manip_cols = ['Manipulative', 'manipulative', 'manipulation_1', 'manipulation']
    manipulative_col = None
    for col in possible_manip_cols:
        if col in df.columns:
            manipulative_col = col
            print(f"Found manipulative column: '{manipulative_col}'")
            break

    # Auto-detect if not found (same logic as sentence level)
    if not text_col:
        for col in df.columns:
            if df[col].dtype == 'object' and not col.startswith('Unnamed'):
                test_val = str(df[col].iloc[0]) if len(df) > 0 else ""
                if len(test_val) > 5:
                    text_col = col
                    print(f"Using column '{text_col}' as text")
                    break

    if not manipulative_col:
        for col in df.columns:
            if col != text_col:
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

    if not text_col or not manipulative_col:
        raise ValueError("Could not find suitable columns!")

    # Clean text data (same as sentence level)
    print(f"Cleaning text from column '{text_col}'")
    try:
        df[text_col] = df[text_col].fillna("")
        df[text_col] = df[text_col].astype(str).apply(lambda x: x.strip())
    except Exception as e:
        print(f"Warning: Error cleaning text: {e}")
        df[text_col] = df[text_col].fillna("").astype(str)

    # Convert labels (same as sentence level)
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
                    print("Warning: Complex label conversion - using threshold > 0")
                    df['label'] = (df[manipulative_col].replace('', np.nan)
                                   .fillna(0)
                                   .astype(float) > 0).astype(int)
        else:
            df['label'] = (df[manipulative_col] > 0).astype(int)

    # Create clean DataFrame - CHANGE: use 'dialogue' instead of 'sentence'
    clean_df = pd.DataFrame({
        'dialogue': df[text_col],  # Changed from 'sentence' to 'dialogue'
        'label': df['label']
    })

    print(f"Preprocessing complete. Dataset contains {len(clean_df)} rows with dialogue text and binary labels.")
    return clean_df

# Preprocess the data
data = preprocess_data(data)

# Print dataset statistics (same as sentence level)
print(f"Total examples: {len(data)}")
print(f"Manipulative examples: {sum(data['label'] == 1)}")
print(f"Non-Manipulative examples: {sum(data['label'] == 0)}")

# Define label mappings (same as sentence level)
id2label = {0: "Non-Manipulative", 1: "Manipulative"}
label2id = {"Non-Manipulative": 0, "Manipulative": 1}

# Extract features and labels - CHANGE: 'dialogue' instead of 'sentence'
dialogues = data['dialogue'].tolist()  # Changed from sentences
labels = data['label'].tolist()

# Check class imbalance (same as sentence level)
class_counts = data['label'].value_counts()
print("Class distribution:")
print(class_counts)

# Apply SMOTE (same logic, just different variable names)
if class_counts[0] != class_counts[1]:
    print("Applying class balancing through SMOTE (Synthetic Minority Over-sampling)...")

    def extract_features(text):
        text = str(text)
        return [
            len(text),
            text.count(' ') + 1,
            sum(1 for c in text if c.isupper()),
            sum(1 for c in text if c in '.,?!;')
        ]

    print("Extracting features for SMOTE...")
    X_features = np.array([extract_features(d) for d in dialogues])  # Changed from sentences
    y = np.array(labels)

    print(f"Original dataset - Class 0: {sum(y == 0)}, Class 1: {sum(y == 1)}")

    try:
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_features, y)

        indices_map = {}
        for i, features in enumerate(X_features):
            indices_map[tuple(features)] = i

        new_dialogues = []  # Changed from new_sentences
        for features in X_resampled:
            features_tuple = tuple(features)
            if features_tuple in indices_map:
                new_dialogues.append(dialogues[indices_map[features_tuple]])
            else:
                distances = [np.linalg.norm(np.array(features) - np.array(f)) for f in X_features]
                closest_idx = np.argmin(distances)
                new_dialogues.append(dialogues[closest_idx])

        dialogues = new_dialogues  # Changed from sentences
        labels = y_resampled.tolist()

        print(f"Balanced dataset using SMOTE - Class 0: {sum(y_resampled == 0)}, Class 1: {sum(y_resampled == 1)}")

    except Exception as e:
        print(f"SMOTE failed: {e}")
        print("Continuing with original imbalanced dataset")

# Split data (same as sentence level)
train_texts, temp_texts, train_labels, temp_labels = train_test_split(
    dialogues, labels, test_size=0.3, random_state=42, stratify=labels
)
val_texts, test_texts, val_labels, test_labels = train_test_split(
    temp_texts, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
)

print(f"Train size: {len(train_texts)}, Validation size: {len(val_texts)}, Test size: {len(test_texts)}")

# Dataset class - MAIN CHANGES: longer max_length and different name
class DialogueDataset(Dataset):  # Changed from SentenceDataset
    def __init__(self, texts, labels, tokenizer, max_length=256):  # INCREASED from 64 to 256
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if pd.isna(text):
            text = ""

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding['labels'] = torch.tensor(label, dtype=torch.long)

        return encoding

# Model setup - MAIN CHANGES: longer max_length, smaller batch size
model_name = "distilroberta-base"  # Same model as sentence level
print(f"Using {model_name} model for dialogue-level training")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# MAIN CHANGES for dialogue level:
max_length = 256  # INCREASED from 64 (dialogues are longer)
batch_size = 4   # REDUCED from 8 (to handle longer sequences)

train_dataset = DialogueDataset(train_texts, train_labels, tokenizer, max_length=max_length)
val_dataset = DialogueDataset(val_texts, val_labels, tokenizer, max_length=max_length)
test_dataset = DialogueDataset(test_texts, test_labels, tokenizer, max_length=max_length)

# Load model (same as sentence level)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2,
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification"
).to(device)

# Initialize weights (same as sentence level)
for name, param in model.named_parameters():
    if 'classifier' in name:
        if 'weight' in name:
            torch.nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            torch.nn.init.zeros_(param)

# Metrics (same as sentence level)
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

# Training arguments - MAIN CHANGES: different output dir, adjusted batch size
import transformers
import inspect

print(f"Transformers version: {transformers.__version__}")
training_args_params = inspect.signature(TrainingArguments.__init__).parameters

training_args_dict = {
    "output_dir": "./dialogue_models/dialogue_model_exp",  # CHANGED from sentence_model_exp2
    "learning_rate": 2e-5,
    "per_device_train_batch_size": batch_size,     # CHANGED from 8 to 4
    "per_device_eval_batch_size": batch_size,      # CHANGED from 8 to 4
    "gradient_accumulation_steps": 8,              # INCREASED from 4 to 8
    "num_train_epochs": 4,
    "weight_decay": 0.01,
    "logging_dir": "./logs",
    "logging_steps": 50,
    "save_total_limit": 1,
    "dataloader_num_workers": 0,
    "fp16": False,
    "seed": 42
}

# Add evaluation parameters (same logic as sentence level)
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

training_args = TrainingArguments(**training_args_dict)
print("Training arguments successfully created")

# Loss function (same as sentence level)
def compute_loss(outputs, labels):
    logits = outputs.logits
    label_smoothing = 0.1
    loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    return loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))

model.compute_loss = compute_loss

# Trainer (same as sentence level)
try:
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
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Using basic Trainer without early stopping callback")

# Training with carbon tracking (same logic, different project name)
carbon_metrics = {
    "total_emissions_kg": 0.0,
    "emissions_rate_kg_per_epoch": 0.0,
    "equivalent_miles_driven": 0.0
}

try:
    print("Starting dialogue-level training...")
    carbon_tracker = EmissionsTracker(
        project_name="mental-manipulation-detection-dialogue",  # CHANGED project name
        experiment_id=f"dialogue-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    carbon_tracker.start()
    trainer.train()
    carbon_emissions = carbon_tracker.stop()
    carbon_metrics = {
        "total_emissions_kg": float(f"{carbon_emissions:.10f}"),
        "emissions_rate_kg_per_epoch": float(f"{carbon_emissions / training_args.num_train_epochs:.10f}") if training_args.num_train_epochs > 0 else 0,
        "equivalent_miles_driven": float(f"{carbon_emissions * 2.24:.10f}")
    }
    print("Training completed successfully!")
except Exception as e:
    print(f"Error during training: {e}")
    print("Try reducing batch size or using a smaller model")

# Save model (same logic, different path)
try:
    print("Saving the model...")
    model_path = "dialogue_models/dialogue_model_exp"  # CHANGED path

    if os.path.exists(model_path):
        print(f"Clearing existing files in {model_path} to free space...")
        for file_name in os.listdir(model_path):
            file_path = os.path.join(model_path, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

    try:
        trainer.save_model(model_path)
        print(f"Model successfully saved to {model_path}")
    except OSError as e:
        if "No space left on device" in str(e):
            print("ERROR: No space left on device. Attempting alternative save method...")
            os.makedirs("./temp_model", exist_ok=True)
            torch.save(model.state_dict(), "./temp_model/dialogue_pytorch_model.bin")
            print("Saved model weights only to ./temp_model/dialogue_pytorch_model.bin")
        else:
            print(f"Error saving model: {e}")
except Exception as e:
    print(f"Unexpected error when saving model: {e}")
    try:
        torch.save(model.state_dict(), "dialogue_model_weights.bin")
        print("Saved model weights only to dialogue_model_weights.bin")
    except Exception as e2:
        print(f"All save attempts failed: {e2}")

# Evaluation (same logic as sentence level)
print("Evaluating on test set...")
all_preds = []
all_labels = []
all_probs = []

try:
    results = trainer.evaluate(test_dataset)
    print("Test Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    accuracy = results["eval_accuracy"]
    f1 = results["eval_macro_f1"]
    precision = results["eval_precision"]
    recall = results["eval_recall"]

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
            probs = torch.softmax(logits, dim=-1)[:, 1]

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Attempting manual evaluation...")

    model.eval()
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

    with torch.no_grad():
        for batch in test_dataloader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)[:, 1]

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

# Threshold Analysis (same as sentence level)
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

best = max(results, key=lambda x: x['f1'])
print(f"\nBest threshold: {best['threshold']:.2f} with F1={best['f1']:.4f}")
best_threshold = best['threshold']

# Confusion Matrix (same as sentence level)
print("\n--- Confusion Matrix and Class-Specific Metrics ---")
conf_matrix = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(conf_matrix)

print("\nDetailed Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["Non-Manipulative", "Manipulative"]))

if conf_matrix.shape == (2, 2):
    true_pos_rate = conf_matrix[1, 1] / (conf_matrix[1, 0] + conf_matrix[1, 1]) if (conf_matrix[1, 0] + conf_matrix[1, 1]) > 0 else 0
    true_neg_rate = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    balanced_acc = (true_pos_rate + true_neg_rate) / 2

    print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
    print(f"Sensitivity (True Positive Rate): {true_pos_rate:.4f}")
    print(f"Specificity (True Negative Rate): {true_neg_rate:.4f}")

# Prediction function (same logic, different name)
def predict_dialogue_manipulation(text, threshold=None):  # CHANGED function name
    if threshold is None:
        threshold = best_threshold if 'best_threshold' in globals() else 0.5

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_length).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    manipulative_prob = probabilities[0][1].item()

    prediction = 1 if manipulative_prob > threshold else 0
    label = id2label[prediction]
    confidence = manipulative_prob if prediction == 1 else 1 - manipulative_prob

    return {
        "label": label,
        "confidence": confidence,
        "raw_manipulative_prob": manipulative_prob
    }

# Test samples (changed to dialogue examples)
test_samples = [
    "Person A: I think you should reconsider this decision. Person B: I'm not sure. Person A: Well, if you don't do this, you'll regret it forever.",
    "Person A: How was your day? Person B: Good, thanks! How about yours? Person A: Pretty good too, want to grab coffee?",
    "Person A: You never listen to me. Person B: That's not true. Person A: See, you're doing it again right now.",
    "Person A: I appreciate your help with this project. Person B: No problem, happy to help!"
]

print("\nTesting on sample dialogues:")
for sample in test_samples:
    result = predict_dialogue_manipulation(sample, best_threshold)
    print(f"Text: '{sample}'")
    print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2f}, Raw prob: {result['raw_manipulative_prob']:.2f})")
    print("-" * 50)

# Experiment tracking (MAIN CHANGE: level="dialogue")
hyperparams = {
    "model_name": model_name,
    "learning_rate": training_args.learning_rate,
    "batch_size": training_args.per_device_train_batch_size,
    "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
    "epochs": training_args.num_train_epochs,
    "weight_decay": training_args.weight_decay,
    "max_length": max_length,
    "threshold": best_threshold if 'best_threshold' in locals() else 0.5,
    "label_smoothing": 0.1
}

metrics = {
    "accuracy": accuracy,
    "f1": f1,
    "precision": precision,
    "recall": recall,
    "balanced_accuracy": balanced_acc if 'balanced_acc' in locals() else None,
    **carbon_metrics
}

# MAIN CHANGE: level="dialogue" instead of "sentence"
notes = "Dialogue-level model trained with balancing by SMOTE."
experiment_id = log_model_performance(
    model_name=model_name,
    level="dialogue",  # CHANGED from "sentence"
    hyperparams=hyperparams,
    metrics=metrics,
    notes=notes
)

try:
    with open(os.path.join(model_path, "experiment_id.txt"), "w") as f:
        f.write(str(experiment_id))
    print(f"Experiment ID {experiment_id} saved with model")
except Exception as e:
    print(f"Could not save experiment ID to model path: {e}")
    print(f"However, experiment was logged successfully with ID: {experiment_id}")

# === EFFICIENT DIALOGUE SELECTION FOR EXPLAINABILITY ===
print("\n" + "=" * 60)
print("DIALOGUE SELECTION FOR EXPLAINABILITY METHODS")
print("=" * 60)

# Categorize dialogues for selection
categories = {
    'manip_high_conf_correct': [],
    'manip_med_conf_correct': [],
    'manip_misclassified': [],
    'non_manip_high_conf_correct': [],
    'non_manip_med_conf_correct': [],
    'non_manip_misclassified': []
}

# Categorize each test dialogue
for i, (text, true_label, pred_label, prob) in enumerate(zip(test_texts, all_labels, all_preds, all_probs)):
    confidence = prob if pred_label == 1 else (1 - prob)
    correct = (true_label == pred_label)

    if true_label == 1:  # Manipulative
        if correct and confidence > 0.8:
            categories['manip_high_conf_correct'].append((i, text, prob, confidence))
        elif correct and 0.6 <= confidence <= 0.8:
            categories['manip_med_conf_correct'].append((i, text, prob, confidence))
        elif not correct:
            categories['manip_misclassified'].append((i, text, prob, confidence))
    else:  # Non-manipulative
        if correct and confidence > 0.8:
            categories['non_manip_high_conf_correct'].append((i, text, prob, confidence))
        elif correct and 0.6 <= confidence <= 0.8:
            categories['non_manip_med_conf_correct'].append((i, text, prob, confidence))
        elif not correct:
            categories['non_manip_misclassified'].append((i, text, prob, confidence))


# Print candidates for each category
def print_category(cat_name, items, max_show=10):
    print(f"\n--- {cat_name.upper().replace('_', ' ')} ({len(items)} total) ---")
    # Sort by confidence (descending for correct, ascending for misclassified)
    if 'misclassified' in cat_name:
        items_sorted = sorted(items, key=lambda x: abs(0.5 - x[2]))  # Closest to 0.5 = most uncertain
    else:
        items_sorted = sorted(items, key=lambda x: x[3], reverse=True)  # Highest confidence first

    for i, (idx, text, prob, conf) in enumerate(items_sorted[:max_show]):
        print(f"{i + 1:2d}. [ID:{idx:3d}] Conf:{conf:.3f} Prob:{prob:.3f}")
        print(f"    \"{text[:100]}{'...' if len(text) > 100 else ''}\"")
        print()


# Print all categories
print_category("Manipulative High Confidence Correct", categories['manip_high_conf_correct'])
print_category("Manipulative Medium Confidence Correct", categories['manip_med_conf_correct'])
print_category("Manipulative Misclassified", categories['manip_misclassified'])
print_category("Non-Manipulative High Confidence Correct", categories['non_manip_high_conf_correct'])
print_category("Non-Manipulative Medium Confidence Correct", categories['non_manip_med_conf_correct'])
print_category("Non-Manipulative Misclassified", categories['non_manip_misclassified'])

print("\n" + "=" * 60)
print("SELECTION RECOMMENDATION:")
print("Pick 2 from each manipulative category (6 total)")
print("Pick 2 from each non-manipulative category (6 total)")
print("This gives you 12 diverse dialogues covering all model behaviors")
print("=" * 60)

print("\n===== DIALOGUE-LEVEL EXPERIMENT TRACKING COMPLETE =====")
print(f"You can view this experiment (ID: {experiment_id}) using the experiment tracker's functions.")