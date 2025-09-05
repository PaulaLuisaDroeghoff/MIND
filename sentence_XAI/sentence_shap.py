from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
import shap
import re
import warnings

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/sentence_level_training/dialogue_models/sentence_model_exp"
# ADD YOUR EXCEL FILE PATH HERE
excel_file_path = "/data/mentalmanip_detailed_sentencelevel.xlsx"  # Update this path
excel_sheet_name = "Sheet1"  # Update if your data is in a different sheet

# ===== SAMPLING CONFIGURATION =====
SAMPLE_SIZE = 50  # Number of sentences to analyze (adjust as needed)
STRATIFIED_SAMPLING = True  # Whether to ensure balanced sampling across labels
RANDOM_SEED = 42  # For reproducible results - SAME AS EXPECTED GRADIENTS


def load_dataset_from_excel(excel_path, sheet_name="Sheet1"):
    """
    Load and prepare the dataset from Excel file for sampling.
    """
    try:
        # Load Excel file
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        print(f"Excel file loaded with columns: {df.columns.tolist()}")
        print(f"Total rows: {len(df)}")

        # Based on your Excel structure, use specific column names
        text_column = "Sentence"
        label_column = "Manipulative"

        # Check if the required columns exist
        if text_column not in df.columns:
            print(f"Error: '{text_column}' column not found!")
            print("Available columns:", df.columns.tolist())
            return None

        if label_column not in df.columns:
            print(f"Error: '{label_column}' column not found!")
            print("Available columns:", df.columns.tolist())
            return None

        print(f"Using text column: '{text_column}'")
        print(f"Using label column: '{label_column}'")

        # Clean the dataset
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].astype(str).str.strip() != '']

        # Standardize column names for the rest of the code
        df = df.rename(columns={text_column: 'text', label_column: 'label'})

        # Ensure labels are numeric 0/1
        df['label'] = pd.to_numeric(df['label'], errors='coerce')
        df = df.dropna(subset=['label'])  # Remove any rows where label conversion failed
        df['label'] = df['label'].astype(int)

        unique_labels = sorted(df['label'].unique())
        print(f"Unique labels found: {unique_labels}")

        # Verify we have 0/1 labels
        if set(unique_labels) != {0, 1}:
            print(f"Warning: Expected labels 0 and 1, but found: {unique_labels}")
            print("Filtering to only include 0 and 1 labels...")
            df = df[df['label'].isin([0, 1])]

        print(f"Dataset cleaned: {len(df)} sentences")
        label_counts = df['label'].value_counts().sort_index()
        print(f"Label distribution:")
        print(f"  Non-Manipulative (0): {label_counts.get(0, 0)}")
        print(f"  Manipulative (1): {label_counts.get(1, 0)}")

        return df

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print("Please check the file path and sheet name.")
        return None


def sample_sentences_from_dataset(df, sample_size=50, stratified=True, random_seed=42):
    """
    Sample sentences from the dataset for analysis.
    IMPORTANT: Uses same random seed as Expected Gradients for identical sampling.
    """
    np.random.seed(random_seed)

    if stratified:
        # Stratified sampling to ensure balanced representation
        sampled_sentences = []

        for label in df['label'].unique():
            label_data = df[df['label'] == label]
            n_samples = min(sample_size // 2, len(label_data))  # Half samples per label

            sampled_label_data = label_data.sample(n=n_samples, random_state=random_seed)
            sampled_sentences.append(sampled_label_data)

        sampled_df = pd.concat(sampled_sentences, ignore_index=True)

    else:
        # Random sampling
        sampled_df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)

    # Convert to the format expected by the analysis function
    selected_sentences = []
    for idx, row in sampled_df.iterrows():
        # Create category based on label for analysis
        category = "Manipulative" if row['label'] == 1 else "Non-Manipulative"

        sentence_data = {
            'text': row['text'],
            'true_label': row['label'],
            'category': category,
            'sentence_id': f"S{idx}"
        }
        selected_sentences.append(sentence_data)

    print(f"Sampled {len(selected_sentences)} sentences:")
    sampled_counts = sampled_df['label'].value_counts()
    print(f"  - Non-Manipulative (0): {sampled_counts.get(0, 0)}")
    print(f"  - Manipulative (1): {sampled_counts.get(1, 0)}")

    return selected_sentences


def focused_shap_analysis(sentences, model_path):
    """
    shap.py analysis focused on sampled sentences with clean comparison output
    """
    print("=== FOCUSED shap.py ANALYSIS ===")
    print("Loading model...")

    try:
        # Load your trained model as a pipeline
        pipe = pipeline(
            "text-classification",
            model=model_path,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

    # Initialize shap.py
    masker = shap.maskers.Text(tokenizer=r'\b\w+\b')
    explainer = shap.Explainer(pipe, masker)

    results = []
    sentence_summaries = []

    print(f"\nAnalyzing {len(sentences)} sampled sentences...\n")

    for i, sentence_data in enumerate(sentences):
        text = sentence_data['text']
        true_label = sentence_data['true_label']
        category = sentence_data['category']
        sentence_id = sentence_data['sentence_id']

        print(f"--- {sentence_id} ({i + 1}/{len(sentences)}): {category} ---")
        print(f"Text: {text}")

        try:
            # Get shap.py values
            shap_values = explainer([text])

            # Get model prediction
            prediction = pipe(text, return_all_scores=True)

            # Handle prediction format
            if isinstance(prediction[0], dict):
                pred_probs = {item['label']: item['score'] for item in prediction}
            else:
                pred_probs = {item['label']: item['score'] for item in prediction[0]}

            # Extract manipulation probability
            if 'Manipulative' in pred_probs:
                manipulation_prob = pred_probs['Manipulative']
                predicted_label = 1 if manipulation_prob > 0.5 else 0
            elif 'LABEL_1' in pred_probs:
                manipulation_prob = pred_probs['LABEL_1']
                predicted_label = 1 if manipulation_prob > 0.5 else 0
            else:
                # Fallback - assume second label is manipulative
                prob_values = list(pred_probs.values())
                manipulation_prob = prob_values[1] if len(prob_values) > 1 else prob_values[0]
                predicted_label = 1 if manipulation_prob > 0.5 else 0

            # Extract shap.py values for manipulative class (LABEL_1/class 1)
            # shap.py values shape: [1, num_tokens, num_classes]
            manip_shap_values = shap_values[:, :, 1].values[0]  # Focus on manipulative class

            # Tokenize text
            tokens = re.findall(r'\w+', text)

            # Ensure we have matching lengths
            min_length = min(len(tokens), len(manip_shap_values))
            tokens = tokens[:min_length]
            manip_shap_values = manip_shap_values[:min_length]

            # Calculate summary statistics
            total_positive = sum([val for val in manip_shap_values if val > 0])
            total_negative = sum([val for val in manip_shap_values if val < 0])
            net_contribution = sum(manip_shap_values)

            # Find top tokens
            token_contributions = list(zip(tokens, manip_shap_values))
            token_contributions.sort(key=lambda x: x[1], reverse=True)  # Sort by shap.py value

            top_positive = [t for t in token_contributions if t[1] > 0][:3]
            top_negative = [t for t in token_contributions if t[1] < 0][:3]

            # Print results
            prediction_status = "✓ CORRECT" if true_label == predicted_label else "✗ INCORRECT"
            print(f"Prediction: {prediction_status}")
            print(f"True Label: {true_label} | Predicted: {predicted_label} | Prob: {manipulation_prob:.3f}")
            print(f"Net shap.py: {net_contribution:.4f} (Pos: {total_positive:.4f}, Neg: {total_negative:.4f})")

            if top_positive:
                print(f"Top Positive: {', '.join([f'{token}({val:.3f})' for token, val in top_positive])}")
            if top_negative:
                print(f"Top Negative: {', '.join([f'{token}({val:.3f})' for token, val in top_negative])}")

            print()  # Blank line for readability

            # Store detailed results
            sentence_summaries.append({
                'sentence_id': sentence_id,
                'category': category,
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'manipulation_prob': manipulation_prob,
                'net_shap': net_contribution,
                'positive_contrib': total_positive,
                'negative_contrib': total_negative,
                'top_positive_tokens': [f"{token}({val:.3f})" for token, val in top_positive],
                'top_negative_tokens': [f"{token}({val:.3f})" for token, val in top_negative],
                'all_tokens': tokens,
                'all_shap_values': manip_shap_values.tolist()
            })

            # Store token-level results
            for token, shap_val in zip(tokens, manip_shap_values):
                results.append({
                    'sentence_id': sentence_id,
                    'category': category,
                    'token': token,
                    'shap_value': shap_val,
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'manipulation_prob': manipulation_prob
                })

        except Exception as e:
            print(f"✗ Error processing {sentence_id}: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(sentence_summaries)


def print_comparison_summary(sentence_summaries):
    """
    Print a clean comparison summary across all sentences
    """
    print("=== shap.py COMPARISON SUMMARY ===")

    # Group by category
    categories = sentence_summaries['category'].unique()

    for category in sorted(categories):
        category_data = sentence_summaries[sentence_summaries['category'] == category]
        print(f"\n--- {category} ---")

        for _, row in category_data.iterrows():
            prediction_icon = "✓" if row['true_label'] == row['predicted_label'] else "✗"
            print(
                f"{row['sentence_id']}: {prediction_icon} Prob:{row['manipulation_prob']:.3f} NetSHAP:{row['net_shap']:.4f}")
            print(f"  Top+: {', '.join(row['top_positive_tokens'][:2])}")
            if row['top_negative_tokens']:
                print(f"  Top-: {', '.join(row['top_negative_tokens'][:2])}")

    print("\n=== KEY INSIGHTS ===")

    # Calculate some summary statistics
    correct_predictions = sentence_summaries[sentence_summaries['true_label'] == sentence_summaries['predicted_label']]
    incorrect_predictions = sentence_summaries[
        sentence_summaries['true_label'] != sentence_summaries['predicted_label']]

    print(f"Correct Predictions: {len(correct_predictions)}/{len(sentence_summaries)}")
    print(f"Average Net shap.py (Correct): {correct_predictions['net_shap'].mean():.4f}")
    if len(incorrect_predictions) > 0:
        print(f"Average Net shap.py (Incorrect): {incorrect_predictions['net_shap'].mean():.4f}")

    # Most important tokens across all sentences
    print(f"\nMost Common High-Impact Tokens:")
    all_token_data = []
    for _, row in sentence_summaries.iterrows():
        for token_info in row['top_positive_tokens'][:2]:  # Top 2 positive tokens
            if '(' in token_info:
                token = token_info.split('(')[0]
                value = float(token_info.split('(')[1].rstrip(')'))
                all_token_data.append({'token': token, 'value': value, 'type': 'positive'})

    if all_token_data:
        token_df = pd.DataFrame(all_token_data)
        top_tokens = token_df.groupby('token')['value'].agg(['mean', 'count']).reset_index()
        top_tokens = top_tokens[top_tokens['count'] >= 2].sort_values('mean', ascending=False)

        print("Token (avg_impact, frequency):")
        for _, row in top_tokens.head(8).iterrows():
            print(f"  {row['token']}: {row['mean']:.3f} ({row['count']}x)")


def main():
    """
    Main execution function for shap.py analysis with dataset sampling.
    """
    print("shap.py ANALYSIS WITH DATASET SAMPLING")
    print("=" * 40)

    # Load pre-sampled sentences for perfect consistency with Expected Gradients
    print("Loading pre-sampled sentences for consistency with Expected Gradients...")

    try:
        sampled_df = pd.read_csv('sampled_sentences.csv')
        selected_sentences = sampled_df.to_dict('records')

        print(f"✓ Loaded {len(selected_sentences)} pre-sampled sentences")

        # Show distribution for verification
        label_counts = sampled_df['true_label'].value_counts()
        print(f"Sentence distribution:")
        print(f"  - Non-Manipulative (0): {label_counts.get(0, 0)}")
        print(f"  - Manipulative (1): {label_counts.get(1, 0)}")

    except FileNotFoundError:
        print("'sampled_sentences.csv' not found!")
        print("Please run the Expected Gradients script first to generate the sampled sentences.")
        return
    except Exception as e:
        print(f"Error loading sampled sentences: {e}")
        return

    # Run shap.py analysis
    print(f"\nRunning shap.py analysis on {len(selected_sentences)} sentences...")
    results_df, summaries_df = focused_shap_analysis(selected_sentences, model_path)

    if results_df is not None and len(results_df) > 0:
        # Print comparison summary
        print_comparison_summary(summaries_df)

        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        token_results_file = f'shap_token_results_extended.csv'
        summaries_file = f'shap_sentence_summaries_extended.csv'

        results_df.to_csv(token_results_file, index=False)
        summaries_df.to_csv(summaries_file, index=False)

        print(f"\n=== FILES SAVED ===")
        print(f"- {token_results_file} (detailed token-level)")
        print(f"- {summaries_file} (sentence-level summaries)")
        print(f"\nTotal tokens analyzed: {len(results_df)}")
        print(f"Sentences processed: {len(summaries_df)}")

    else:
        print("Analysis failed - no results generated")


if __name__ == "__main__":
    main()