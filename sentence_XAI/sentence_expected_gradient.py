import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/sentence_level_training/dialogue_models/sentence_model_exp"
# ADD YOUR EXCEL FILE PATH HERE
excel_file_path = "/data/mentalmanip_detailed_sentencelevel.xlsx"  # Update this path
excel_sheet_name = "Sheet1"  # Update if your data is in a different sheet

# ===== SAMPLING CONFIGURATION =====
SAMPLE_SIZE = 100  # Number of sentences to analyze (adjust as needed)
STRATIFIED_SAMPLING = True  # Whether to ensure balanced sampling across labels
RANDOM_SEED = 42  # For reproducible results

# ===== REFERENCE TEXTS FOR BASELINES =====
reference_texts = [
    "Kelly, get to work on those new uniforms. I'm not running a summer camp here.",
    "Look, you can't stay here. Just take your champagne and go.",
    "Are you going to leave, Mr. Kirkeby, or do I have to throw you out?",
    "Your capacity for self-delusion is remarkable, Dr. Singer.",
    "To go tonight, with me, as though we know nothing. I've already made my move.",
    "The old man will never bring in the New Year.",
    "Maybe she puts her work ahead of her personal appearance.",
    "Pain cuts the deepest, isn't that what you said?",
    "What's goin' on?",
    "I could always get us a bottle.",
    "No. I'll take you home.",
    "Tell 'em your father gave it to you.",
    "The weather is nice today.",
    "This is just a factual statement.",
    "Please review the attached document."
]


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


class FocusedExpectedGradientsExplainer:
    """
    Expected Gradients implementation focused on selected sentences for comparison.
    """

    def __init__(self, model, tokenizer, device=None, max_length=64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(True)

    def _prepare_reference_samples(self, reference_texts, n_samples=20):
        """Prepare reference samples for Expected Gradients."""
        # Sample n_samples from reference texts
        if len(reference_texts) > n_samples:
            selected_texts = np.random.choice(reference_texts, n_samples, replace=False)
        else:
            selected_texts = reference_texts

        # Tokenize reference texts
        reference_inputs = self.tokenizer(
            list(selected_texts),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Move to device
        reference_inputs = {k: v.to(self.device) for k, v in reference_inputs.items()}

        # Get embeddings for reference samples
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(reference_inputs['input_ids'])

        return embeddings, reference_inputs['attention_mask']

    def _get_gradients(self, input_embeds, attention_mask, target_class):
        """Compute gradients of the model output with respect to input embeddings."""
        # Clear any existing gradients
        self.model.zero_grad()

        # Create a fresh tensor that requires gradients
        input_embeds = input_embeds.clone().detach().requires_grad_(True)

        # Forward pass with embeddings
        outputs = self.model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask
        )

        # Get prediction for target class
        target_logit = outputs.logits[0, target_class]

        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=target_logit,
            inputs=input_embeds,
            retain_graph=False,
            create_graph=False,
            only_inputs=True
        )[0]

        return gradients

    def explain(self, text, reference_texts, target_class=None, n_steps=10, n_reference_samples=20):
        """Generate Expected Gradients explanation for a given text."""
        # Tokenize input text
        inputs = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = inputs['input_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            manipulation_prob = probabilities[0, 1].item()  # Probability of manipulative class

        # Use predicted class if target_class not specified
        if target_class is None:
            target_class = predicted_class

        # Get input embeddings
        input_embeds = self.model.get_input_embeddings()(input_ids)

        # Prepare reference samples
        reference_embeds, reference_masks = self._prepare_reference_samples(
            reference_texts, n_reference_samples
        )

        # Compute Expected Gradients
        total_gradients = torch.zeros_like(input_embeds)

        for ref_embeds, ref_mask in zip(reference_embeds, reference_masks):
            ref_embeds = ref_embeds.unsqueeze(0)  # Add batch dimension
            ref_mask = ref_mask.unsqueeze(0)

            # Compute path from reference to input
            for step in range(n_steps):
                alpha = step / (n_steps - 1) if n_steps > 1 else 1.0

                # Interpolate between reference and input
                interpolated_embeds = ref_embeds + alpha * (input_embeds - ref_embeds)
                interpolated_mask = attention_mask  # Use input's attention mask

                # Compute gradients at this point
                gradients = self._get_gradients(
                    interpolated_embeds, interpolated_mask, target_class
                )

                total_gradients += gradients

        # Average over all reference samples and steps
        avg_gradients = total_gradients / (n_reference_samples * n_steps)

        # Compute attributions (gradient * (input - average_reference))
        avg_reference = torch.mean(reference_embeds, dim=0, keepdim=True)
        attributions = avg_gradients * (input_embeds - avg_reference)

        # Sum attributions across embedding dimensions to get token-level attributions
        token_attributions = torch.sum(attributions, dim=-1).squeeze().detach().cpu().numpy()

        # Get tokens for analysis
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())

        # Filter valid tokens and attributions
        valid_tokens = []
        valid_attributions = []

        for token, attr, mask in zip(tokens, token_attributions, attention_mask.squeeze().cpu().numpy()):
            if mask == 1 and token not in ['<pad>', '</s>', '<s>']:
                clean_token = token.replace('Ġ', '').replace('▁', '').strip()
                if clean_token:
                    valid_tokens.append(clean_token)
                    valid_attributions.append(attr)

        return {
            'text': text,
            'tokens': valid_tokens,
            'expected_gradients': valid_attributions,
            'predicted_class': predicted_class,
            'manipulation_prob': manipulation_prob,
            'target_class': target_class
        }


def load_model(model_path):
    """Load the trained model and tokenizer."""

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "Non-Manipulative", 1: "Manipulative"},
            label2id={"Non-Manipulative": 0, "Manipulative": 1}
        )

        model.to(device)
        model.eval()

        print(f"✓ Model loaded from: {model_path}")
        return model, tokenizer, device

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def focused_expected_gradients_analysis(sentences, model_path):
    """
    Expected Gradients analysis focused on selected sentences with clean comparison output.
    """
    print("=== FOCUSED EXPECTED GRADIENTS ANALYSIS ===")
    print("Loading model...")

    # Load model
    model, tokenizer, device = load_model(model_path)

    # Initialize explainer
    explainer = FocusedExpectedGradientsExplainer(model, tokenizer, device, max_length=64)

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
            # Get Expected Gradients explanation
            explanation = explainer.explain(
                text,
                reference_texts,
                n_steps=10,
                n_reference_samples=15  # Reduced for efficiency
            )

            predicted_label = explanation['predicted_class']
            manipulation_prob = explanation['manipulation_prob']
            eg_values = explanation['expected_gradients']
            tokens = explanation['tokens']

            # Calculate summary statistics
            if eg_values:
                max_gradient = max(eg_values)
                min_gradient = min(eg_values)
                total_positive = sum([val for val in eg_values if val > 0])
                total_negative = sum([val for val in eg_values if val < 0])

                # Find top contributing tokens
                token_contributions = list(zip(tokens, eg_values))
                token_contributions.sort(key=lambda x: x[1], reverse=True)

                top_positive = [t for t in token_contributions if t[1] > 0][:3]
                top_negative = [t for t in token_contributions if t[1] < 0][:3]
            else:
                max_gradient = min_gradient = total_positive = total_negative = 0
                top_positive = top_negative = []

            # Print results
            prediction_status = "✓ CORRECT" if true_label == predicted_label else "✗ INCORRECT"
            print(f"Prediction: {prediction_status}")
            print(f"True Label: {true_label} | Predicted: {predicted_label} | Prob: {manipulation_prob:.3f}")
            print(f"Max ExpGrad: {max_gradient:.4f} | Min ExpGrad: {min_gradient:.4f}")

            if top_positive:
                pos_tokens_str = ', '.join([f"{token}({val:.3f})" for token, val in top_positive])
                print(f"Top Positive: {pos_tokens_str}")
            if top_negative:
                neg_tokens_str = ', '.join([f"{token}({val:.3f})" for token, val in top_negative])
                print(f"Top Negative: {neg_tokens_str}")

            print()  # Blank line for readability

            # Store detailed results
            sentence_summaries.append({
                'sentence_id': sentence_id,
                'category': category,
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'manipulation_prob': manipulation_prob,
                'max_expected_gradient': max_gradient,
                'min_expected_gradient': min_gradient,
                'total_positive': total_positive,
                'total_negative': total_negative,
                'top_positive_tokens': [f"{token}({val:.3f})" for token, val in top_positive],
                'top_negative_tokens': [f"{token}({val:.3f})" for token, val in top_negative],
                'all_tokens': tokens,
                'all_expected_gradients': eg_values
            })

            # Store token-level results
            for token, eg_value in zip(tokens, eg_values):
                results.append({
                    'sentence_id': sentence_id,
                    'category': category,
                    'token': token,
                    'expected_gradient_value': eg_value,
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'manipulation_prob': manipulation_prob
                })

        except Exception as e:
            print(f"✗ Error processing {sentence_id}: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(sentence_summaries)


def print_expected_gradients_comparison_summary(sentence_summaries):
    """
    Print a clean comparison summary across all sentences for Expected Gradients analysis.
    """
    print("=== EXPECTED GRADIENTS COMPARISON SUMMARY ===")

    # Group by category
    categories = sentence_summaries['category'].unique()

    for category in sorted(categories):
        category_data = sentence_summaries[sentence_summaries['category'] == category]
        print(f"\n--- {category} ---")

        for _, row in category_data.iterrows():
            prediction_icon = "✓" if row['true_label'] == row['predicted_label'] else "✗"
            print(
                f"{row['sentence_id']}: {prediction_icon} Prob:{row['manipulation_prob']:.3f} MaxExpGrad:{row['max_expected_gradient']:.4f}")
            print(f"  Top+: {', '.join(row['top_positive_tokens'][:2])}")
            if row['top_negative_tokens']:
                print(f"  Top-: {', '.join(row['top_negative_tokens'][:2])}")

    print("\n=== KEY INSIGHTS ===")

    # Calculate summary statistics
    correct_predictions = sentence_summaries[sentence_summaries['true_label'] == sentence_summaries['predicted_label']]
    incorrect_predictions = sentence_summaries[
        sentence_summaries['true_label'] != sentence_summaries['predicted_label']]

    print(f"Correct Predictions: {len(correct_predictions)}/{len(sentence_summaries)}")
    print(f"Average Max Expected Gradient (Correct): {correct_predictions['max_expected_gradient'].mean():.4f}")
    if len(incorrect_predictions) > 0:
        print(f"Average Max Expected Gradient (Incorrect): {incorrect_predictions['max_expected_gradient'].mean():.4f}")

    # Most important tokens by expected gradient magnitude
    print(f"\nMost Common High-Gradient Tokens:")
    all_token_data = []
    for _, row in sentence_summaries.iterrows():
        for token_info in row['top_positive_tokens'][:2]:  # Top 2 positive gradient tokens
            if '(' in token_info:
                token = token_info.split('(')[0]
                value = float(token_info.split('(')[1].rstrip(')'))
                all_token_data.append({'token': token, 'value': value, 'type': 'positive'})

    if all_token_data:
        token_df = pd.DataFrame(all_token_data)
        top_tokens = token_df.groupby('token')['value'].agg(['mean', 'count']).reset_index()
        top_tokens = top_tokens[top_tokens['count'] >= 2].sort_values('mean', ascending=False)

        print("Token (avg_expected_gradient, frequency):")
        for _, row in top_tokens.head(8).iterrows():
            print(f"  {row['token']}: {row['mean']:.3f} ({row['count']}x)")


def main():
    """
    Main execution function for focused Expected Gradients analysis with dataset sampling.
    """
    print("EXPECTED GRADIENTS ANALYSIS WITH DATASET SAMPLING")
    print("=" * 52)

    # Load dataset from Excel
    print("Loading dataset from Excel file...")
    df = load_dataset_from_excel(excel_file_path, excel_sheet_name)

    if df is None:
        print("Failed to load dataset. Please check the file path and column names.")
        return

    # Sample sentences from dataset
    print(f"\nSampling {SAMPLE_SIZE} sentences from dataset...")
    selected_sentences = sample_sentences_from_dataset(
        df,
        sample_size=SAMPLE_SIZE,
        stratified=STRATIFIED_SAMPLING,
        random_seed=RANDOM_SEED
    )

    # Save sampled sentences for consistency with shap.py analysis
    sampled_sentences_df = pd.DataFrame(selected_sentences)
    sampled_sentences_df.to_csv('sampled_sentences.csv', index=False)

    # Run Expected Gradients analysis
    print(f"\nRunning Expected Gradients analysis on {len(selected_sentences)} sentences...")
    results_df, summaries_df = focused_expected_gradients_analysis(selected_sentences, model_path)

    if results_df is not None and len(results_df) > 0:
        # Print comparison summary
        print_expected_gradients_comparison_summary(summaries_df)

        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        token_results_file = f'expected_gradients_token_results_extended.csv'
        summaries_file = f'expected_gradients_sentence_summaries_extended.csv'

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