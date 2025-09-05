import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/sentence_level_training/dialogue_models/sentence_model_exp"


class FocusedRawAttentionExplainer:
    """
    Raw Attention implementation focused on selected sentences for comparison.
    """

    def __init__(self, model, tokenizer, device=None, max_length=64):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

    def explain(self, text: str, layer_idx: int = -1) -> dict:
        """Extract raw attention weights for a given text."""

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

        # Get model prediction and attention weights
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True  # Get attention weights
            )

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            manipulation_prob = probabilities[0, 1].item()  # Probability of manipulative class

        # Extract attention weights from specified layer
        attention_weights = outputs.attentions[layer_idx][0]  # Remove batch dimension

        # Average across all attention heads
        attention_matrix = attention_weights.mean(dim=0)

        # Get tokens and valid length
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())
        valid_length = attention_mask.sum().item()
        tokens = tokens[:valid_length]
        attention_matrix = attention_matrix[:valid_length, :valid_length]

        # CLS token attention (how much CLS attends to each token)
        cls_attention = attention_matrix[0, :].cpu().numpy()

        # Clean tokens for display
        clean_tokens = []
        for token in tokens:
            clean_token = token.replace('Ġ', ' ').replace('▁', ' ')
            if clean_token in ['<s>', '</s>', '<pad>']:
                clean_token = f'[{clean_token.replace("<", "").replace(">", "")}]'
            clean_tokens.append(clean_token)

        return {
            'text': text,
            'tokens': clean_tokens,
            'predicted_class': predicted_class,
            'manipulation_prob': manipulation_prob,
            'cls_attention': cls_attention,
            'valid_length': valid_length,
            'layer_used': layer_idx
        }


def load_model(model_path: str):
    """Load the trained model for attention analysis."""

    # Determine device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            num_labels=2,
            id2label={0: "Non-Manipulative", 1: "Manipulative"},
            label2id={"Non-Manipulative": 0, "Manipulative": 1}
        )
        print(f"✓ Model loaded from: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

    model.to(device)
    model.eval()
    return model, tokenizer, device


def focused_attention_analysis(sentences, model_path):
    """
    Raw Attention analysis focused on sampled sentences with clean comparison output.
    """
    print("=== FOCUSED RAW ATTENTION ANALYSIS ===")
    print("Loading model...")

    # Load model
    model, tokenizer, device = load_model(model_path)

    # Initialize explainer
    explainer = FocusedRawAttentionExplainer(model, tokenizer, device, max_length=64)

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
            # Get attention explanation
            explanation = explainer.explain(text, layer_idx=-1)  # Use last layer

            predicted_label = explanation['predicted_class']
            manipulation_prob = explanation['manipulation_prob']
            cls_attention = explanation['cls_attention']
            tokens = explanation['tokens']

            # Calculate summary statistics
            total_attention = np.sum(cls_attention)
            max_attention = np.max(cls_attention)

            # Find top attention tokens
            token_attention_pairs = list(zip(tokens, cls_attention))
            token_attention_pairs.sort(key=lambda x: x[1], reverse=True)

            # Get top 3 tokens (excluding special tokens)
            top_tokens = []
            for token, attention in token_attention_pairs:
                if not token.startswith('[') and len(top_tokens) < 3:
                    top_tokens.append((token, attention))

            # Print results
            prediction_status = "✓ CORRECT" if true_label == predicted_label else "✗ INCORRECT"
            print(f"Prediction: {prediction_status}")
            print(f"True Label: {true_label} | Predicted: {predicted_label} | Prob: {manipulation_prob:.3f}")
            print(f"Max Attention: {max_attention:.4f} | Total: {total_attention:.4f}")

            if top_tokens:
                print(f"Top Attention: {', '.join([f'{token}({att:.3f})' for token, att in top_tokens])}")

            print()  # Blank line for readability

            # Store detailed results
            sentence_summaries.append({
                'sentence_id': sentence_id,
                'category': category,
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'manipulation_prob': manipulation_prob,
                'max_attention': max_attention,
                'total_attention': total_attention,
                'top_attention_tokens': [f"{token}({att:.3f})" for token, att in top_tokens],
                'all_tokens': tokens,
                'all_attention_values': cls_attention.tolist()
            })

            # Store token-level results
            for token, attention_val in zip(tokens, cls_attention):
                if not token.startswith('['):  # Skip special tokens
                    results.append({
                        'sentence_id': sentence_id,
                        'category': category,
                        'token': token,
                        'attention_value': attention_val,
                        'text': text,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'manipulation_prob': manipulation_prob
                    })

        except Exception as e:
            print(f"✗ Error processing {sentence_id}: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(sentence_summaries)


def print_attention_comparison_summary(sentence_summaries):
    """
    Print a clean comparison summary across all sentences for attention analysis.
    """
    print("=== RAW ATTENTION COMPARISON SUMMARY ===")

    # Group by category
    categories = sentence_summaries['category'].unique()

    for category in sorted(categories):
        category_data = sentence_summaries[sentence_summaries['category'] == category]
        print(f"\n--- {category} ---")

        for _, row in category_data.iterrows():
            prediction_icon = "✓" if row['true_label'] == row['predicted_label'] else "✗"
            print(
                f"{row['sentence_id']}: {prediction_icon} Prob:{row['manipulation_prob']:.3f} MaxAtt:{row['max_attention']:.4f}")
            print(f"  Top: {', '.join(row['top_attention_tokens'][:2])}")

    print("\n=== KEY INSIGHTS ===")

    # Calculate summary statistics
    correct_predictions = sentence_summaries[sentence_summaries['true_label'] == sentence_summaries['predicted_label']]
    incorrect_predictions = sentence_summaries[
        sentence_summaries['true_label'] != sentence_summaries['predicted_label']]

    print(f"Correct Predictions: {len(correct_predictions)}/{len(sentence_summaries)}")
    print(f"Average Max Attention (Correct): {correct_predictions['max_attention'].mean():.4f}")
    if len(incorrect_predictions) > 0:
        print(f"Average Max Attention (Incorrect): {incorrect_predictions['max_attention'].mean():.4f}")

    # Most frequently high-attention tokens
    print(f"\nMost Common High-Attention Tokens:")
    all_token_data = []
    for _, row in sentence_summaries.iterrows():
        for token_info in row['top_attention_tokens'][:2]:  # Top 2 tokens
            if '(' in token_info:
                token = token_info.split('(')[0]
                value = float(token_info.split('(')[1].rstrip(')'))
                all_token_data.append({'token': token, 'value': value})

    if all_token_data:
        token_df = pd.DataFrame(all_token_data)
        top_tokens = token_df.groupby('token')['value'].agg(['mean', 'count']).reset_index()
        top_tokens = top_tokens[top_tokens['count'] >= 2].sort_values('mean', ascending=False)

        print("Token (avg_attention, frequency):")
        for _, row in top_tokens.head(8).iterrows():
            print(f"  {row['token']}: {row['mean']:.3f} ({row['count']}x)")


def main():
    """
    Main execution function for Raw Attention analysis with dataset sampling.
    """
    print("RAW ATTENTION ANALYSIS WITH DATASET SAMPLING")
    print("=" * 44)

    # Load pre-sampled sentences for perfect consistency with other methods
    print("Loading pre-sampled sentences for consistency with other explainability methods...")

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

    # Run Raw Attention analysis
    print(f"\nRunning Raw Attention analysis on {len(selected_sentences)} sentences...")
    results_df, summaries_df = focused_attention_analysis(selected_sentences, model_path)

    if results_df is not None and len(results_df) > 0:
        # Print comparison summary
        print_attention_comparison_summary(summaries_df)

        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        token_results_file = f'raw_attention_token_results_extended.csv'
        summaries_file = f'raw_attention_sentence_summaries_extended.csv'

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