import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/sentence_level_training/dialogue_models/sentence_model_exp"


def load_model(model_path):
    """Load the trained model and tokenizer."""

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(device)
        model.eval()

        print(f"✓ Model loaded from: {model_path}")
        return model, tokenizer, device

    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise


def analyze_with_token_occlusion(sentence, model, tokenizer, device):
    """
    Analyze sentence by occluding each token and measuring impact on manipulation prediction.
    """

    # Tokenize input
    input_encoded = tokenizer(
        sentence,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=64
    )
    input_ids = input_encoded.input_ids.to(device)

    # Get baseline prediction
    with torch.no_grad():
        baseline_outputs = model(**input_encoded.to(device))
        baseline_pred = torch.softmax(baseline_outputs.logits, dim=-1)[0, 1].item()

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    results = []

    # Use mask token for occlusion
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id else tokenizer.unk_token_id

    for i, token in enumerate(tokens):
        if token not in ['<s>', '</s>', '<pad>', tokenizer.pad_token]:
            # Create modified input with this token masked
            modified_ids = input_ids.clone()
            modified_ids[0, i] = mask_id

            # Get prediction with masked token
            with torch.no_grad():
                modified_encoded = {
                    'input_ids': modified_ids,
                    'attention_mask': (modified_ids != tokenizer.pad_token_id).long()
                }
                outputs = model(**modified_encoded)
                masked_pred = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            # Calculate importance as difference (how much manipulation score changes when token is removed)
            importance = baseline_pred - masked_pred

            # Clean token representation
            clean_token = token.replace('Ġ', '').strip()
            if clean_token:
                results.append({
                    'token': clean_token,
                    'position': i,
                    'occlusion_impact': importance,
                    'baseline_pred': baseline_pred,
                    'masked_pred': masked_pred
                })

    return results, baseline_pred


def focused_token_occlusion_analysis(sentences, model_path):
    """
    Token Occlusion analysis focused on sampled sentences with clean comparison output.
    """
    print("=== FOCUSED TOKEN OCCLUSION ANALYSIS ===")
    print("Loading model...")

    # Load model
    model, tokenizer, device = load_model(model_path)

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
            # Get token occlusion analysis
            token_results, baseline_pred = analyze_with_token_occlusion(text, model, tokenizer, device)

            predicted_label = 1 if baseline_pred > 0.5 else 0

            # Calculate summary statistics
            if token_results:
                occlusion_values = [result['occlusion_impact'] for result in token_results]
                max_impact = max(occlusion_values) if occlusion_values else 0
                min_impact = min(occlusion_values) if occlusion_values else 0
                total_positive_impact = sum([val for val in occlusion_values if val > 0])
                total_negative_impact = sum([val for val in occlusion_values if val < 0])

                # Find top contributing tokens
                token_results.sort(key=lambda x: x['occlusion_impact'], reverse=True)
                top_positive = [t for t in token_results if t['occlusion_impact'] > 0][:3]
                top_negative = [t for t in token_results if t['occlusion_impact'] < 0][:3]
            else:
                max_impact = min_impact = total_positive_impact = total_negative_impact = 0
                top_positive = top_negative = []

            # Print results
            prediction_status = "✓ CORRECT" if true_label == predicted_label else "✗ INCORRECT"
            print(f"Prediction: {prediction_status}")
            print(f"True Label: {true_label} | Predicted: {predicted_label} | Prob: {baseline_pred:.3f}")
            print(f"Max Impact: {max_impact:.4f} | Min Impact: {min_impact:.4f}")

            if top_positive:
                pos_tokens_str = ', '.join([f"{t['token']}({t['occlusion_impact']:.3f})" for t in top_positive])
                print(f"Top Positive: {pos_tokens_str}")
            if top_negative:
                neg_tokens_str = ', '.join([f"{t['token']}({t['occlusion_impact']:.3f})" for t in top_negative])
                print(f"Top Negative: {neg_tokens_str}")

            print()  # Blank line for readability

            # Store detailed results
            sentence_summaries.append({
                'sentence_id': sentence_id,
                'category': category,
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'manipulation_prob': baseline_pred,
                'max_impact': max_impact,
                'min_impact': min_impact,
                'total_positive_impact': total_positive_impact,
                'total_negative_impact': total_negative_impact,
                'top_positive_tokens': [f"{t['token']}({t['occlusion_impact']:.3f})" for t in top_positive],
                'top_negative_tokens': [f"{t['token']}({t['occlusion_impact']:.3f})" for t in top_negative],
                'num_tokens_analyzed': len(token_results)
            })

            # Store token-level results
            for token_result in token_results:
                results.append({
                    'sentence_id': sentence_id,
                    'category': category,
                    'token': token_result['token'],
                    'occlusion_impact': token_result['occlusion_impact'],
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'manipulation_prob': baseline_pred
                })

        except Exception as e:
            print(f"✗ Error processing {sentence_id}: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(sentence_summaries)


def print_occlusion_comparison_summary(sentence_summaries):
    """
    Print a clean comparison summary across all sentences for token occlusion analysis.
    """
    print("=== TOKEN OCCLUSION COMPARISON SUMMARY ===")

    # Group by category
    categories = sentence_summaries['category'].unique()

    for category in sorted(categories):
        category_data = sentence_summaries[sentence_summaries['category'] == category]
        print(f"\n--- {category} ---")

        for _, row in category_data.iterrows():
            prediction_icon = "✓" if row['true_label'] == row['predicted_label'] else "✗"
            print(
                f"{row['sentence_id']}: {prediction_icon} Prob:{row['manipulation_prob']:.3f} MaxImpact:{row['max_impact']:.4f}")
            print(f"  Top+: {', '.join(row['top_positive_tokens'][:2])}")
            if row['top_negative_tokens']:
                print(f"  Top-: {', '.join(row['top_negative_tokens'][:2])}")

    print("\n=== KEY INSIGHTS ===")

    # Calculate summary statistics
    correct_predictions = sentence_summaries[sentence_summaries['true_label'] == sentence_summaries['predicted_label']]
    incorrect_predictions = sentence_summaries[
        sentence_summaries['true_label'] != sentence_summaries['predicted_label']]

    print(f"Correct Predictions: {len(correct_predictions)}/{len(sentence_summaries)}")
    print(f"Average Max Impact (Correct): {correct_predictions['max_impact'].mean():.4f}")
    if len(incorrect_predictions) > 0:
        print(f"Average Max Impact (Incorrect): {incorrect_predictions['max_impact'].mean():.4f}")

    # Most important tokens by occlusion impact
    print(f"\nMost Common High-Impact Tokens:")
    all_token_data = []
    for _, row in sentence_summaries.iterrows():
        for token_info in row['top_positive_tokens'][:2]:  # Top 2 positive impact tokens
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
    Main execution function for Token Occlusion analysis with dataset sampling.
    """
    print("TOKEN OCCLUSION ANALYSIS WITH DATASET SAMPLING")
    print("=" * 47)

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

    # Run Token Occlusion analysis
    print(f"\nRunning Token Occlusion analysis on {len(selected_sentences)} sentences...")
    results_df, summaries_df = focused_token_occlusion_analysis(selected_sentences, model_path)

    if results_df is not None and len(results_df) > 0:
        # Print comparison summary
        print_occlusion_comparison_summary(summaries_df)

        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        token_results_file = f'token_occlusion_token_results_extended.csv'
        summaries_file = f'token_occlusion_sentence_summaries_extended.csv'

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