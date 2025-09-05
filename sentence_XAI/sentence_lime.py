from transformers import pipeline
import numpy as np
import pandas as pd
import torch
from lime.lime_text import LimeTextExplainer
import re
import warnings

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/sentence_level_training/dialogue_models/sentence_model_exp"


def custom_tokenizer(text):
    """Same tokenizer as original LIME code"""
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens


def focused_lime_analysis(sentences, model_path):
    """
    LIME analysis focused on sampled sentences with clean comparison output
    """
    print("=== FOCUSED LIME ANALYSIS ===")
    print("Loading model...")

    try:
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

    def predict_proba(texts):
        """Prediction function for LIME - adapted for your model"""
        preds = pipe(texts, return_all_scores=True)
        probabilities = np.array([[pred['score'] for pred in preds_single] for preds_single in preds])
        return probabilities

    # Initialize LIME explainer
    explainer = LimeTextExplainer(
        class_names=['LABEL_0', 'LABEL_1'],
        split_expression=lambda x: custom_tokenizer(x)
    )

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
            # Tokenize for LIME
            tokens = custom_tokenizer(text)

            # Generate LIME explanation
            exp = explainer.explain_instance(
                text,
                predict_proba,
                num_features=len(tokens),
                num_samples=100
            )

            # Get model prediction
            prediction = pipe(text, return_all_scores=True)

            # Handle different prediction formats
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
                # Fallback
                sorted_preds = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)
                predicted_label = 1 if 'manip' in sorted_preds[0][0].lower() else 0
                manipulation_prob = sorted_preds[0][1] if predicted_label == 1 else sorted_preds[1][1]

            # Focus on manipulative class (label=1)
            explanation_list = exp.as_list(label=1)
            token_value_dict = {token: value for token, value in explanation_list}

            # Calculate sentence-level statistics
            lime_values = [token_value_dict.get(token, 0) for token in tokens]
            total_positive_contrib = sum([val for val in lime_values if val > 0])
            total_negative_contrib = sum([val for val in lime_values if val < 0])
            net_contribution = sum(lime_values)

            # Find top contributing tokens
            token_contributions = [(token, token_value_dict.get(token, 0)) for token in tokens]
            token_contributions.sort(key=lambda x: x[1], reverse=True)  # Sort by LIME value

            top_positive = [t for t in token_contributions if t[1] > 0][:3]
            top_negative = [t for t in token_contributions if t[1] < 0][:3]

            # Print results
            prediction_status = "✓ CORRECT" if true_label == predicted_label else "✗ INCORRECT"
            print(f"Prediction: {prediction_status}")
            print(f"True Label: {true_label} | Predicted: {predicted_label} | Prob: {manipulation_prob:.3f}")
            print(
                f"Net LIME: {net_contribution:.4f} (Pos: {total_positive_contrib:.4f}, Neg: {total_negative_contrib:.4f})")

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
                'net_lime': net_contribution,
                'positive_contrib': total_positive_contrib,
                'negative_contrib': total_negative_contrib,
                'top_positive_tokens': [f"{token}({val:.3f})" for token, val in top_positive],
                'top_negative_tokens': [f"{token}({val:.3f})" for token, val in top_negative],
                'all_tokens': tokens,
                'all_lime_values': lime_values
            })

            # Store token-level results
            for token in tokens:
                lime_value = token_value_dict.get(token, 0)
                results.append({
                    'sentence_id': sentence_id,
                    'category': category,
                    'token': token,
                    'lime_value': lime_value,
                    'text': text,
                    'true_label': true_label,
                    'predicted_label': predicted_label,
                    'manipulation_prob': manipulation_prob
                })

        except Exception as e:
            print(f"✗ Error processing {sentence_id}: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(sentence_summaries)


def print_lime_comparison_summary(sentence_summaries):
    """
    Print a clean comparison summary across all sentences for LIME analysis
    """
    print("=== LIME COMPARISON SUMMARY ===")

    # Group by category
    categories = sentence_summaries['category'].unique()

    for category in sorted(categories):
        category_data = sentence_summaries[sentence_summaries['category'] == category]
        print(f"\n--- {category} ---")

        for _, row in category_data.iterrows():
            prediction_icon = "✓" if row['true_label'] == row['predicted_label'] else "✗"
            print(
                f"{row['sentence_id']}: {prediction_icon} Prob:{row['manipulation_prob']:.3f} NetLIME:{row['net_lime']:.4f}")
            print(f"  Top+: {', '.join(row['top_positive_tokens'][:2])}")
            if row['top_negative_tokens']:
                print(f"  Top-: {', '.join(row['top_negative_tokens'][:2])}")

    print("\n=== KEY INSIGHTS ===")

    # Calculate summary statistics
    correct_predictions = sentence_summaries[sentence_summaries['true_label'] == sentence_summaries['predicted_label']]
    incorrect_predictions = sentence_summaries[
        sentence_summaries['true_label'] != sentence_summaries['predicted_label']]

    print(f"Correct Predictions: {len(correct_predictions)}/{len(sentence_summaries)}")
    print(f"Average Net LIME (Correct): {correct_predictions['net_lime'].mean():.4f}")
    if len(incorrect_predictions) > 0:
        print(f"Average Net LIME (Incorrect): {incorrect_predictions['net_lime'].mean():.4f}")

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
    Main execution function for LIME analysis with dataset sampling.
    """
    print("LIME ANALYSIS WITH DATASET SAMPLING")
    print("=" * 36)

    # Load pre-sampled sentences for perfect consistency with Expected Gradients and shap.py
    print("Loading pre-sampled sentences for consistency with Expected Gradients and shap.py...")

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

    # Run LIME analysis
    print(f"\nRunning LIME analysis on {len(selected_sentences)} sentences...")
    results_df, summaries_df = focused_lime_analysis(selected_sentences, model_path)

    if results_df is not None and len(results_df) > 0:
        # Print comparison summary
        print_lime_comparison_summary(summaries_df)

        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        token_results_file = f'lime_token_results_extended.csv'
        summaries_file = f'lime_sentence_summaries_extended.csv'

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