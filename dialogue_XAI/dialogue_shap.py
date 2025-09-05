from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
import dialogue_shap
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/dialogue_level_training/dialogue_models/dialogue_model_exp"


# ===== SENTENCE BOUNDARY DETECTION =====
def get_sentence_boundaries(text, tokens):
    """
    Map token positions to sentence boundaries in the dialogue.
    """
    # Split dialogue into sentences using multiple delimiters
    sentences = re.split(r'[.!?]+\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_boundaries = []
    current_token_idx = 0

    for sentence in sentences:
        if not sentence:
            continue

        # Get tokens for this sentence using word boundaries
        sentence_tokens = re.findall(r'\b\w+\b', sentence)

        # Find matching tokens in the full token list
        start_idx = current_token_idx
        end_idx = start_idx

        # Match tokens from this sentence
        matched_tokens = 0
        while end_idx < len(tokens) and matched_tokens < len(sentence_tokens):
            if tokens[end_idx] not in ['<s>', '</s>', '<pad>']:
                if re.sub(r'[^\w]', '', tokens[end_idx].lower()) in [re.sub(r'[^\w]', '', t.lower()) for t in
                                                                     sentence_tokens]:
                    matched_tokens += 1
            end_idx += 1

        if matched_tokens > 0:  # Only add if we found matching tokens
            sentence_boundaries.append({
                'sentence': sentence,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'tokens': tokens[start_idx:end_idx]
            })
            current_token_idx = end_idx

    return sentence_boundaries


# ===== EXTENDED dialogue_shap.py ANALYSIS =====
def shap_analysis_extended_dialogues(model_path):
    """
    dialogue_shap.py analysis using the sampled dialogues with sentence-level aggregation
    """

    # Load pre-sampled dialogues for consistency with other methods
    print("Loading pre-sampled dialogues for consistency with other methods...")

    try:
        sampled_df = pd.read_csv('sampled_dialogues.csv')

        print(f"✓ Loaded {len(sampled_df)} pre-sampled dialogues")

        # Show distribution for verification
        label_counts = sampled_df['label'].value_counts()
        print(f"Dialogue distribution:")
        print(f"  - Non-Manipulative (0): {label_counts.get(0, 0)}")
        print(f"  - Manipulative (1): {label_counts.get(1, 0)}")

    except FileNotFoundError:
        print("'sampled_dialogues.csv' not found!")
        print("Please run the Extended Expected Gradients script first to generate the sampled dialogues.")
        return None, None, None
    except Exception as e:
        print(f"Error loading sampled dialogues: {e}")
        return None, None, None

    try:
        # Load model as pipeline
        pipe = pipeline(
            "text-classification",
            model=model_path,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ Model loaded successfully for dialogue_shap.py analysis")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    # Use the same text masker
    masker = dialogue_shap.maskers.Text(tokenizer=r'\b\w+\b')
    explainer = dialogue_shap.Explainer(pipe, masker)

    token_results = []
    sentence_results = []
    dialogue_summaries = []

    # Model outputs LABEL_0 (non-manipulative) and LABEL_1 (manipulative)
    class_names = ['LABEL_0', 'LABEL_1']

    print(f"\nAnalyzing {len(sampled_df)} dialogues with dialogue_shap.py...")

    for index, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing dialogues"):
        dialogue_id = row['dialogue_id']
        text_input = [row['text']]

        try:
            shap_values = explainer(text_input)

            # Focus on manipulative class (LABEL_1)
            label_index = class_names.index("LABEL_1")
            specific_shap_values = shap_values[:, :, label_index].values

            # Tokenize using same approach
            tokens = re.findall(r'\w+', row['text'])

            # Get model prediction
            prediction = pipe(row['text'], return_all_scores=True)

            # Handle prediction format
            if isinstance(prediction[0], dict):
                pred_probs = {item['label']: item['score'] for item in prediction}
            else:
                pred_probs = {item['label']: item['score'] for item in prediction[0]}

            if 'LABEL_1' in pred_probs:
                predicted_label = 1 if pred_probs['LABEL_1'] > pred_probs['LABEL_0'] else 0
                manipulation_prob = pred_probs['LABEL_1']
            else:
                sorted_preds = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)
                predicted_label = 1 if 'manip' in sorted_preds[0][0].lower() else 0
                manipulation_prob = sorted_preds[0][1] if predicted_label == 1 else sorted_preds[1][1]

            # Get sentence boundaries
            sentence_boundaries = get_sentence_boundaries(row['text'], tokens)

            # Calculate sentence-level contributions
            sentence_contributions = []

            for sent_idx, sent_info in enumerate(sentence_boundaries):
                # Get dialogue_shap.py values for tokens in this sentence
                start_idx = sent_info['start_idx']
                end_idx = min(sent_info['end_idx'], len(specific_shap_values[0]))

                if start_idx < len(specific_shap_values[0]):
                    sentence_shap_values = specific_shap_values[0][start_idx:end_idx]
                    sentence_tokens = tokens[start_idx:end_idx]

                    # Aggregate token contributions for this sentence
                    sentence_contribution = sum(sentence_shap_values)
                    positive_contrib = sum([val for val in sentence_shap_values if val > 0])
                    negative_contrib = sum([val for val in sentence_shap_values if val < 0])

                    # Find top contributing tokens in this sentence
                    token_contributions = list(zip(sentence_tokens, sentence_shap_values))
                    token_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_tokens = token_contributions[:5]

                    sentence_info_detailed = {
                        'dialogue_id': dialogue_id,
                        'sentence_id': sent_idx,
                        'sentence': sent_info['sentence'],
                        'sentence_contribution': sentence_contribution,
                        'positive_contribution': positive_contrib,
                        'negative_contribution': negative_contrib,
                        'num_tokens': len(sentence_tokens),
                        'avg_token_contribution': sentence_contribution / len(
                            sentence_tokens) if sentence_tokens else 0,
                        'top_tokens': [f"{token}({val:.3f})" for token, val in top_tokens],
                        'dialogue_label': row['label'],
                        'predicted_label': predicted_label,
                        'manipulation_prob': manipulation_prob,
                        'category': row['category']
                    }

                    sentence_contributions.append(sentence_info_detailed)
                    sentence_results.append(sentence_info_detailed)

                    # Store individual token results
                    for token, value in zip(sentence_tokens, sentence_shap_values):
                        token_results.append({
                            'dialogue_id': dialogue_id,
                            'sentence_id': sent_idx,
                            'token': token,
                            'shap_value': value,
                            'sentence': sent_info['sentence'],
                            'dialogue': row['text'],
                            'actual_label': row['label'],
                            'predicted_label': predicted_label,
                            'category': row['category']
                        })

            # Create dialogue summary
            dialogue_summary = {
                'dialogue_id': dialogue_id,
                'dialogue': row['text'],
                'actual_label': row['label'],
                'predicted_label': predicted_label,
                'manipulation_prob': manipulation_prob,
                'category': row['category'],
                'num_sentences': len(sentence_boundaries),
                'total_contribution': sum([s['sentence_contribution'] for s in sentence_contributions])
            }
            dialogue_summaries.append(dialogue_summary)

        except Exception as e:
            print(f"Error processing dialogue {dialogue_id}: {e}")
            continue

    return pd.DataFrame(token_results), pd.DataFrame(sentence_results), pd.DataFrame(dialogue_summaries)


def analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries):
    """Analyze patterns across the extended dialogue set"""
    print(f"\n{'=' * 80}")
    print("EXTENDED dialogue_shap.py DIALOGUE ANALYSIS")
    print(f"{'=' * 80}")

    print(f"Analysis Summary:")
    print(f"  Total dialogues analyzed: {len(dialogue_summaries)}")
    print(f"  Total sentences analyzed: {len(sentence_results)}")
    print(f"  Manipulative dialogues: {len(dialogue_summaries[dialogue_summaries['actual_label'] == 1])}")
    print(f"  Non-manipulative dialogues: {len(dialogue_summaries[dialogue_summaries['actual_label'] == 0])}")

    # Prediction accuracy
    correct_predictions = len(
        dialogue_summaries[dialogue_summaries['actual_label'] == dialogue_summaries['predicted_label']])
    print(
        f"  Model accuracy: {correct_predictions}/{len(dialogue_summaries)} ({correct_predictions / len(dialogue_summaries) * 100:.1f}%)")

    # Average contributions by category
    print(f"\nAverage sentence contributions by category:")
    for category in sentence_results['category'].unique():
        category_data = sentence_results[sentence_results['category'] == category]
        avg_contrib = category_data['sentence_contribution'].mean()
        std_contrib = category_data['sentence_contribution'].std()
        print(f"  {category}: {avg_contrib:+.4f} ± {std_contrib:.4f}")

    # Top contributing sentences across all dialogues
    print(f"\nTop 10 Most Influential Sentences (by absolute dialogue_shap.py contribution):")
    print("-" * 80)

    top_sentences = sentence_results.reindex(
        sentence_results['sentence_contribution'].abs().sort_values(ascending=False).index
    ).head(10)

    for i, (_, row) in enumerate(top_sentences.iterrows(), 1):
        direction = "→ MANIP" if row['sentence_contribution'] > 0 else "→ NON-MANIP"
        correct = "✓" if row['dialogue_label'] == row['predicted_label'] else "✗"
        print(f"{i:2d}. [{row['sentence_contribution']:+.4f}] {direction} {correct} (D{row['dialogue_id']})")
        print(f"    Category: {row['category']}")
        print(f"    \"{row['sentence'][:100]}...\"")
        print()


def main():
    """Main function for Extended dialogue_shap.py analysis"""
    print("=== EXTENDED dialogue_shap.py DIALOGUE ANALYSIS ===")
    print("Using sampled dialogues for consistency with other methods")

    # Run Extended dialogue_shap.py analysis
    print("\n1. Running Extended dialogue_shap.py analysis...")
    try:
        token_results, sentence_results, dialogue_summaries = shap_analysis_extended_dialogues(model_path)

        if token_results is None:
            print("Analysis failed - no results generated")
            return

        print(f"dialogue_shap.py analysis complete!")
        print(f"- Token-level results: {len(token_results)}")
        print(f"- Sentence-level results: {len(sentence_results)}")
        print(f"- Dialogue summaries: {len(dialogue_summaries)}")

        # Enhanced analysis
        if len(sentence_results) > 0:
            analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries)

    except Exception as e:
        print(f"dialogue_shap.py analysis failed: {e}")
        return

    # Save results
    print("\n2. Saving extended results...")
    token_results.to_csv('shap_dialogue_tokens_extended.csv', index=False)
    sentence_results.to_csv('shap_dialogue_sentences_extended.csv', index=False)
    dialogue_summaries.to_csv('shap_dialogue_summaries_extended.csv', index=False)

    print("Files saved:")
    print("- shap_dialogue_tokens_extended.csv")
    print("- shap_dialogue_sentences_extended.csv")
    print("- shap_dialogue_summaries_extended.csv")
    print("\nResults ready for extended analysis and cross-method comparison!")


if __name__ == "__main__":
    main()