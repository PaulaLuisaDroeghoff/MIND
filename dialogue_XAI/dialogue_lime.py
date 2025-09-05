from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
from dialogue_lime.lime_text import LimeTextExplainer
import re
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
from tqdm import tqdm

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/dialogue_level_training/dialogue_models/dialogue_model_exp"


# ===== CUSTOM TOKENIZER (same as your original) =====
def custom_tokenizer(text):
    """Same tokenizer as your original LIME code - keep unchanged"""
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens


# ===== SENTENCE BOUNDARY DETECTION =====
def get_sentence_boundaries_lime(text, tokens):
    """
    Map token positions to sentence boundaries for LIME (same logic as other methods)
    """
    # Split dialogue into sentences
    sentences = re.split(r'[.!?]+\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_boundaries = []
    current_token_idx = 0

    for sentence in sentences:
        if not sentence:
            continue

        # Get tokens for this sentence using the same tokenizer as LIME
        sentence_tokens = custom_tokenizer(sentence)

        # Estimate token boundaries (simpler approach for consistency)
        start_idx = current_token_idx
        end_idx = min(start_idx + len(sentence_tokens), len(tokens))

        if start_idx < len(tokens):
            sentence_boundaries.append({
                'sentence': sentence,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'tokens': tokens[start_idx:end_idx]
            })
            current_token_idx = end_idx

    return sentence_boundaries


# ===== EXTENDED LIME ANALYSIS =====
def lime_extended_dialogues(model_path):
    """
    LIME analysis using the sampled dialogues.
    Output format matches other methods for easy comparison.
    """

    # Load pre-sampled dialogues for consistency with Expected Gradients
    print("Loading pre-sampled dialogues for consistency with Expected Gradients...")

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
        pipe = pipeline(
            "text-classification",
            model=model_path,
            return_all_scores=True,
            device=0 if torch.cuda.is_available() else -1
        )
        print("✓ Dialogue model loaded successfully for LIME analysis")
    except Exception as e:
        print(f"Error loading model for LIME: {e}")
        return None, None, None

    def predict_proba(texts):
        """Prediction function for LIME - adapted for your dialogue model"""
        preds = pipe(texts, return_all_scores=True)
        probabilities = np.array([[pred['score'] for pred in preds_single] for preds_single in preds])
        return probabilities

    # Initialize LIME explainer
    explainer = LimeTextExplainer(
        class_names=['LABEL_0', 'LABEL_1'],
        split_expression=lambda x: custom_tokenizer(x)
    )

    all_token_results = []
    all_sentence_results = []
    dialogue_summaries = []

    print(f"\nAnalyzing {len(sampled_df)} dialogues with LIME...")

    for index, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing dialogues"):
        dialogue_text = row['text']
        dialogue_id = row['dialogue_id']
        tokens = custom_tokenizer(dialogue_text)

        try:
            # Generate LIME explanation for the full dialogue
            exp = explainer.explain_instance(
                dialogue_text,
                predict_proba,
                num_features=len(tokens),
                num_samples=100
            )

            # Get model prediction
            prediction = pipe(dialogue_text, return_all_scores=True)

            # Handle different prediction formats
            if isinstance(prediction[0], dict):
                pred_probs = {item['label']: item['score'] for item in prediction}
            else:
                pred_probs = {item['label']: item['score'] for item in prediction[0]}

            # Determine prediction
            if 'LABEL_1' in pred_probs:
                predicted_label = 1 if pred_probs['LABEL_1'] > pred_probs['LABEL_0'] else 0
                manipulation_prob = pred_probs['LABEL_1']
            else:
                sorted_preds = sorted(pred_probs.items(), key=lambda x: x[1], reverse=True)
                predicted_label = 1 if 'manip' in sorted_preds[0][0].lower() else 0
                manipulation_prob = sorted_preds[0][1] if predicted_label == 1 else sorted_preds[1][1]

            # Focus on manipulative class (label=1)
            explanation_list = exp.as_list(label=1)
            token_value_dict = {token: value for token, value in explanation_list}

            # Get sentence boundaries
            sentence_boundaries = get_sentence_boundaries_lime(dialogue_text, tokens)

            # Calculate sentence-level contributions
            sentence_contributions = []

            for sent_idx, sent_info in enumerate(sentence_boundaries):
                # Get LIME values for tokens in this sentence
                start_idx = sent_info['start_idx']
                end_idx = min(sent_info['end_idx'], len(tokens))

                if start_idx < len(tokens):
                    sentence_tokens = tokens[start_idx:end_idx]
                    sentence_lime_values = [token_value_dict.get(token, 0) for token in sentence_tokens]

                    # Aggregate token contributions for this sentence
                    sentence_contribution = sum(sentence_lime_values)
                    avg_contribution = np.mean(sentence_lime_values) if sentence_lime_values else 0
                    max_contribution = max(sentence_lime_values) if sentence_lime_values else 0
                    min_contribution = min(sentence_lime_values) if sentence_lime_values else 0
                    positive_contrib = sum([val for val in sentence_lime_values if val > 0])
                    negative_contrib = sum([val for val in sentence_lime_values if val < 0])

                    # Find top contributing tokens in this sentence
                    token_contributions = [(token, token_value_dict.get(token, 0)) for token in sentence_tokens]
                    token_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                    top_tokens = token_contributions[:5]

                    # Store sentence result
                    sentence_info_detailed = {
                        'dialogue_id': dialogue_id,
                        'sentence_id': sent_idx,
                        'sentence': sent_info['sentence'],
                        'sentence_contribution': sentence_contribution,
                        'avg_contribution': avg_contribution,
                        'max_contribution': max_contribution,
                        'min_contribution': min_contribution,
                        'positive_contribution': positive_contrib,
                        'negative_contribution': negative_contrib,
                        'num_tokens': len(sentence_tokens),
                        'tokens': sentence_tokens,
                        'token_attributions': sentence_lime_values,
                        'top_tokens': [f"{token}({val:.3f})" for token, val in top_tokens],
                        'actual_label': row['label'],
                        'predicted_class': predicted_label,
                        'manipulation_prob': manipulation_prob,
                        'category': row['category'],
                        'dialogue': dialogue_text
                    }

                    sentence_contributions.append(sentence_info_detailed)
                    all_sentence_results.append(sentence_info_detailed)

                    # Store individual token results
                    for token in sentence_tokens:
                        value = token_value_dict.get(token, 0)
                        all_token_results.append({
                            'dialogue_id': dialogue_id,
                            'sentence_id': sent_idx,
                            'token': token,
                            'lime_value': value,
                            'sentence': sent_info['sentence'],
                            'dialogue': dialogue_text,
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
            print(f"Error processing dialogue {dialogue_id} with LIME: {e}")
            continue

    return pd.DataFrame(all_token_results), pd.DataFrame(all_sentence_results), pd.DataFrame(dialogue_summaries)


def analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries):
    """Analyze patterns across the extended dialogue set"""
    print(f"\n{'=' * 80}")
    print("EXTENDED LIME DIALOGUE ANALYSIS")
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
        print(f"  {category}: {avg_contrib:.4f} ± {std_contrib:.4f}")

    # Top contributing sentences
    print(f"\nTop 10 Most Influential Sentences (by absolute LIME contribution):")
    print("-" * 80)

    top_sentences = sentence_results.reindex(
        sentence_results['sentence_contribution'].abs().sort_values(ascending=False).index
    ).head(10)

    for i, (_, row) in enumerate(top_sentences.iterrows(), 1):
        direction = "→ MANIP" if row['sentence_contribution'] > 0 else "→ NON-MANIP"
        correct = "✓" if row['actual_label'] == row['predicted_class'] else "✗"
        print(f"{i:2d}. [{row['sentence_contribution']:+.4f}] {direction} {correct} (D{row['dialogue_id']})")
        print(f"    Category: {row['category']}")
        print(f"    \"{row['sentence'][:100]}...\"")
        print()


def main():
    """Main function for Extended LIME analysis"""
    print("=== EXTENDED LIME DIALOGUE ANALYSIS ===")
    print("Using sampled dialogues for consistency with Expected Gradients")

    # Run Extended LIME analysis
    print("\n1. Running Extended LIME analysis...")
    try:
        token_results, sentence_results, dialogue_summaries = lime_extended_dialogues(model_path)

        if token_results is None:
            print("Analysis failed - no results generated")
            return

        print(f"LIME analysis complete!")
        print(f"- Token-level results: {len(token_results)}")
        print(f"- Sentence-level results: {len(sentence_results)}")
        print(f"- Dialogue summaries: {len(dialogue_summaries)}")

        # Enhanced analysis
        if len(sentence_results) > 0:
            analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries)

    except Exception as e:
        print(f"LIME analysis failed: {e}")
        return

    # Save results
    print("\n2. Saving extended results...")
    token_results.to_csv('lime_dialogue_tokens_extended.csv', index=False)
    sentence_results.to_csv('lime_dialogue_sentences_extended.csv', index=False)
    dialogue_summaries.to_csv('lime_dialogue_summaries_extended.csv', index=False)

    print("Files saved:")
    print("- lime_dialogue_tokens_extended.csv")
    print("- lime_dialogue_sentences_extended.csv")
    print("- lime_dialogue_summaries_extended.csv")
    print("\nResults ready for extended analysis and cross-method comparison!")


if __name__ == "__main__":
    main()