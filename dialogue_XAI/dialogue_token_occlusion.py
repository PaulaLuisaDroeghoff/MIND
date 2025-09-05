import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/dialogue_level_training/dialogue_models/dialogue_model_exp"

# Load model & tokenizer for dialogue-level analysis
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Ensure pad token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


def get_sentence_boundaries_token_occlusion(text, tokens):
    """
    Map token positions to sentence boundaries for dialogue analysis.
    Same logic as other methods for consistency.
    """
    # Split dialogue into sentences
    sentences = re.split(r'[.!?]+\s*', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_boundaries = []
    current_token_idx = 0

    for sentence in sentences:
        if not sentence:
            continue

        # Estimate token count for this sentence
        sentence_word_count = len(re.findall(r'\b\w+\b', sentence))
        estimated_tokens = int(sentence_word_count * 1.3)  # Account for subword splits

        start_idx = current_token_idx
        end_idx = min(start_idx + estimated_tokens, len(tokens))

        # Skip special tokens
        while start_idx < len(tokens) and tokens[start_idx] in ['<s>', '</s>', '<pad>', tokenizer.pad_token]:
            start_idx += 1

        if start_idx < len(tokens):
            sentence_boundaries.append({
                'sentence': sentence,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'tokens': tokens[start_idx:end_idx] if end_idx <= len(tokens) else tokens[start_idx:]
            })
            current_token_idx = end_idx

    return sentence_boundaries


def analyze_dialogue_with_token_occlusion(dialogue_text, dialogue_id, category, max_length=512):
    """
    Token Occlusion analysis for a single dialogue with sentence aggregation.
    Maintains same output structure as other methods for easy comparison.
    """

    # Tokenize input dialogue
    input_encoded = tokenizer(
        dialogue_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    input_ids = input_encoded.input_ids.to(device)
    attention_mask = input_encoded.attention_mask.to(device)

    # Get baseline prediction (full dialogue)
    with torch.no_grad():
        baseline_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        baseline_pred = torch.softmax(baseline_outputs.logits, dim=-1)[0, 1].item()

    # Get tokens and find valid (non-padding) tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    valid_length = attention_mask.sum().item()
    tokens = tokens[:valid_length]

    # Get sentence boundaries
    sentence_boundaries = get_sentence_boundaries_token_occlusion(dialogue_text, tokens)

    # Analyze each token
    token_results = []
    mask_id = tokenizer.mask_token_id if tokenizer.mask_token_id else tokenizer.unk_token_id

    for i, token in enumerate(tokens):
        if token not in ['<s>', '</s>', '<pad>', tokenizer.pad_token]:
            # Create modified input with this token replaced by [MASK]
            modified_ids = input_ids.clone()
            modified_ids[0, i] = mask_id

            # Get prediction with masked token
            with torch.no_grad():
                modified_encoded = {
                    'input_ids': modified_ids,
                    'attention_mask': attention_mask
                }
                outputs = model(**modified_encoded)
                masked_pred = torch.softmax(outputs.logits, dim=-1)[0, 1].item()

            # Calculate importance (how much manipulation score drops when token is removed)
            importance = baseline_pred - masked_pred

            # Find which sentence this token belongs to
            sentence_id = -1
            for sent_idx, sent_info in enumerate(sentence_boundaries):
                if sent_info['start_idx'] <= i < sent_info['end_idx']:
                    sentence_id = sent_idx
                    break

            clean_token = token.replace('Ġ', '').replace('▁', '').strip()
            if clean_token:
                token_results.append({
                    'dialogue_id': dialogue_id,
                    'sentence_id': sentence_id,
                    'token': clean_token,
                    'occlusion_value': importance,
                    'baseline_pred': baseline_pred,
                    'masked_pred': masked_pred,
                    'dialogue': dialogue_text,
                    'category': category
                })

    # Aggregate results by sentence
    sentence_results = []

    for sent_idx, sent_info in enumerate(sentence_boundaries):
        # Get tokens for this sentence
        sentence_tokens = [r for r in token_results if r['sentence_id'] == sent_idx]

        if sentence_tokens:
            # Calculate sentence-level statistics
            sentence_attributions = [t['occlusion_value'] for t in sentence_tokens]
            sentence_contribution = sum(sentence_attributions)
            avg_contribution = np.mean(sentence_attributions)
            max_contribution = max(sentence_attributions)
            min_contribution = min(sentence_attributions)

            # Find top contributing tokens in this sentence
            sorted_tokens = sorted(sentence_tokens, key=lambda x: abs(x['occlusion_value']), reverse=True)
            top_tokens = sorted_tokens[:5]

            sentence_result = {
                'dialogue_id': dialogue_id,
                'sentence_id': sent_idx,
                'sentence': sent_info['sentence'],
                'sentence_contribution': sentence_contribution,
                'avg_contribution': avg_contribution,
                'max_contribution': max_contribution,
                'min_contribution': min_contribution,
                'num_tokens': len(sentence_tokens),
                'tokens': [t['token'] for t in sentence_tokens],
                'token_attributions': [t['occlusion_value'] for t in sentence_tokens],
                'top_tokens': [f"{t['token']}({t['occlusion_value']:.3f})" for t in top_tokens],
                'category': category
            }

            sentence_results.append(sentence_result)

    return {
        'dialogue_text': dialogue_text,
        'baseline_prediction': baseline_pred,
        'predicted_class': 1 if baseline_pred > 0.5 else 0,
        'confidence': baseline_pred,
        'num_sentences': len(sentence_boundaries),
        'num_tokens': len(token_results),
        'token_results': token_results,
        'sentence_results': sentence_results
    }


def token_occlusion_extended_dialogues(model_path):
    """
    Token Occlusion analysis using the sampled dialogues.
    Output format matches other methods for easy comparison.
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

    all_token_results = []
    all_sentence_results = []
    dialogue_summaries = []

    print(f"\nAnalyzing {len(sampled_df)} dialogues with Token Occlusion...")

    for index, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing dialogues"):
        dialogue_id = row['dialogue_id']

        try:
            # Analyze this dialogue
            analysis = analyze_dialogue_with_token_occlusion(
                row['text'],
                dialogue_id,
                row['category']
            )

            # Add metadata to results
            for token_result in analysis['token_results']:
                token_result['actual_label'] = row['label']
                token_result['predicted_class'] = analysis['predicted_class']

            for sentence_result in analysis['sentence_results']:
                sentence_result['actual_label'] = row['label']
                sentence_result['predicted_class'] = analysis['predicted_class']
                sentence_result['dialogue'] = row['text']
                sentence_result['manipulation_prob'] = analysis['confidence']

            all_token_results.extend(analysis['token_results'])
            all_sentence_results.extend(analysis['sentence_results'])

            # Create dialogue summary
            dialogue_summary = {
                'dialogue_id': dialogue_id,
                'dialogue': row['text'],
                'actual_label': row['label'],
                'predicted_label': analysis['predicted_class'],
                'manipulation_prob': analysis['confidence'],
                'category': row['category'],
                'num_sentences': analysis['num_sentences'],
                'total_contribution': sum([s['sentence_contribution'] for s in analysis['sentence_results']])
            }
            dialogue_summaries.append(dialogue_summary)

        except Exception as e:
            print(f"Error processing dialogue {dialogue_id}: {e}")
            continue

    return pd.DataFrame(all_token_results), pd.DataFrame(all_sentence_results), pd.DataFrame(dialogue_summaries)


def analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries):
    """Analyze patterns across the extended dialogue set"""
    print(f"\n{'=' * 80}")
    print("EXTENDED TOKEN OCCLUSION DIALOGUE ANALYSIS")
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
    print(f"\nTop 10 Most Influential Sentences (by absolute Token Occlusion contribution):")
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
    """Main function for Extended Token Occlusion analysis"""
    print("=== EXTENDED TOKEN OCCLUSION DIALOGUE ANALYSIS ===")
    print("Using sampled dialogues for consistency with other methods")

    # Run Extended Token Occlusion analysis
    print("\n1. Running Extended Token Occlusion analysis...")
    try:
        token_results, sentence_results, dialogue_summaries = token_occlusion_extended_dialogues(model_path)

        if token_results is None:
            print("Analysis failed - no results generated")
            return

        print(f"Token Occlusion analysis complete!")
        print(f"- Token-level results: {len(token_results)}")
        print(f"- Sentence-level results: {len(sentence_results)}")
        print(f"- Dialogue summaries: {len(dialogue_summaries)}")

        # Enhanced analysis
        if len(sentence_results) > 0:
            analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries)

    except Exception as e:
        print(f"Token Occlusion analysis failed: {e}")
        return

    # Save results
    print("\n2. Saving extended results...")
    token_results.to_csv('token_occlusion_dialogue_tokens_extended.csv', index=False)
    sentence_results.to_csv('token_occlusion_dialogue_sentences_extended.csv', index=False)
    dialogue_summaries.to_csv('token_occlusion_dialogue_summaries_extended.csv', index=False)

    print("Files saved:")
    print("- token_occlusion_dialogue_tokens_extended.csv")
    print("- token_occlusion_dialogue_sentences_extended.csv")
    print("- token_occlusion_dialogue_summaries_extended.csv")
    print("\nResults ready for extended analysis and cross-method comparison!")


if __name__ == "__main__":
    main()