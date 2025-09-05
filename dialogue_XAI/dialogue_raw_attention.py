import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
import re
import json
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/dialogue_level_training/dialogue_models/dialogue_model_exp"


class RawAttentionDialogueExplainer:
    """
    Raw Attention implementation for dialogue-level manipulation detection.
    Adapted to use sampled dialogues and create comparable outputs.
    """

    def __init__(self, model, tokenizer, device=None, max_length=512):
        """
        Initialize the Raw Attention explainer for dialogues.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

    def get_sentence_boundaries(self, text: str, tokens: List[str]) -> List[Dict]:
        """
        Map token positions to sentence boundaries (same logic as other methods).
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
            while start_idx < len(tokens) and tokens[start_idx] in ['<s>', '</s>', '<pad>']:
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

    def explain_dialogue(self, dialogue_text: str, dialogue_id: int, category: str,
                         layer_idx: int = -1, head_idx: Optional[int] = None) -> Dict:
        """
        Extract raw attention weights for a dialogue and aggregate by sentences.
        Output structure matches other methods for comparison.
        """
        # Tokenize input dialogue
        inputs = self.tokenizer(
            dialogue_text,
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
                output_attentions=True
            )

            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()
            manipulation_prob = probabilities[0, 1].item()  # Probability of manipulative class

        # Extract attention weights from specified layer
        attention_weights = outputs.attentions[layer_idx][0]  # Remove batch dimension

        # Average across attention heads if no specific head is requested
        if head_idx is None:
            attention_matrix = attention_weights.mean(dim=0)
        else:
            attention_matrix = attention_weights[head_idx]

        # Get tokens for analysis
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids.squeeze().cpu().numpy())

        # Filter out padding tokens
        valid_length = attention_mask.sum().item()
        tokens = tokens[:valid_length]
        attention_matrix = attention_matrix[:valid_length, :valid_length]

        # Get CLS token attention (most important for classification)
        cls_attention = attention_matrix[0, :].cpu().numpy()

        # Get sentence boundaries
        sentence_boundaries = self.get_sentence_boundaries(dialogue_text, tokens)

        # Aggregate attention by sentences
        sentence_results = []
        token_results = []

        for sent_idx, sent_info in enumerate(sentence_boundaries):
            start_idx = sent_info['start_idx']
            end_idx = min(sent_info['end_idx'], len(cls_attention))

            if start_idx < len(cls_attention) and end_idx > start_idx:
                # Get attention weights for tokens in this sentence
                sentence_cls_attention = cls_attention[start_idx:end_idx]
                sentence_tokens = tokens[start_idx:end_idx]

                # Aggregate attention for this sentence
                sentence_contribution = np.sum(sentence_cls_attention)  # Total attention
                avg_contribution = np.mean(sentence_cls_attention)
                max_contribution = np.max(sentence_cls_attention)
                min_contribution = np.min(sentence_cls_attention)

                # Find top contributing tokens in this sentence
                token_attention_pairs = list(zip(sentence_tokens, sentence_cls_attention))
                token_attention_pairs.sort(key=lambda x: x[1], reverse=True)
                top_tokens = token_attention_pairs[:5]

                # Store sentence result
                sentence_result = {
                    'dialogue_id': dialogue_id,
                    'sentence_id': sent_idx,
                    'sentence': sent_info['sentence'],
                    'sentence_contribution': sentence_contribution,
                    'avg_contribution': avg_contribution,
                    'max_contribution': max_contribution,
                    'min_contribution': min_contribution,
                    'num_tokens': len(sentence_tokens),
                    'tokens': sentence_tokens,
                    'token_attributions': sentence_cls_attention.tolist(),
                    'top_tokens': [f"{token}({attention:.3f})" for token, attention in top_tokens],
                    'category': category,
                    'actual_label': None,  # Will be filled later
                    'predicted_class': predicted_class,
                    'manipulation_prob': manipulation_prob
                }

                sentence_results.append(sentence_result)

                # Store individual token results
                for token, attention_val in zip(sentence_tokens, sentence_cls_attention):
                    clean_token = token.replace('Ġ', '').replace('▁', '').strip()
                    if clean_token:
                        token_results.append({
                            'dialogue_id': dialogue_id,
                            'sentence_id': sent_idx,
                            'token': clean_token,
                            'attention_value': float(attention_val),
                            'sentence': sent_info['sentence'],
                            'dialogue': dialogue_text,
                            'category': category,
                            'actual_label': None,  # Will be filled later
                            'predicted_class': predicted_class
                        })

        return {
            'dialogue_text': dialogue_text,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'manipulation_prob': manipulation_prob,
            'num_sentences': len(sentence_boundaries),
            'num_tokens': len(token_results),
            'token_results': token_results,
            'sentence_results': sentence_results,
            'layer_used': layer_idx,
            'head_used': head_idx
        }


def raw_attention_extended_dialogues(model_path):
    """
    Raw Attention analysis using the sampled dialogues.
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

    # Load model
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            num_labels=2,
            id2label={0: "Non-Manipulative", 1: "Manipulative"},
            label2id={"Non-Manipulative": 0, "Manipulative": 1}
        )
        tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✓ Successfully loaded dialogue model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

    # Initialize explainer
    explainer = RawAttentionDialogueExplainer(model, tokenizer, device, max_length=512)

    all_token_results = []
    all_sentence_results = []
    dialogue_summaries = []

    print(f"\nAnalyzing {len(sampled_df)} dialogues with Raw Attention...")

    for index, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing dialogues"):
        dialogue_id = row['dialogue_id']

        try:
            # Analyze this dialogue
            analysis = explainer.explain_dialogue(
                row['text'],
                dialogue_id,
                row['category'],
                layer_idx=-1  # Use last layer
            )

            # Add metadata to results
            for token_result in analysis['token_results']:
                token_result['actual_label'] = row['label']

            for sentence_result in analysis['sentence_results']:
                sentence_result['actual_label'] = row['label']
                sentence_result['dialogue'] = row['text']

            all_token_results.extend(analysis['token_results'])
            all_sentence_results.extend(analysis['sentence_results'])

            # Create dialogue summary
            dialogue_summary = {
                'dialogue_id': dialogue_id,
                'dialogue': row['text'],
                'actual_label': row['label'],
                'predicted_label': analysis['predicted_class'],
                'manipulation_prob': analysis['manipulation_prob'],
                'category': row['category'],
                'num_sentences': analysis['num_sentences'],
                'total_contribution': sum([s['sentence_contribution'] for s in analysis['sentence_results']]),
                'layer_used': analysis['layer_used'],
                'head_used': analysis['head_used']
            }
            dialogue_summaries.append(dialogue_summary)

        except Exception as e:
            print(f"Error processing dialogue {dialogue_id}: {e}")
            continue

    return pd.DataFrame(all_token_results), pd.DataFrame(all_sentence_results), pd.DataFrame(dialogue_summaries)


def analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries):
    """Analyze patterns across the extended dialogue set"""
    print(f"\n{'=' * 80}")
    print("EXTENDED RAW ATTENTION DIALOGUE ANALYSIS")
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
    print(f"\nAverage sentence attention by category:")
    for category in sentence_results['category'].unique():
        category_data = sentence_results[sentence_results['category'] == category]
        avg_contrib = category_data['sentence_contribution'].mean()
        std_contrib = category_data['sentence_contribution'].std()
        print(f"  {category}: {avg_contrib:.4f} ± {std_contrib:.4f}")

    # Top contributing sentences across all dialogues
    print(f"\nTop 10 Most Influential Sentences (by Raw Attention contribution):")
    print("-" * 80)

    # Sort by attention contribution (higher = more attended to)
    top_sentences = sentence_results.nlargest(10, 'sentence_contribution')

    for i, (_, row) in enumerate(top_sentences.iterrows(), 1):
        attention_level = "HIGH-ATTN" if row['sentence_contribution'] > sentence_results[
            'sentence_contribution'].median() else "LOW-ATTN"
        correct = "✓" if row['actual_label'] == row['predicted_class'] else "✗"
        print(f"{i:2d}. [{row['sentence_contribution']:.4f}] {attention_level} {correct} (D{row['dialogue_id']})")
        print(f"    Category: {row['category']}")
        print(f"    \"{row['sentence'][:100]}...\"")
        print()


def main():
    """Main function for Extended Raw Attention analysis"""
    print("=== EXTENDED RAW ATTENTION DIALOGUE ANALYSIS ===")
    print("Using sampled dialogues for consistency with other methods")

    # Run Extended Raw Attention analysis
    print("\n1. Running Extended Raw Attention analysis...")
    try:
        token_results, sentence_results, dialogue_summaries = raw_attention_extended_dialogues(model_path)

        if token_results is None:
            print("Analysis failed - no results generated")
            return

        print(f"Raw Attention analysis complete!")
        print(f"- Token-level results: {len(token_results)}")
        print(f"- Sentence-level results: {len(sentence_results)}")
        print(f"- Dialogue summaries: {len(dialogue_summaries)}")

        # Enhanced analysis
        if len(sentence_results) > 0:
            analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries)

    except Exception as e:
        print(f"Raw Attention analysis failed: {e}")
        return

    # Save results
    print("\n2. Saving extended results...")
    token_results.to_csv('raw_attention_dialogue_tokens_extended.csv', index=False)
    sentence_results.to_csv('raw_attention_dialogue_sentences_extended.csv', index=False)
    dialogue_summaries.to_csv('raw_attention_dialogue_summaries_extended.csv', index=False)

    print("Files saved:")
    print("- raw_attention_dialogue_tokens_extended.csv")
    print("- raw_attention_dialogue_sentences_extended.csv")
    print("- raw_attention_dialogue_summaries_extended.csv")
    print("\nResults ready for extended analysis and cross-method comparison!")


if __name__ == "__main__":
    main()