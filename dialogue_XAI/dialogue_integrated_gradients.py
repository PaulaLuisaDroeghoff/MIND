from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import re
import json
from datetime import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/dialogue_level_training/dialogue_models/dialogue_model_exp"


# ===== DIALOGUE MODEL WRAPPER FOR INTEGRATED GRADIENTS =====
class DialogueManipulationModelWrapper(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        import os
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False
        )
        self.model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def forward(self, embeddings, attention_mask=None):
        """Forward method expects embeddings as input and returns manipulation probability"""
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        probs = F.softmax(outputs.logits, dim=-1)
        return probs[:, 1]  # Manipulative class probability

    def embed_input(self, input_ids):
        """Get embeddings for input_ids"""
        return self.model.get_input_embeddings()(input_ids)


# ===== SENTENCE BOUNDARY DETECTION =====
def get_sentence_boundaries_ig(text, tokens):
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
        while start_idx < len(tokens) and tokens[start_idx] in ['<s>', '</s>', '<pad>', '[CLS]', '[SEP]']:
            start_idx += 1

        if start_idx < len(tokens):
            sentence_boundaries.append({
                'sentence': sentence,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'estimated_tokens': estimated_tokens
            })
            current_token_idx = end_idx

    return sentence_boundaries


# ===== TOKENIZER FUNCTIONS =====
def custom_tokenizer(text):
    """Same tokenizer as your other methods for consistency"""
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens


def prepare_dialogue_inputs(text, tokenizer, max_length=512):
    """Prepare dialogue text for model input"""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True
    )
    return encoded


# ===== ENHANCED ATTRIBUTION MAPPING =====
def map_attributions_to_custom_tokens_dialogue(model_tokens, model_attributions, custom_tokens, original_text,
                                               tokenizer):
    """
    Enhanced mapping of model subword tokens to custom word tokens for dialogue
    """
    # Remove special tokens and their attributions
    clean_tokens = []
    clean_attributions = []

    for i, (token, attr) in enumerate(zip(model_tokens, model_attributions)):
        if token not in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>', tokenizer.unk_token, tokenizer.pad_token]:
            clean_tokens.append(token)
            # Handle multi-dimensional attributions
            if isinstance(attr, (np.ndarray, torch.Tensor)):
                if hasattr(attr, 'detach'):
                    attr = attr.detach().cpu().numpy()
                # Take L2 norm across embedding dimensions
                attr_score = float(np.linalg.norm(attr))
            else:
                attr_score = float(attr)
            clean_attributions.append(attr_score)

    # Map to custom tokens
    word_attributions = []

    for custom_token in custom_tokens:
        total_attr = 0.0
        count = 0

        for model_token, attr in zip(clean_tokens, clean_attributions):
            # Clean model token
            clean_model_token = model_token.replace('##', '').replace('Ġ', '').replace('▁', '').lower()
            custom_lower = custom_token.lower()

            # Enhanced matching for dialogue context
            if (clean_model_token in custom_lower or
                    custom_lower in clean_model_token or
                    clean_model_token == custom_lower or
                    abs(len(clean_model_token) - len(custom_lower)) <= 1):
                total_attr += attr
                count += 1

        # Average or use zero if no match
        if count > 0:
            word_attributions.append(total_attr / count)
        else:
            word_attributions.append(0.0)

    return word_attributions


# ===== EXTENDED INTEGRATED GRADIENTS ANALYSIS =====
def integrated_gradients_extended_dialogues(model_path):
    """
    Integrated Gradients analysis using the sampled dialogues.
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
        # Load wrapped model
        model_wrapper = DialogueManipulationModelWrapper(model_path)
        tokenizer = model_wrapper.tokenizer
        device = model_wrapper.device

        print("✓ Dialogue model loaded successfully for Integrated Gradients analysis")

        # Initialize Integrated Gradients
        ig = IntegratedGradients(model_wrapper)

        all_token_results = []
        all_sentence_results = []
        dialogue_summaries = []

        print(f"\nAnalyzing {len(sampled_df)} dialogues with Integrated Gradients...")

        for index, row in tqdm(sampled_df.iterrows(), total=len(sampled_df), desc="Processing dialogues"):
            dialogue_text = row['text']
            dialogue_id = row['dialogue_id']

            try:
                # Prepare inputs
                encoded = prepare_dialogue_inputs(dialogue_text, tokenizer)
                input_ids = encoded['input_ids'].to(device)
                attention_mask = encoded['attention_mask'].to(device)

                # Get embeddings with requires_grad=True
                embeddings = model_wrapper.embed_input(input_ids)
                embeddings.requires_grad_()

                # Baseline embeddings (pad tokens)
                baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
                baseline_embeddings = model_wrapper.embed_input(baseline_input_ids)

                # Compute attributions
                attributions, convergence_delta = ig.attribute(
                    embeddings,
                    baselines=baseline_embeddings,
                    additional_forward_args=(attention_mask,),
                    n_steps=25,
                    return_convergence_delta=True,
                    internal_batch_size=1
                )

                # Get model prediction
                with torch.no_grad():
                    prediction_score = model_wrapper(embeddings, attention_mask).item()
                    predicted_label = 1 if prediction_score > 0.5 else 0

                # Convert to tokens and attributions
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                token_attributions = attributions[0].cpu().detach().numpy()

                # Get sentence boundaries
                sentence_boundaries = get_sentence_boundaries_ig(dialogue_text, tokens)

                # Use custom tokenization for consistency
                custom_tokens = custom_tokenizer(dialogue_text)

                if not custom_tokens:
                    continue

                # Map model tokens to custom tokens
                mapped_attributions = map_attributions_to_custom_tokens_dialogue(
                    tokens, token_attributions, custom_tokens, dialogue_text, tokenizer
                )

                # Map custom tokens to sentences
                token_to_sentence = {}
                current_custom_idx = 0

                for sent_idx, sent_info in enumerate(sentence_boundaries):
                    sentence_words = custom_tokenizer(sent_info['sentence'])
                    for _ in range(len(sentence_words)):
                        if current_custom_idx < len(custom_tokens):
                            token_to_sentence[current_custom_idx] = sent_idx
                            current_custom_idx += 1

                # Store token-level results
                for token_idx, (token, attribution) in enumerate(zip(custom_tokens, mapped_attributions)):
                    if token.strip():
                        sentence_id = token_to_sentence.get(token_idx, -1)
                        all_token_results.append({
                            'dialogue_id': dialogue_id,
                            'sentence_id': sentence_id,
                            'token': token,
                            'ig_value': float(attribution),
                            'dialogue': dialogue_text,
                            'actual_label': int(row['label']),
                            'predicted_label': int(predicted_label),
                            'prediction_score': float(prediction_score),
                            'convergence_delta': float(convergence_delta.item()),
                            'category': row['category']
                        })

                # Aggregate by sentences
                sentence_contributions = {}
                for token_idx, attribution in enumerate(mapped_attributions):
                    sentence_id = token_to_sentence.get(token_idx, -1)
                    if sentence_id >= 0:
                        if sentence_id not in sentence_contributions:
                            sentence_contributions[sentence_id] = []
                        sentence_contributions[sentence_id].append(attribution)

                # Create sentence-level results
                for sent_idx, sent_info in enumerate(sentence_boundaries):
                    if sent_idx in sentence_contributions:
                        sent_attributions = sentence_contributions[sent_idx]
                        sent_tokens = [custom_tokens[i] for i in range(len(custom_tokens))
                                       if token_to_sentence.get(i, -1) == sent_idx]

                        sentence_result = {
                            'dialogue_id': dialogue_id,
                            'sentence_id': sent_idx,
                            'sentence': sent_info['sentence'],
                            'sentence_contribution': sum(sent_attributions),
                            'avg_contribution': np.mean(sent_attributions),
                            'max_contribution': max(sent_attributions),
                            'min_contribution': min(sent_attributions),
                            'num_tokens': len(sent_attributions),
                            'tokens': sent_tokens,
                            'token_attributions': sent_attributions,
                            'top_tokens': [f"{sent_tokens[i]}({sent_attributions[i]:.3f})"
                                           for i in sorted(range(len(sent_attributions)),
                                                           key=lambda x: abs(sent_attributions[x]), reverse=True)[:5]],
                            'actual_label': int(row['label']),
                            'predicted_label': int(predicted_label),
                            'manipulation_prob': float(prediction_score),
                            'category': row['category'],
                            'dialogue': dialogue_text
                        }

                        all_sentence_results.append(sentence_result)

                # Create dialogue summary
                dialogue_summary = {
                    'dialogue_id': dialogue_id,
                    'dialogue': row['text'],
                    'actual_label': int(row['label']),
                    'predicted_label': int(predicted_label),
                    'manipulation_prob': float(prediction_score),
                    'category': row['category'],
                    'num_sentences': len(sentence_boundaries),
                    'total_contribution': sum(mapped_attributions),
                    'convergence_delta': float(convergence_delta.item())
                }
                dialogue_summaries.append(dialogue_summary)

            except Exception as e:
                print(f"Error processing dialogue {dialogue_id}: {e}")
                continue

        return pd.DataFrame(all_token_results), pd.DataFrame(all_sentence_results), pd.DataFrame(dialogue_summaries)

    except Exception as e:
        print(f"Error in Extended Integrated Gradients dialogue analysis: {e}")
        raise


def analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries):
    """Analyze patterns across the extended dialogue set"""
    print(f"\n{'=' * 80}")
    print("EXTENDED INTEGRATED GRADIENTS DIALOGUE ANALYSIS")
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

    # Convergence quality
    avg_convergence = dialogue_summaries['convergence_delta'].mean()
    print(f"  Average convergence delta: {avg_convergence:.6f}")

    # Average contributions by category
    print(f"\nAverage sentence contributions by category:")
    for category in sentence_results['category'].unique():
        category_data = sentence_results[sentence_results['category'] == category]
        avg_contrib = category_data['sentence_contribution'].mean()
        std_contrib = category_data['sentence_contribution'].std()
        print(f"  {category}: {avg_contrib:.6f} ± {std_contrib:.6f}")

    # Top contributing sentences
    print(f"\nTop 10 Most Influential Sentences (by absolute IG contribution):")
    print("-" * 80)

    top_sentences = sentence_results.reindex(
        sentence_results['sentence_contribution'].abs().sort_values(ascending=False).index
    ).head(10)

    for i, (_, row) in enumerate(top_sentences.iterrows(), 1):
        direction = "→ MANIP" if row['sentence_contribution'] > 0 else "→ NON-MANIP"
        correct = "✓" if row['actual_label'] == row['predicted_label'] else "✗"
        print(f"{i:2d}. [{row['sentence_contribution']:+.6f}] {direction} {correct} (D{row['dialogue_id']})")
        print(f"    Category: {row['category']}")
        print(f"    \"{row['sentence'][:100]}...\"")
        print()


def main():
    """Main function for Extended Integrated Gradients analysis"""
    print("=== EXTENDED INTEGRATED GRADIENTS DIALOGUE ANALYSIS ===")
    print("Using sampled dialogues for consistency with Expected Gradients")

    # Run Extended Integrated Gradients analysis
    print("\n1. Running Extended Integrated Gradients analysis...")
    try:
        token_results, sentence_results, dialogue_summaries = integrated_gradients_extended_dialogues(model_path)

        if token_results is None:
            print("Analysis failed - no results generated")
            return

        print(f"Integrated Gradients analysis complete!")
        print(f"- Token-level results: {len(token_results)}")
        print(f"- Sentence-level results: {len(sentence_results)}")
        print(f"- Dialogue summaries: {len(dialogue_summaries)}")

        # Enhanced analysis
        if len(sentence_results) > 0:
            analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries)

    except Exception as e:
        print(f"Integrated Gradients analysis failed: {e}")
        return

    # Save results
    print("\n2. Saving extended results...")
    token_results.to_csv('integrated_gradients_dialogue_tokens_extended.csv', index=False)
    sentence_results.to_csv('integrated_gradients_dialogue_sentences_extended.csv', index=False)
    dialogue_summaries.to_csv('integrated_gradients_dialogue_summaries_extended.csv', index=False)

    print("Files saved:")
    print("- integrated_gradients_dialogue_tokens_extended.csv")
    print("- integrated_gradients_dialogue_sentences_extended.csv")
    print("- integrated_gradients_dialogue_summaries_extended.csv")
    print("\nResults ready for extended analysis and cross-method comparison!")


if __name__ == "__main__":
    main()