from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
import re
import warnings

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/sentence_level_training/dialogue_models/sentence_model_exp"


# ===== MODEL WRAPPER FOR INTEGRATED GRADIENTS =====
class ManipulationModelWrapper(torch.nn.Module):
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
        """Forward method expects embeddings as input (requires_grad=True)"""
        outputs = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask
        )
        probs = F.softmax(outputs.logits, dim=-1)
        return probs[:, 1]  # Manipulative class probability

    def embed_input(self, input_ids):
        """Get embeddings for input_ids"""
        return self.model.get_input_embeddings()(input_ids)


def custom_tokenizer(text):
    """Same tokenizer as other methods for consistency"""
    tokens = re.split(r'\W+', text)
    tokens = [token for token in tokens if token]
    return tokens


def prepare_model_inputs(text, tokenizer, max_length=64):
    """Prepare text for model input"""
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True
    )
    return encoded


def map_attributions_to_custom_tokens(model_tokens, model_attributions, custom_tokens, tokenizer):
    """Map model subword tokens to custom word tokens"""
    # Remove special tokens and their attributions
    clean_tokens = []
    clean_attributions = []

    for i, (token, attr) in enumerate(zip(model_tokens, model_attributions)):
        if token not in ['[CLS]', '[SEP]', '[PAD]', tokenizer.unk_token, tokenizer.pad_token, '<s>', '</s>']:
            clean_tokens.append(token)
            # Handle multi-dimensional attributions by taking the L2 norm
            if isinstance(attr, (np.ndarray, torch.Tensor)):
                if hasattr(attr, 'detach'):
                    attr = attr.detach().cpu().numpy()
                # Take L2 norm across embedding dimensions to get single attribution score
                attr_score = float(np.linalg.norm(attr))
            else:
                attr_score = float(attr)
            clean_attributions.append(attr_score)

    # Map attributions for each custom token
    word_attributions = []

    for custom_token in custom_tokens:
        total_attr = 0.0
        count = 0

        for model_token, attr in zip(clean_tokens, clean_attributions):
            # Remove subword markers and compare
            clean_model_token = model_token.replace('##', '').replace('Ġ', '').lower()
            custom_lower = custom_token.lower()

            # Simple matching - if tokens are similar
            if (clean_model_token in custom_lower or
                    custom_lower in clean_model_token or
                    clean_model_token == custom_lower):
                total_attr += attr
                count += 1

        # Average or use zero if no match
        if count > 0:
            word_attributions.append(total_attr / count)
        else:
            word_attributions.append(0.0)

    return word_attributions


def focused_integrated_gradients_analysis(sentences, model_path):
    """
    Integrated Gradients analysis focused on sampled sentences with clean comparison output
    """
    print("=== FOCUSED INTEGRATED GRADIENTS ANALYSIS ===")
    print("Loading model...")

    try:
        # Load wrapped model
        model_wrapper = ManipulationModelWrapper(model_path)
        tokenizer = model_wrapper.tokenizer
        device = model_wrapper.device
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None, None

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model_wrapper)

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
            # Prepare inputs
            encoded = prepare_model_inputs(text, tokenizer)
            input_ids = encoded['input_ids'].to(device)
            attention_mask = encoded['attention_mask'].to(device)

            # Get embeddings with requires_grad=True
            embeddings = model_wrapper.embed_input(input_ids)
            embeddings.requires_grad_()

            # Baseline embeddings (embedding of pad tokens)
            baseline_input_ids = torch.full_like(input_ids, tokenizer.pad_token_id)
            baseline_embeddings = model_wrapper.embed_input(baseline_input_ids)

            # Compute attributions on embeddings
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

            # Use same tokenization as other methods for consistency
            custom_tokens = custom_tokenizer(text)

            if not custom_tokens:
                print(f"No valid tokens found for {sentence_id}")
                continue

            # Map model tokens to custom tokens
            mapped_attributions = map_attributions_to_custom_tokens(
                tokens, token_attributions, custom_tokens, tokenizer
            )

            # Calculate summary statistics
            if mapped_attributions:
                max_attribution = max(mapped_attributions)
                min_attribution = min(mapped_attributions)
                total_positive = sum([val for val in mapped_attributions if val > 0])
                total_negative = sum([val for val in mapped_attributions if val < 0])

                # Find top contributing tokens
                token_contributions = list(zip(custom_tokens, mapped_attributions))
                token_contributions.sort(key=lambda x: x[1], reverse=True)

                top_positive = [t for t in token_contributions if t[1] > 0][:3]
                top_negative = [t for t in token_contributions if t[1] < 0][:3]
            else:
                max_attribution = min_attribution = total_positive = total_negative = 0
                top_positive = top_negative = []

            # Print results
            prediction_status = "✓ CORRECT" if true_label == predicted_label else "✗ INCORRECT"
            print(f"Prediction: {prediction_status}")
            print(f"True Label: {true_label} | Predicted: {predicted_label} | Prob: {prediction_score:.3f}")
            print(f"Max Grad: {max_attribution:.4f} | Min Grad: {min_attribution:.4f}")

            if top_positive:
                pos_tokens_str = ', '.join([f"{token}({attr:.3f})" for token, attr in top_positive])
                print(f"Top Positive: {pos_tokens_str}")
            if top_negative:
                neg_tokens_str = ', '.join([f"{token}({attr:.3f})" for token, attr in top_negative])
                print(f"Top Negative: {neg_tokens_str}")

            print()  # Blank line for readability

            # Store detailed results
            sentence_summaries.append({
                'sentence_id': sentence_id,
                'category': category,
                'text': text,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'manipulation_prob': prediction_score,
                'max_gradient': max_attribution,
                'min_gradient': min_attribution,
                'total_positive': total_positive,
                'total_negative': total_negative,
                'convergence_delta': float(convergence_delta.item()),
                'top_positive_tokens': [f"{token}({attr:.3f})" for token, attr in top_positive],
                'top_negative_tokens': [f"{token}({attr:.3f})" for token, attr in top_negative],
                'all_tokens': custom_tokens,
                'all_gradients': mapped_attributions
            })

            # Store token-level results
            for token, attribution in zip(custom_tokens, mapped_attributions):
                if token.strip():
                    results.append({
                        'sentence_id': sentence_id,
                        'category': category,
                        'token': token,
                        'gradient_value': attribution,
                        'text': text,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'manipulation_prob': prediction_score
                    })

        except Exception as e:
            print(f"✗ Error processing {sentence_id}: {e}")
            continue

    return pd.DataFrame(results), pd.DataFrame(sentence_summaries)


def print_gradients_comparison_summary(sentence_summaries):
    """
    Print a clean comparison summary across all sentences for integrated gradients analysis
    """
    print("=== INTEGRATED GRADIENTS COMPARISON SUMMARY ===")

    # Group by category
    categories = sentence_summaries['category'].unique()

    for category in sorted(categories):
        category_data = sentence_summaries[sentence_summaries['category'] == category]
        print(f"\n--- {category} ---")

        for _, row in category_data.iterrows():
            prediction_icon = "✓" if row['true_label'] == row['predicted_label'] else "✗"
            print(
                f"{row['sentence_id']}: {prediction_icon} Prob:{row['manipulation_prob']:.3f} MaxGrad:{row['max_gradient']:.4f}")
            print(f"  Top+: {', '.join(row['top_positive_tokens'][:2])}")
            if row['top_negative_tokens']:
                print(f"  Top-: {', '.join(row['top_negative_tokens'][:2])}")

    print("\n=== KEY INSIGHTS ===")

    # Calculate summary statistics
    correct_predictions = sentence_summaries[sentence_summaries['true_label'] == sentence_summaries['predicted_label']]
    incorrect_predictions = sentence_summaries[
        sentence_summaries['true_label'] != sentence_summaries['predicted_label']]

    print(f"Correct Predictions: {len(correct_predictions)}/{len(sentence_summaries)}")
    print(f"Average Max Gradient (Correct): {correct_predictions['max_gradient'].mean():.4f}")
    if len(incorrect_predictions) > 0:
        print(f"Average Max Gradient (Incorrect): {incorrect_predictions['max_gradient'].mean():.4f}")

    # Most important tokens by gradient magnitude
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

        print("Token (avg_gradient, frequency):")
        for _, row in top_tokens.head(8).iterrows():
            print(f"  {row['token']}: {row['mean']:.3f} ({row['count']}x)")


def main():
    """
    Main execution function for Integrated Gradients analysis with dataset sampling.
    """
    print("INTEGRATED GRADIENTS ANALYSIS WITH DATASET SAMPLING")
    print("=" * 52)

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

    # Run Integrated Gradients analysis
    print(f"\nRunning Integrated Gradients analysis on {len(selected_sentences)} sentences...")
    results_df, summaries_df = focused_integrated_gradients_analysis(selected_sentences, model_path)

    if results_df is not None and len(results_df) > 0:
        # Print comparison summary
        print_gradients_comparison_summary(summaries_df)

        # Save results with timestamp
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        token_results_file = f'integrated_gradients_token_results_extended.csv'
        summaries_file = f'integrated_gradients_sentence_summaries_extended.csv'

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