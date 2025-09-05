import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import warnings
import json
from datetime import datetime
import re
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ===== CRITICAL PATHS - UPDATE THESE =====
model_path = "/LLM_training/dialogue_level_training/dialogue_models/dialogue_model_exp"
# ADD YOUR DIALOGUE DATASET PATH HERE
dialogue_dataset_path = "/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/data/dialoguelevel_mentalmanip_detailed.xlsx"  # Update this path

# ===== SAMPLING CONFIGURATION =====
SAMPLE_SIZE = 50  # Number of dialogues to analyze (adjust as needed)
STRATIFIED_SAMPLING = True  # Whether to ensure balanced sampling across labels
RANDOM_SEED = 42  # For reproducible results


def load_dialogue_dataset(dataset_path):
    """
    Load and prepare the dialogue dataset for sampling.
    Handles both CSV and Excel files automatically.
    """
    try:
        # Determine file type and load accordingly
        if dataset_path.endswith('.xlsx') or dataset_path.endswith('.xls'):
            print("Loading Excel file...")
            df = pd.read_excel(dataset_path)
            print("✓ Successfully loaded Excel file")
        elif dataset_path.endswith('.csv'):
            print("Loading CSV file...")
            # Try different encodings for CSV
            encodings_to_try = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']

            df = None
            for encoding in encodings_to_try:
                try:
                    print(f"Trying encoding: {encoding}")
                    df = pd.read_csv(dataset_path, encoding=encoding)
                    print(f"✓ Successfully loaded with encoding: {encoding}")
                    break
                except Exception as e:
                    print(f"✗ Failed with encoding {encoding}: {e}")
                    continue

            if df is None:
                print("Failed to load CSV file with any encoding")
                return None
        else:
            print("Unsupported file format. Please use .xlsx, .xls, or .csv files")
            return None

        print(f"Dataset loaded with columns: {df.columns.tolist()}")
        print(f"Total rows: {len(df)}")

        # Common possible column names - adjust based on your dataset structure
        text_column_options = ['dialogue', 'text', 'content', 'conversation', 'Text', 'Dialogue', 'Content']
        label_column_options = ['Manipulative', 'label', 'target', 'manipulation', 'class', 'is_manipulative', 'Label', 'Target',
                                'Manipulation']

        # Try to identify dialogue text column
        text_column = None
        for col in text_column_options:
            if col in df.columns:
                text_column = col
                break

        if text_column is None:
            print("Could not find text column automatically.")
            print("Available columns:", df.columns.tolist())
            print("Please specify which column contains the dialogue text.")
            return None

        # Try to identify label column
        label_column = None
        for col in label_column_options:
            if col in df.columns:
                label_column = col
                break

        if label_column is None:
            print("Could not find label column automatically.")
            print("Available columns:", df.columns.tolist())
            print("Please specify which column contains the labels.")
            return None

        print(f"✓ Using text column: '{text_column}'")
        print(f"✓ Using label column: '{label_column}'")

        # Clean the dataset
        original_length = len(df)
        df = df.dropna(subset=[text_column, label_column])
        df = df[df[text_column].astype(str).str.strip() != '']

        print(f"Cleaned dataset: {len(df)} dialogues (removed {original_length - len(df)} empty/invalid rows)")

        # Standardize column names
        df = df.rename(columns={text_column: 'text', label_column: 'label'})

        # Ensure labels are 0/1
        unique_labels = df['label'].unique()
        print(f"Unique labels found: {unique_labels}")

        # Handle different label formats
        if set(unique_labels) == {0, 1}:
            print("✓ Labels already in 0/1 format")
        elif set(unique_labels) == {0.0, 1.0}:
            print("✓ Converting float labels to int")
            df['label'] = df['label'].astype(int)
        else:
            print("Converting labels to 0/1 format...")
            if df['label'].dtype == 'object':
                # Handle string labels
                unique_labels = sorted(unique_labels)
                label_mapping = {unique_labels[0]: 0, unique_labels[1]: 1}
                df['label'] = df['label'].map(label_mapping)
                print(f"✓ Label mapping applied: {label_mapping}")
            else:
                # Handle other numeric labels
                df['label'] = (df['label'] != 0).astype(int)
                print("✓ Converted to binary 0/1 labels")

        # Final validation
        df = df.dropna(subset=['label'])
        df = df[df['label'].isin([0, 1])]

        print(f"Final dataset: {len(df)} dialogues")
        label_counts = df['label'].value_counts().sort_index()
        print(f"Label distribution:")
        print(f"  Non-Manipulative (0): {label_counts.get(0, 0)}")
        print(f"  Manipulative (1): {label_counts.get(1, 0)}")

        return df

    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check:")
        print("1. File path is correct")
        print("2. File is a valid Excel (.xlsx/.xls) or CSV file")
        print("3. File is not corrupted or password-protected")
        return None


def sample_dialogues_from_dataset(df, sample_size=50, stratified=True, random_seed=42):
    """
    Sample dialogues from the dataset for analysis.
    """
    np.random.seed(random_seed)

    if stratified:
        # Stratified sampling to ensure balanced representation
        sampled_dialogues = []

        for label in df['label'].unique():
            label_data = df[df['label'] == label]
            n_samples = min(sample_size // 2, len(label_data))  # Half samples per label

            sampled_label_data = label_data.sample(n=n_samples, random_state=random_seed)
            sampled_dialogues.append(sampled_label_data)

        sampled_df = pd.concat(sampled_dialogues, ignore_index=True)

    else:
        # Random sampling
        sampled_df = df.sample(n=min(sample_size, len(df)), random_state=random_seed)

    # Add dialogue_id for consistent referencing
    sampled_df = sampled_df.reset_index(drop=True)
    sampled_df['dialogue_id'] = sampled_df.index

    # Create category labels for analysis
    sampled_df['category'] = sampled_df['label'].apply(
        lambda x: 'manipulative' if x == 1 else 'non_manipulative'
    )

    print(f"Sampled {len(sampled_df)} dialogues:")
    sampled_counts = sampled_df['label'].value_counts()
    print(f"  - Non-Manipulative (0): {sampled_counts.get(0, 0)}")
    print(f"  - Manipulative (1): {sampled_counts.get(1, 0)}")

    return sampled_df


# ===== YOUR ORIGINAL EXPECTED GRADIENTS CLASS (UNCHANGED) =====
class DialogueExpectedGradientsExplainer:
    """
    Expected Gradients implementation for dialogue-level manipulation detection.

    Expected Gradients improves upon Integrated Gradients by using multiple
    reference samples from the data distribution instead of a single baseline.
    This version extends to dialogue analysis with sentence-level aggregation.
    """

    def __init__(self, model, tokenizer, device=None, max_length=512):
        """
        Initialize the Expected Gradients explainer for dialogues.

        Args:
            model: Your trained dialogue-level DistilRoBERTa model
            tokenizer: The tokenizer used for your model
            device: Device to run computations on
            max_length: Maximum sequence length for dialogues (increased from 64)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.model.to(self.device)
        self.model.eval()

        # Enable gradients for embeddings
        self.model.requires_grad_(True)

    def get_sentence_boundaries_eg(self, text: str, tokens: List[str]) -> List[Dict]:
        """
        Map token positions to sentence boundaries for dialogue analysis.

        Args:
            text: Original dialogue text
            tokens: List of tokenized tokens

        Returns:
            List of sentence boundary dictionaries
        """
        # Split dialogue into sentences
        sentences = re.split(r'[.!?]+\s*', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        sentence_boundaries = []
        current_token_idx = 0

        for sentence in sentences:
            if not sentence:
                continue

            # Estimate token count for this sentence (accounting for subword tokenization)
            sentence_word_count = len(re.findall(r'\b\w+\b', sentence))
            estimated_tokens = int(sentence_word_count * 1.4)  # Account for subword splits

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

    def _prepare_reference_samples(self, reference_texts: List[str], n_samples: int = 20) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Prepare reference dialogue samples for Expected Gradients.

        Args:
            reference_texts: List of reference dialogues from your dataset
            n_samples: Number of reference samples to use (reduced for dialogues)

        Returns:
            Tuple of (reference embeddings, reference attention masks)
        """
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

    def _get_gradients(self, input_embeds: torch.Tensor, attention_mask: torch.Tensor,
                       target_class: int) -> torch.Tensor:
        """
        Compute gradients of the model output with respect to input embeddings.

        Args:
            input_embeds: Input embeddings
            attention_mask: Attention mask
            target_class: Target class for gradient computation

        Returns:
            Gradients with respect to input embeddings
        """
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

    def explain_dialogue(self, dialogue_text: str, reference_texts: List[str],
                         target_class: Optional[int] = None, n_steps: int = 15,
                         n_reference_samples: int = 20) -> Dict:
        """
        Generate Expected Gradients explanation for a dialogue with sentence aggregation.

        Args:
            dialogue_text: Dialogue text to explain
            reference_texts: Reference dialogues from your dataset
            target_class: Target class (0 or 1). If None, uses predicted class
            n_steps: Number of integration steps (reduced for efficiency)
            n_reference_samples: Number of reference samples to use

        Returns:
            Dictionary containing token-level and sentence-level attributions
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

        # Get model prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(outputs.logits, dim=-1).item()
            confidence = probabilities[0, predicted_class].item()

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

        for ref_idx, (ref_embeds, ref_mask) in enumerate(zip(reference_embeds, reference_masks)):
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

        # Filter valid tokens (remove padding)
        valid_length = attention_mask.sum().item()
        tokens = tokens[:valid_length]
        token_attributions = token_attributions[:valid_length]

        # Get sentence boundaries
        sentence_boundaries = self.get_sentence_boundaries_eg(dialogue_text, tokens)

        # Aggregate attributions by sentences
        sentence_results = []

        for sent_idx, sent_info in enumerate(sentence_boundaries):
            start_idx = sent_info['start_idx']
            end_idx = min(sent_info['end_idx'], len(token_attributions))

            if start_idx < len(token_attributions) and end_idx > start_idx:
                # Get attributions for tokens in this sentence
                sentence_token_attributions = token_attributions[start_idx:end_idx]
                sentence_tokens = tokens[start_idx:end_idx]

                # Aggregate attributions for this sentence
                sentence_contribution = np.sum(sentence_token_attributions)
                avg_contribution = np.mean(sentence_token_attributions)
                max_contribution = np.max(sentence_token_attributions)
                min_contribution = np.min(sentence_token_attributions)

                # Find top contributing tokens in this sentence
                token_attr_pairs = list(zip(sentence_tokens, sentence_token_attributions))
                token_attr_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
                top_tokens = token_attr_pairs[:5]

                sentence_result = {
                    'sentence_id': sent_idx,
                    'sentence': sent_info['sentence'],
                    'sentence_contribution': sentence_contribution,
                    'avg_contribution': avg_contribution,
                    'max_contribution': max_contribution,
                    'min_contribution': min_contribution,
                    'num_tokens': len(sentence_token_attributions),
                    'tokens': sentence_tokens,
                    'token_attributions': sentence_token_attributions.tolist(),
                    'top_tokens': [
                        {'token': token, 'attribution': float(attr)}
                        for token, attr in top_tokens
                    ]
                }

                sentence_results.append(sentence_result)

        # Create comprehensive results
        results = {
            'dialogue_text': dialogue_text,
            'tokens': tokens,
            'token_attributions': token_attributions,
            'predicted_class': predicted_class,
            'target_class': target_class,
            'confidence': confidence,
            'probabilities': probabilities.squeeze().detach().cpu().numpy(),
            'sentence_boundaries': sentence_boundaries,
            'sentence_results': sentence_results,
            'n_reference_samples': n_reference_samples,
            'n_steps': n_steps,
            'input_ids': input_ids.squeeze().cpu().numpy()[:valid_length],
            'attention_mask': attention_mask.squeeze().cpu().numpy()[:valid_length]
        }

        return results


# ===== MODEL LOADING FUNCTION =====
def load_your_dialogue_model(model_path: str, device=None):
    """Load your trained dialogue model for Expected Gradients analysis"""
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            num_labels=2,
            id2label={0: "Non-Manipulative", 1: "Manipulative"},
            label2id={"Non-Manipulative": 0, "Manipulative": 1}
        )
        print(f"✓ Successfully loaded dialogue model from: {model_path}")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        raise

    model.to(device)
    model.eval()

    return model, tokenizer, device


# ===== EXTENDED ANALYSIS FUNCTION =====
def analyze_dialogue_expected_gradients_extended():
    """
    Analyze dialogue-level manipulation using Expected Gradients on extended dataset.
    """
    print("=== EXTENDED EXPECTED GRADIENTS DIALOGUE ANALYSIS ===")

    # Load dialogue dataset
    print("Loading dialogue dataset...")
    df = load_dialogue_dataset(dialogue_dataset_path)

    if df is None:
        print("Failed to load dialogue dataset. Please check the file path and column names.")
        return None, None, None

    # Sample dialogues from dataset
    print(f"\nSampling {SAMPLE_SIZE} dialogues from dataset...")
    sampled_data = sample_dialogues_from_dataset(
        df,
        sample_size=SAMPLE_SIZE,
        stratified=STRATIFIED_SAMPLING,
        random_seed=RANDOM_SEED
    )

    # Save sampled dialogues for consistency across methods
    sampled_data.to_csv('sampled_dialogues.csv', index=False)
    print(f"✓ Sampled dialogues saved to 'sampled_dialogues.csv' for cross-method consistency")

    # Load model
    print("\nLoading dialogue model...")
    model, tokenizer, device = load_your_dialogue_model(model_path)

    # Initialize explainer
    explainer = DialogueExpectedGradientsExplainer(model, tokenizer, device, max_length=512)

    # Use sampled dialogues as reference samples
    reference_dialogues = sampled_data['text'].tolist()

    # Results storage
    token_results = []
    sentence_results = []
    dialogue_summaries = []

    print(f"\nAnalyzing {len(sampled_data)} dialogues with Expected Gradients...")

    # Analyze each dialogue
    for index, row in tqdm(sampled_data.iterrows(), total=len(sampled_data), desc="Processing dialogues"):
        dialogue_text = row['text']
        label = row['label']
        dialogue_id = row['dialogue_id']

        try:
            # Get Expected Gradients explanation
            explanation = explainer.explain_dialogue(
                dialogue_text=dialogue_text,
                reference_texts=reference_dialogues,
                n_steps=10,  # Reduced for efficiency
                n_reference_samples=15
            )

            predicted_label = explanation['predicted_class']
            manipulation_prob = explanation['confidence']

            # Process sentence results
            for sent_result in explanation['sentence_results']:
                # Convert to standardized format
                sentence_info = {
                    'dialogue_id': dialogue_id,
                    'sentence_id': sent_result['sentence_id'],
                    'sentence': sent_result['sentence'],
                    'sentence_contribution': sent_result['sentence_contribution'],
                    'positive_contribution': sum([attr for attr in sent_result['token_attributions'] if attr > 0]),
                    'negative_contribution': sum([attr for attr in sent_result['token_attributions'] if attr < 0]),
                    'num_tokens': sent_result['num_tokens'],
                    'avg_token_contribution': sent_result['avg_contribution'],
                    'top_tokens': [f"{token['token']}({token['attribution']:.3f})" for token in
                                   sent_result['top_tokens']],
                    'dialogue_label': label,
                    'predicted_label': predicted_label,
                    'manipulation_prob': manipulation_prob,
                    'category': row['category']
                }
                sentence_results.append(sentence_info)

                # Store individual token results
                for i, (token, attribution) in enumerate(zip(sent_result['tokens'], sent_result['token_attributions'])):
                    token_results.append({
                        'dialogue_id': dialogue_id,
                        'sentence_id': sent_result['sentence_id'],
                        'token': token,
                        'eg_value': attribution,
                        'sentence': sent_result['sentence'],
                        'dialogue': dialogue_text,
                        'actual_label': label,
                        'predicted_label': predicted_label,
                        'category': row['category']
                    })

            # Create dialogue summary
            dialogue_summary = {
                'dialogue_id': dialogue_id,
                'dialogue': dialogue_text,
                'actual_label': label,
                'predicted_label': predicted_label,
                'manipulation_prob': manipulation_prob,
                'category': row['category'],
                'num_sentences': len(explanation['sentence_results']),
                'total_contribution': sum([s['sentence_contribution'] for s in explanation['sentence_results']])
            }
            dialogue_summaries.append(dialogue_summary)

        except Exception as e:
            print(f"Error analyzing dialogue {dialogue_id}: {e}")
            continue

    return pd.DataFrame(token_results), pd.DataFrame(sentence_results), pd.DataFrame(dialogue_summaries)


# ===== ANALYSIS FUNCTIONS =====
def analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries):
    """Analyze patterns across the extended dialogue set"""
    print(f"\n{'=' * 80}")
    print("EXTENDED EXPECTED GRADIENTS DIALOGUE ANALYSIS")
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
    print(f"\nTop 10 Most Influential Sentences (by absolute EG contribution):")
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


# ===== MAIN EXECUTION =====
def main():
    print("=== EXTENDED EXPECTED GRADIENTS DIALOGUE ANALYSIS ===")
    print(f"Analyzing {SAMPLE_SIZE} dialogues with Expected Gradients")

    # Run Extended Expected Gradients analysis
    print("\n1. Running Extended Expected Gradients analysis...")
    try:
        token_results, sentence_results, dialogue_summaries = analyze_dialogue_expected_gradients_extended()

        if token_results is None:
            print("Analysis failed - no results generated")
            return

        print(f"Expected Gradients analysis complete!")
        print(f"- Token-level results: {len(token_results)}")
        print(f"- Sentence-level results: {len(sentence_results)}")
        print(f"- Dialogue summaries: {len(dialogue_summaries)}")

        # Enhanced analysis
        if len(sentence_results) > 0:
            analyze_dialogue_patterns_extended(sentence_results, dialogue_summaries)

    except Exception as e:
        print(f"Expected Gradients analysis failed: {e}")
        return

    # Save results
    print("\n2. Saving extended results...")
    token_results.to_csv('expected_gradients_dialogue_tokens_extended.csv', index=False)
    sentence_results.to_csv('expected_gradients_dialogue_sentences_extended.csv', index=False)
    dialogue_summaries.to_csv('expected_gradients_dialogue_summaries_extended.csv', index=False)

    print("Files saved:")
    print("- expected_gradients_dialogue_tokens_extended.csv")
    print("- expected_gradients_dialogue_sentences_extended.csv")
    print("- expected_gradients_dialogue_summaries_extended.csv")
    print("- sampled_dialogues.csv (for cross-method consistency)")
    print("\nResults ready for extended analysis and cross-method comparison!")


if __name__ == "__main__":
    main()