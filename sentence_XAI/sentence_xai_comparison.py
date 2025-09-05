import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import warnings
import glob
from datetime import datetime

warnings.filterwarnings('ignore')


class StreamlinedXAIComparison:
    def __init__(self):
        self.methods_config = {
            'SHAP': {
                'token_file': 'shap_token_results_extended.csv',
                'summary_file': 'shap_sentence_summaries_extended.csv',
                'score_col': 'shap_value',
                'summary_score_col': 'net_shap'
            },
            'LIME': {
                'token_file': 'lime_token_results_extended.csv',
                'summary_file': 'lime_sentence_summaries_extended.csv',
                'score_col': 'lime_value',
                'summary_score_col': 'net_lime'
            },
            'Token_Occlusion': {
                'token_file': 'token_occlusion_token_results_extended.csv',
                'summary_file': 'token_occlusion_sentence_summaries_extended.csv',
                'score_col': 'occlusion_impact',
                'summary_score_col': 'max_impact'
            },
            'Integrated_Gradients': {
                'token_file': 'integrated_gradients_token_results_extended.csv',
                'summary_file': 'integrated_gradients_sentence_summaries_extended.csv',
                'score_col': 'gradient_value',
                'summary_score_col': 'max_gradient'
            },
            'Expected_Gradients': {
                'token_file': 'expected_gradients_token_results_extended.csv',
                'summary_file': 'expected_gradients_sentence_summaries_extended.csv',
                'score_col': 'expected_gradient_value',
                'summary_score_col': 'max_expected_gradient'
            },
            'Raw_Attention': {
                'token_file': 'raw_attention_token_results_extended.csv',
                'summary_file': 'raw_attention_sentence_summaries_extended.csv',
                'score_col': 'attention_value',
                'summary_score_col': 'max_attention'
            }
        }

        self.token_data = {}
        self.summary_data = {}

    def load_data(self):
        """Load all XAI method results"""
        print("Loading XAI method data...")

        for method, config in self.methods_config.items():
            try:
                # Load token-level data
                self.token_data[method] = pd.read_csv(config['token_file'])

                # Load sentence-level summaries
                self.summary_data[method] = pd.read_csv(config['summary_file'])

                print(f"✓ {method}: {len(self.token_data[method])} tokens, {len(self.summary_data[method])} sentences")

            except FileNotFoundError as e:
                print(f"✗ {method}: File not found - {e}")

        print(f"\nLoaded {len(self.token_data)} methods successfully")
        print(f"Total sentences in dataset: {len(self.get_sentence_ids())}\n")

    def get_sentence_ids(self):
        """Get all unique sentence IDs"""
        if not self.token_data:
            return []

        first_method = next(iter(self.token_data.keys()))
        return sorted(self.token_data[first_method]['sentence_id'].unique())

    def get_top_tokens(self, method, sentence_id, top_k=5, by_absolute=False):
        """Get top-k tokens for a method and sentence"""
        if method not in self.token_data:
            return []

        df = self.token_data[method]
        sentence_data = df[df['sentence_id'] == sentence_id].copy()

        if sentence_data.empty:
            return []

        score_col = self.methods_config[method]['score_col']

        if by_absolute:
            sentence_data['abs_score'] = sentence_data[score_col].abs()
            top_tokens = sentence_data.nlargest(top_k, 'abs_score')
        else:
            top_tokens = sentence_data.nlargest(top_k, score_col)

        return top_tokens['token'].tolist()

    def calculate_token_overlap(self, sentence_id, top_k=5, by_absolute=False):
        """Calculate pairwise token overlap for a sentence"""
        methods = list(self.token_data.keys())
        n_methods = len(methods)

        # Get top tokens for each method
        top_tokens = {}
        for method in methods:
            top_tokens[method] = self.get_top_tokens(method, sentence_id, top_k, by_absolute)

        # Calculate overlap matrix
        overlap_matrix = np.zeros((n_methods, n_methods))

        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i == j:
                    overlap_matrix[i][j] = 1.0
                else:
                    set1 = set(top_tokens[method1])
                    set2 = set(top_tokens[method2])
                    if len(set1) > 0 and len(set2) > 0:
                        overlap = len(set1.intersection(set2)) / max(len(set1), len(set2))
                        overlap_matrix[i][j] = overlap

        return overlap_matrix, methods

    def calculate_aggregate_overlap_statistics(self, top_k=3):
        """Calculate aggregate overlap statistics across all sentences"""
        sentence_ids = self.get_sentence_ids()
        methods = list(self.token_data.keys())
        n_methods = len(methods)

        print(f"Calculating aggregate overlap statistics for {len(sentence_ids)} sentences...")

        # Store all pairwise overlaps
        all_overlaps = defaultdict(list)
        overall_overlaps = []

        for sentence_id in sentence_ids:
            overlap_matrix, _ = self.calculate_token_overlap(sentence_id, top_k)

            # Store pairwise overlaps
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i != j:
                        pair_key = f"{method1}_vs_{method2}"
                        all_overlaps[pair_key].append(overlap_matrix[i][j])

            # Calculate average overlap for this sentence (excluding diagonal)
            mask = np.ones(overlap_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_overlap = np.mean(overlap_matrix[mask])
            overall_overlaps.append(avg_overlap)

        # Calculate summary statistics
        overlap_summary = {
            'overall_mean': np.mean(overall_overlaps),
            'overall_std': np.std(overall_overlaps),
            'overall_median': np.median(overall_overlaps),
            'pairwise_overlaps': {}
        }

        # Calculate pairwise statistics
        for pair, overlaps in all_overlaps.items():
            overlap_summary['pairwise_overlaps'][pair] = {
                'mean': np.mean(overlaps),
                'std': np.std(overlaps),
                'median': np.median(overlaps),
                'min': np.min(overlaps),
                'max': np.max(overlaps)
            }

        return overlap_summary

    def normalize_token(self, token):
        """Normalize tokens to handle different tokenization schemes"""
        # Handle NaN or None values
        if pd.isna(token) or token is None:
            return ""

        # Convert to string if it's not already
        token = str(token)

        # Remove leading/trailing whitespace
        token = token.strip()

        # Handle special tokenization patterns
        if token.startswith(' '):
            token = token[1:]

        return token

    def get_normalized_token_data(self, method, sentence_id):
        """Get token data with normalized tokens"""
        df = self.token_data[method]
        sentence_data = df[df['sentence_id'] == sentence_id].copy()

        if sentence_data.empty:
            return {}

        score_col = self.methods_config[method]['score_col']

        # Normalize tokens and aggregate scores for duplicates
        normalized_data = {}
        for _, row in sentence_data.iterrows():
            # Skip rows with NaN tokens or scores
            if pd.isna(row['token']) or pd.isna(row[score_col]):
                continue

            normalized_token = self.normalize_token(row['token'])
            score = row[score_col]

            # Skip empty tokens after normalization
            if not normalized_token:
                continue

            if normalized_token in normalized_data:
                # If token appears multiple times, take the maximum absolute score
                if abs(score) > abs(normalized_data[normalized_token]):
                    normalized_data[normalized_token] = score
            else:
                normalized_data[normalized_token] = score

        return normalized_data

    def compute_cosine_similarity(self, vector1, vector2):
        """Compute cosine similarity between two vectors"""
        if len(vector1) == 0 or len(vector2) == 0:
            return np.nan
        try:
            similarity = cosine_similarity(vector1.reshape(1, -1), vector2.reshape(1, -1))[0][0]
            return similarity
        except:
            return np.nan

    def compute_pearson_correlation(self, vector1, vector2):
        """Compute Pearson correlation between two vectors"""
        if len(vector1) < 2 or len(vector2) < 2:
            return np.nan
        try:
            correlation, _ = pearsonr(vector1, vector2)
            return correlation if not np.isnan(correlation) else np.nan
        except:
            return np.nan

    def to_probability_distribution(self, values):
        """Convert values to probability distribution for Jensen-Shannon divergence"""
        # Handle negative values by shifting
        min_val = np.min(values)
        if min_val < 0:
            values = values + abs(min_val)

        # Convert to probability distribution
        total = np.sum(values)
        if total > 0:
            values = values / total
        else:
            values = np.ones_like(values) / len(values)

        return values

    def compute_js_divergence(self, vector1, vector2):
        """Compute Jensen-Shannon divergence between two vectors"""
        if len(vector1) == 0 or len(vector2) == 0:
            return np.nan
        try:
            prob1 = self.to_probability_distribution(vector1.copy())
            prob2 = self.to_probability_distribution(vector2.copy())

            # Ensure no zero values for JS divergence
            epsilon = 1e-10
            prob1 = prob1 + epsilon
            prob2 = prob2 + epsilon
            prob1 = prob1 / np.sum(prob1)
            prob2 = prob2 / np.sum(prob2)

            js_div = jensenshannon(prob1, prob2)
            return js_div if not np.isnan(js_div) else np.nan
        except:
            return np.nan

    def calculate_pairwise_divergences_normalized(self, sentence_id):
        """Calculate divergences with normalized tokenization"""
        methods = list(self.token_data.keys())

        # Get normalized token data for each method
        method_data = {}
        for method in methods:
            method_data[method] = self.get_normalized_token_data(method, sentence_id)

        # Initialize results dictionary
        divergences = {}

        # Calculate pairwise divergences
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:  # Don't compare method with itself
                    pair_key = f"{method1}_vs_{method2}"

                    # Find common tokens between this pair
                    tokens1 = set(method_data[method1].keys())
                    tokens2 = set(method_data[method2].keys())
                    common_tokens = tokens1.intersection(tokens2)

                    if len(common_tokens) >= 3:  # Need at least 3 common tokens
                        # Get aligned score vectors for common tokens
                        sorted_tokens = sorted(common_tokens)
                        scores1 = np.array([method_data[method1][token] for token in sorted_tokens])
                        scores2 = np.array([method_data[method2][token] for token in sorted_tokens])

                        # Calculate divergence measures
                        cosine_sim = self.compute_cosine_similarity(scores1, scores2)
                        pearson_corr = self.compute_pearson_correlation(scores1, scores2)
                        js_div = self.compute_js_divergence(scores1, scores2)

                        divergences[pair_key] = {
                            'cosine_similarity': cosine_sim,
                            'pearson_correlation': pearson_corr,
                            'jensen_shannon_divergence': js_div,
                            'n_tokens': len(common_tokens)
                        }

        return divergences

    def analyze_method_divergences_across_sentences(self):
        """Analyze divergence measures across all sentences"""
        sentence_ids = self.get_sentence_ids()
        all_divergences = defaultdict(lambda: defaultdict(list))

        print(f"Analyzing divergences across {len(sentence_ids)} sentences...")

        valid_sentences = 0
        for sentence_id in sentence_ids:
            divergences = self.calculate_pairwise_divergences_normalized(sentence_id)

            if divergences:
                valid_sentences += 1
                for pair, measures in divergences.items():
                    for measure_name, value in measures.items():
                        if measure_name != 'n_tokens' and not np.isnan(value):
                            all_divergences[pair][measure_name].append(value)

        print(f"Found valid divergences for {valid_sentences}/{len(sentence_ids)} sentences")

        # Calculate summary statistics
        summary_divergences = {}
        for pair, measures in all_divergences.items():
            summary_divergences[pair] = {}
            for measure_name, values in measures.items():
                if len(values) > 0:
                    summary_divergences[pair][measure_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'n_sentences': len(values)
                    }

        return summary_divergences

    def analyze_by_prediction_quality(self, top_k=3):
        """Analyze method agreement by prediction correctness"""
        sentence_ids = self.get_sentence_ids()

        correct_overlaps = []
        incorrect_overlaps = []

        for sentence_id in sentence_ids:
            # Get prediction correctness from first available method
            first_method = next(iter(self.summary_data.keys()))
            sentence_info = self.summary_data[first_method][
                self.summary_data[first_method]['sentence_id'] == sentence_id
                ]

            if sentence_info.empty:
                continue

            is_correct = sentence_info['true_label'].iloc[0] == sentence_info['predicted_label'].iloc[0]

            # Calculate overlap for this sentence
            overlap_matrix, methods = self.calculate_token_overlap(sentence_id, top_k)

            # Average overlap (excluding diagonal)
            mask = np.ones(overlap_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_overlap = np.mean(overlap_matrix[mask])

            if is_correct:
                correct_overlaps.append(avg_overlap)
            else:
                incorrect_overlaps.append(avg_overlap)

        return {
            'correct_predictions': correct_overlaps,
            'incorrect_predictions': incorrect_overlaps
        }

    def analyze_by_sentence_type(self, top_k=3):
        """Analyze method agreement by sentence type"""
        sentence_ids = self.get_sentence_ids()

        # Get true labels from first available method
        first_method = next(iter(self.summary_data.keys()))
        summary_df = self.summary_data[first_method]

        manip_overlaps = []
        non_manip_overlaps = []

        for sentence_id in sentence_ids:
            sentence_info = summary_df[summary_df['sentence_id'] == sentence_id]

            if sentence_info.empty:
                continue

            is_manipulative = sentence_info['true_label'].iloc[0] == 1

            # Calculate overlap for this sentence
            overlap_matrix, methods = self.calculate_token_overlap(sentence_id, top_k)

            # Average overlap (excluding diagonal)
            mask = np.ones(overlap_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_overlap = np.mean(overlap_matrix[mask])

            if is_manipulative:
                manip_overlaps.append(avg_overlap)
            else:
                non_manip_overlaps.append(avg_overlap)

        return {
            'manipulative_sentences': manip_overlaps,
            'non_manipulative_sentences': non_manip_overlaps
        }

    def create_divergence_matrices(self):
        """Create matrices for each divergence measure"""
        divergences = self.analyze_method_divergences_across_sentences()

        # Define custom method order - YOUR PREFERRED ORDER
        custom_order = [
            'SHAP',
            'LIME',
            'Raw_Attention',
            'Integrated_Gradients',
            'Expected_Gradients',
            'Token_Occlusion'
        ]

        # Filter to only include methods that actually exist in your data
        available_methods = list(self.token_data.keys())
        methods = [method for method in custom_order if method in available_methods]

        # Add any methods not in custom_order to the end
        for method in available_methods:
            if method not in methods:
                methods.append(method)

        n_methods = len(methods)

        # Initialize matrices
        cosine_matrix = np.full((n_methods, n_methods), np.nan)
        pearson_matrix = np.full((n_methods, n_methods), np.nan)
        js_matrix = np.full((n_methods, n_methods), np.nan)

        # Fill diagonal with appropriate values
        for i in range(n_methods):
            cosine_matrix[i, i] = 1.0  # Perfect similarity
            pearson_matrix[i, i] = 1.0  # Perfect correlation
            js_matrix[i, i] = 0.0  # No divergence

        # Fill matrices with mean values
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    pair_key = f"{method1}_vs_{method2}"
                    reverse_pair = f"{method2}_vs_{method1}"

                    # Try both directions
                    if pair_key in divergences:
                        data = divergences[pair_key]
                    elif reverse_pair in divergences:
                        data = divergences[reverse_pair]
                    else:
                        continue

                    if 'cosine_similarity' in data:
                        cosine_matrix[i, j] = data['cosine_similarity']['mean']
                    if 'pearson_correlation' in data:
                        pearson_matrix[i, j] = data['pearson_correlation']['mean']
                    if 'jensen_shannon_divergence' in data:
                        js_matrix[i, j] = data['jensen_shannon_divergence']['mean']

        return {
            'cosine_similarity': cosine_matrix,
            'pearson_correlation': pearson_matrix,
            'jensen_shannon_divergence': js_matrix,
            'methods': methods
        }

    def create_streamlined_visualizations(self):
        """Create streamlined visualizations focusing on key metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Streamlined XAI Methods Comparison Analysis', fontsize=16, fontweight='bold')

        # 1. Overall Overlap Statistics
        ax1 = axes[0, 0]
        overlap_stats = self.calculate_aggregate_overlap_statistics()

        quality_analysis = self.analyze_by_prediction_quality()
        type_analysis = self.analyze_by_sentence_type()

        categories = ['Overall', 'Correct\nPredictions', 'Incorrect\nPredictions',
                      'Manipulative\nSentences', 'Non-Manipulative\nSentences']

        means = [
            overlap_stats['overall_mean'],
            np.mean(quality_analysis['correct_predictions']) if quality_analysis['correct_predictions'] else 0,
            np.mean(quality_analysis['incorrect_predictions']) if quality_analysis['incorrect_predictions'] else 0,
            np.mean(type_analysis['manipulative_sentences']) if type_analysis['manipulative_sentences'] else 0,
            np.mean(type_analysis['non_manipulative_sentences']) if type_analysis['non_manipulative_sentences'] else 0
        ]

        stds = [
            overlap_stats['overall_std'],
            np.std(quality_analysis['correct_predictions']) if quality_analysis['correct_predictions'] else 0,
            np.std(quality_analysis['incorrect_predictions']) if quality_analysis['incorrect_predictions'] else 0,
            np.std(type_analysis['manipulative_sentences']) if type_analysis['manipulative_sentences'] else 0,
            np.std(type_analysis['non_manipulative_sentences']) if type_analysis['non_manipulative_sentences'] else 0
        ]

        bars = ax1.bar(categories, means, yerr=stds, capsize=5, alpha=0.7)
        ax1.set_title('Average Token Overlap by Category')
        ax1.set_ylabel('Average Overlap Score')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                     f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

        matrices = self.create_divergence_matrices()
        methods = matrices['methods']

        # Cosine Similarity Matrix
        ax4 = axes[1, 0]
        cosine_matrix = matrices['cosine_similarity']
        im4 = ax4.imshow(cosine_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax4.set_title('Sentence- Level Cosine Similarity Matrix')
        ax4.set_xticks(range(len(methods)))
        ax4.set_yticks(range(len(methods)))
        ax4.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax4.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # Add values to cells
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(cosine_matrix[i, j]):
                    color = "white" if abs(cosine_matrix[i, j]) > 0.5 else "black"
                    ax4.text(j, i, f'{cosine_matrix[i, j]:.2f}',
                             ha="center", va="center", color=color, fontsize=10)

        # Pearson Correlation Matrix
        ax5 = axes[1, 1]
        pearson_matrix = matrices['pearson_correlation']
        im5 = ax5.imshow(pearson_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax5.set_title('Sentence- Level Pearson Correlation Matrix')
        ax5.set_xticks(range(len(methods)))
        ax5.set_yticks(range(len(methods)))
        ax5.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax5.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

        # Add values to cells
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(pearson_matrix[i, j]):
                    color = "white" if abs(pearson_matrix[i, j]) > 0.5 else "black"
                    ax5.text(j, i, f'{pearson_matrix[i, j]:.2f}',
                             ha="center", va="center", color=color, fontsize=10)

        # Jensen-Shannon Divergence Matrix
        ax6 = axes[1, 2]
        js_matrix = matrices['jensen_shannon_divergence']
        im6 = ax6.imshow(js_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax6.set_title('Sentence- Level Jensen-Shannon Divergence Matrix')
        ax6.set_xticks(range(len(methods)))
        ax6.set_yticks(range(len(methods)))
        ax6.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax6.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)

        # Add values to cells
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(js_matrix[i, j]):
                    color = "white" if js_matrix[i, j] > 0.5 else "black"
                    ax6.text(j, i, f'{js_matrix[i, j]:.2f}',
                             ha="center", va="center", color=color, fontsize=10)

        # 5. Top Method Pairs by Similarity
        ax2 = axes[0, 1]
        divergences = self.analyze_method_divergences_across_sentences()

        # Get top 10 method pairs by cosine similarity
        cosine_pairs = []
        for pair, data in divergences.items():
            if 'cosine_similarity' in data:
                cosine_pairs.append((pair, data['cosine_similarity']['mean']))

        cosine_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = cosine_pairs[:10]

        if top_pairs:
            pair_names = [pair.replace('_vs_', '\nvs\n') for pair, _ in top_pairs]
            similarities = [sim for _, sim in top_pairs]

            bars = ax2.barh(range(len(pair_names)), similarities, alpha=0.7)
            ax2.set_yticks(range(len(pair_names)))
            ax2.set_yticklabels(pair_names, fontsize=8)
            ax2.set_xlabel('Cosine Similarity')
            ax2.set_title('Top 10 Method Pairs by Similarity')
            ax2.grid(axis='x', alpha=0.3)

            # Add value labels
            for i, (bar, sim) in enumerate(zip(bars, similarities)):
                ax2.text(sim + 0.01, i, f'{sim:.3f}', va='center', fontsize=8)

        # 6. Prediction Quality vs Sentence Type Comparison
        ax3 = axes[0, 2]

        # Prepare data for comparison
        comparison_data = [
            quality_analysis['correct_predictions'],
            quality_analysis['incorrect_predictions'],
            type_analysis['manipulative_sentences'],
            type_analysis['non_manipulative_sentences']
        ]

        labels = ['Correct', 'Incorrect', 'Manipulative', 'Non-Manipulative']

        # Remove empty lists
        filtered_data = []
        filtered_labels = []
        for data, label in zip(comparison_data, labels):
            if data:
                filtered_data.append(data)
                filtered_labels.append(label)

        if filtered_data:
            bp = ax3.boxplot(filtered_data, labels=filtered_labels, patch_artist=True)
            ax3.set_title('Method Agreement Distribution')
            ax3.set_ylabel('Average Token Overlap')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(axis='y', alpha=0.3)

            # Color the boxes
            colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

        plt.tight_layout()

        # Save the combined plot (your original functionality)
        plt.savefig('streamlined_xai_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # NEW: Save individual plots for the three matrices
        print("\nSaving individual matrix plots...")

        # 1. Individual Cosine Similarity Matrix
        fig_cosine, ax_cosine = plt.subplots(1, 1, figsize=(8, 6))
        im_cosine = ax_cosine.imshow(cosine_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_cosine.set_title('Sentence-Level Cosine Similarity Matrix', fontsize=14, fontweight='bold')
        ax_cosine.set_xticks(range(len(methods)))
        ax_cosine.set_yticks(range(len(methods)))
        ax_cosine.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax_cosine.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im_cosine, ax=ax_cosine, fraction=0.046, pad=0.04)

        # Add values to cells
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(cosine_matrix[i, j]):
                    color = "white" if abs(cosine_matrix[i, j]) > 0.5 else "black"
                    ax_cosine.text(j, i, f'{cosine_matrix[i, j]:.2f}',
                                   ha="center", va="center", color=color, fontsize=10)

        plt.tight_layout()
        plt.savefig('sentence_cosine_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Individual Pearson Correlation Matrix
        fig_pearson, ax_pearson = plt.subplots(1, 1, figsize=(8, 6))
        im_pearson = ax_pearson.imshow(pearson_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_pearson.set_title('Sentence-Level Pearson Correlation Matrix', fontsize=14, fontweight='bold')
        ax_pearson.set_xticks(range(len(methods)))
        ax_pearson.set_yticks(range(len(methods)))
        ax_pearson.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax_pearson.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im_pearson, ax=ax_pearson, fraction=0.046, pad=0.04)

        # Add values to cells
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(pearson_matrix[i, j]):
                    color = "white" if abs(pearson_matrix[i, j]) > 0.5 else "black"
                    ax_pearson.text(j, i, f'{pearson_matrix[i, j]:.2f}',
                                    ha="center", va="center", color=color, fontsize=10)

        plt.tight_layout()
        plt.savefig('sentence_pearson_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Individual Jensen-Shannon Divergence Matrix
        fig_js, ax_js = plt.subplots(1, 1, figsize=(8, 6))
        im_js = ax_js.imshow(js_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax_js.set_title('Sentence-Level Jensen-Shannon Divergence Matrix', fontsize=14, fontweight='bold')
        ax_js.set_xticks(range(len(methods)))
        ax_js.set_yticks(range(len(methods)))
        ax_js.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax_js.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im_js, ax=ax_js, fraction=0.046, pad=0.04)

        # Add values to cells
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(js_matrix[i, j]):
                    color = "white" if js_matrix[i, j] > 0.5 else "black"
                    ax_js.text(j, i, f'{js_matrix[i, j]:.2f}',
                               ha="center", va="center", color=color, fontsize=10)

        plt.tight_layout()
        plt.savefig('sentence_jensen_shannon_divergence_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Individual matrix plots saved:")
        print("  - sentence_cosine_similarity_matrix.png")
        print("  - sentence_pearson_correlation_matrix.png")
        print("  - sentence_jensen_shannon_divergence_matrix.png")

        return fig, matrices

    def generate_streamlined_report(self):
        """Generate a focused report on key metrics"""
        print("=" * 80)
        print("STREAMLINED XAI METHODS COMPARISON REPORT")
        print("=" * 80)

        sentence_ids = self.get_sentence_ids()

        # Dataset Overview
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   Total sentences analyzed: {len(sentence_ids)}")
        print(f"   Methods compared: {len(self.token_data)}")
        print(f"   Methods: {', '.join(self.token_data.keys())}")

        # Aggregate Overlap Statistics
        print(f"\n2. AGGREGATE TOKEN OVERLAP STATISTICS:")
        overlap_stats = self.calculate_aggregate_overlap_statistics()

        print(f"   Overall average overlap: {overlap_stats['overall_mean']:.3f} ± {overlap_stats['overall_std']:.3f}")
        print(f"   Overall median overlap: {overlap_stats['overall_median']:.3f}")

        # Top and bottom method pairs
        print(f"\n   Top 5 method pairs by overlap:")
        sorted_pairs = sorted(overlap_stats['pairwise_overlaps'].items(),
                              key=lambda x: x[1]['mean'], reverse=True)
        for i, (pair, stats) in enumerate(sorted_pairs[:5]):
            print(f"   {i + 1}. {pair}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        print(f"\n   Bottom 5 method pairs by overlap:")
        for i, (pair, stats) in enumerate(sorted_pairs[-5:]):
            print(f"   {i + 1}. {pair}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        # Prediction Quality Analysis
        print(f"\n3. AGREEMENT BY PREDICTION QUALITY:")
        quality_analysis = self.analyze_by_prediction_quality()

        if quality_analysis['correct_predictions']:
            correct_mean = np.mean(quality_analysis['correct_predictions'])
            correct_std = np.std(quality_analysis['correct_predictions'])
            print(
                f"   Correct predictions: {correct_mean:.3f} ± {correct_std:.3f} (n={len(quality_analysis['correct_predictions'])})")

        if quality_analysis['incorrect_predictions']:
            incorrect_mean = np.mean(quality_analysis['incorrect_predictions'])
            incorrect_std = np.std(quality_analysis['incorrect_predictions'])
            print(
                f"   Incorrect predictions: {incorrect_mean:.3f} ± {incorrect_std:.3f} (n={len(quality_analysis['incorrect_predictions'])})")

        # Sentence Type Analysis
        print(f"\n4. AGREEMENT BY SENTENCE TYPE:")
        type_analysis = self.analyze_by_sentence_type()

        if type_analysis['manipulative_sentences']:
            manip_mean = np.mean(type_analysis['manipulative_sentences'])
            manip_std = np.std(type_analysis['manipulative_sentences'])
            print(
                f"   Manipulative sentences: {manip_mean:.3f} ± {manip_std:.3f} (n={len(type_analysis['manipulative_sentences'])})")

        if type_analysis['non_manipulative_sentences']:
            non_manip_mean = np.mean(type_analysis['non_manipulative_sentences'])
            non_manip_std = np.std(type_analysis['non_manipulative_sentences'])
            print(
                f"   Non-manipulative sentences: {non_manip_mean:.3f} ± {non_manip_std:.3f} (n={len(type_analysis['non_manipulative_sentences'])})")

        # Divergence Analysis
        print(f"\n5. DIVERGENCE ANALYSIS:")
        divergences = self.analyze_method_divergences_across_sentences()

        print(f"   Method pairs with valid divergences: {len(divergences)}")

        # Top correlations
        print(f"\n   Top 5 method pairs by Cosine Similarity:")
        cosine_pairs = [(pair, data['cosine_similarity']['mean'])
                        for pair, data in divergences.items()
                        if 'cosine_similarity' in data]
        cosine_pairs.sort(key=lambda x: x[1], reverse=True)

        for i, (pair, sim) in enumerate(cosine_pairs[:5]):
            n_sentences = divergences[pair]['cosine_similarity']['n_sentences']
            std = divergences[pair]['cosine_similarity']['std']
            print(f"   {i + 1}. {pair}: {sim:.3f} ± {std:.3f} (n={n_sentences})")

        # Top Pearson correlations
        print(f"\n   Top 5 method pairs by Pearson Correlation:")
        pearson_pairs = [(pair, data['pearson_correlation']['mean'])
                         for pair, data in divergences.items()
                         if 'pearson_correlation' in data]
        pearson_pairs.sort(key=lambda x: x[1], reverse=True)

        for i, (pair, corr) in enumerate(pearson_pairs[:5]):
            n_sentences = divergences[pair]['pearson_correlation']['n_sentences']
            std = divergences[pair]['pearson_correlation']['std']
            print(f"   {i + 1}. {pair}: {corr:.3f} ± {std:.3f} (n={n_sentences})")

        # Lowest Jensen-Shannon divergences (most similar)
        print(f"\n   Top 5 method pairs by Jensen-Shannon Divergence (Lower = More Similar):")
        js_pairs = [(pair, data['jensen_shannon_divergence']['mean'])
                    for pair, data in divergences.items()
                    if 'jensen_shannon_divergence' in data]
        js_pairs.sort(key=lambda x: x[1])

        for i, (pair, div) in enumerate(js_pairs[:5]):
            n_sentences = divergences[pair]['jensen_shannon_divergence']['n_sentences']
            std = divergences[pair]['jensen_shannon_divergence']['std']
            print(f"   {i + 1}. {pair}: {div:.3f} ± {std:.3f} (n={n_sentences})")

        # Cross-measure consistency
        print(f"\n6. CROSS-MEASURE CONSISTENCY:")
        consistent_pairs = []
        for pair, data in divergences.items():
            if all(measure in data for measure in
                   ['cosine_similarity', 'pearson_correlation', 'jensen_shannon_divergence']):
                cs_rank = next((i for i, (p, _) in enumerate(cosine_pairs) if p == pair), len(cosine_pairs))
                pc_rank = next((i for i, (p, _) in enumerate(pearson_pairs) if p == pair), len(pearson_pairs))
                js_rank = next((i for i, (p, _) in enumerate(js_pairs) if p == pair), len(js_pairs))

                avg_rank = (cs_rank + pc_rank + js_rank) / 3
                consistent_pairs.append((pair, avg_rank, cs_rank, pc_rank, js_rank))

        consistent_pairs.sort(key=lambda x: x[1])

        print("   Most consistent method pairs across all measures:")
        for i, (pair, avg_rank, cs_rank, pc_rank, js_rank) in enumerate(consistent_pairs[:5]):
            print(f"   {i + 1}. {pair}: avg_rank={avg_rank:.1f} (CS:{cs_rank + 1}, PC:{pc_rank + 1}, JS:{js_rank + 1})")

        # Summary insights
        print(f"\n7. KEY INSIGHTS:")
        if quality_analysis['correct_predictions'] and quality_analysis['incorrect_predictions']:
            if correct_mean > incorrect_mean:
                print(
                    f"   ✓ Methods show higher agreement on correct predictions ({correct_mean:.3f} vs {incorrect_mean:.3f})")
            else:
                print(
                    f"   ✗ Methods show higher agreement on incorrect predictions ({incorrect_mean:.3f} vs {correct_mean:.3f})")

        if type_analysis['manipulative_sentences'] and type_analysis['non_manipulative_sentences']:
            if manip_mean > non_manip_mean:
                print(
                    f"   ✓ Methods show higher agreement on manipulative sentences ({manip_mean:.3f} vs {non_manip_mean:.3f})")
            else:
                print(
                    f"   ✓ Methods show higher agreement on non-manipulative sentences ({non_manip_mean:.3f} vs {manip_mean:.3f})")

        if consistent_pairs:
            most_consistent = consistent_pairs[0][0]
            print(f"   ✓ Most consistent method pair across all measures: {most_consistent}")

        print(f"\n{'=' * 80}")

    def save_summary_statistics(self):
        """Save key statistics to CSV files"""
        print("\nSaving summary statistics...")

        # Save overlap statistics
        overlap_stats = self.calculate_aggregate_overlap_statistics()
        overlap_df = pd.DataFrame([
            {'metric': 'overall_mean', 'value': overlap_stats['overall_mean']},
            {'metric': 'overall_std', 'value': overlap_stats['overall_std']},
            {'metric': 'overall_median', 'value': overlap_stats['overall_median']}
        ])

        # Add pairwise overlaps
        pairwise_rows = []
        for pair, stats in overlap_stats['pairwise_overlaps'].items():
            for stat_name, stat_value in stats.items():
                pairwise_rows.append({
                    'metric': f'{pair}_{stat_name}',
                    'value': stat_value
                })

        pairwise_df = pd.DataFrame(pairwise_rows)
        overlap_summary = pd.concat([overlap_df, pairwise_df], ignore_index=True)
        overlap_summary.to_csv('overlap_statistics_summary_extended.csv', index=False)

        # Save divergence statistics
        divergences = self.analyze_method_divergences_across_sentences()
        divergence_rows = []

        for pair, measures in divergences.items():
            for measure_name, stats in measures.items():
                for stat_name, stat_value in stats.items():
                    divergence_rows.append({
                        'method_pair': pair,
                        'measure': measure_name,
                        'statistic': stat_name,
                        'value': stat_value
                    })

        if divergence_rows:
            divergence_df = pd.DataFrame(divergence_rows)
            divergence_df.to_csv('divergence_statistics_summary_extended.csv', index=False)

        # Save prediction quality analysis
        quality_analysis = self.analyze_by_prediction_quality()
        type_analysis = self.analyze_by_sentence_type()

        quality_rows = []
        for category, values in quality_analysis.items():
            if values:
                quality_rows.append({
                    'category': category,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'n_sentences': len(values)
                })

        for category, values in type_analysis.items():
            if values:
                quality_rows.append({
                    'category': category,
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'median': np.median(values),
                    'n_sentences': len(values)
                })

        if quality_rows:
            quality_df = pd.DataFrame(quality_rows)
            quality_df.to_csv('category_analysis_summary_extended.csv', index=False)

        print("✓ Summary statistics saved to CSV files")


def main():
    """Main execution function"""
    print("Starting Streamlined XAI Methods Comparison Analysis...")
    print("=" * 80)

    # Initialize comparator
    comparator = StreamlinedXAIComparison()

    # Load data
    comparator.load_data()

    if len(comparator.token_data) < 2:
        print("Need at least 2 methods to compare. Please check your data files.")
        return

    # Generate streamlined report
    comparator.generate_streamlined_report()

    # Create visualizations
    print("\nGenerating streamlined visualizations...")
    fig, matrices = comparator.create_streamlined_visualizations()

    # Save summary statistics
    comparator.save_summary_statistics()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Files generated:")
    print("- streamlined_xai_comparison.png (combined visualizations)")
    print("- sentence_cosine_similarity_matrix.png (individual matrix)")
    print("- sentence_pearson_correlation_matrix.png (individual matrix)")
    print("- sentence_jensen_shannon_divergence_matrix.png (individual matrix)")
    print("- overlap_statistics_summary_extended.csv (overlap statistics)")
    print("- divergence_statistics_summary_extended.csv (divergence measures)")
    print("- category_analysis_summary_extended.csv (prediction quality & sentence type analysis)")
    print("=" * 80)


if __name__ == "__main__":
    main()