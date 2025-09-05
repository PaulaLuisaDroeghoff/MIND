import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kendalltau, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon
import warnings

warnings.filterwarnings('ignore')


class ExtendedDialogueXAIComparison:
    def __init__(self):
        self.methods_config = {
            'SHAP': {
                'sentence_file': 'shap_dialogue_sentences_extended.csv',
                'sentence_score_col': 'sentence_contribution',
                'tokens_file': 'shap_dialogue_tokens_extended.csv',
                'token_score_col': 'shap_value'
            },
            'LIME': {
                'sentence_file': 'lime_dialogue_sentences_extended.csv',
                'sentence_score_col': 'sentence_contribution',
                'tokens_file': 'lime_dialogue_tokens_extended.csv',
                'token_score_col': 'lime_value'
            },
            'Raw_Attention': {
                'sentence_file': 'raw_attention_dialogue_sentences_extended.csv',
                'sentence_score_col': 'sentence_contribution',
                'tokens_file': 'raw_attention_dialogue_tokens_extended.csv',
                'token_score_col': 'attention_value'
            },
            'Integrated_Gradients': {
                'sentence_file': 'integrated_gradients_dialogue_sentences_extended.csv',
                'sentence_score_col': 'sentence_contribution',
                'tokens_file': 'integrated_gradients_dialogue_tokens_extended.csv',
                'token_score_col': 'ig_value'
            },
            'Expected_Gradients': {
                'sentence_file': 'expected_gradients_dialogue_sentences_extended.csv',
                'sentence_score_col': 'sentence_contribution',
                'tokens_file': 'expected_gradients_dialogue_tokens_extended.csv',
                'token_score_col': 'eg_value'
            },
            'Token_Occlusion': {
                'sentence_file': 'token_occlusion_dialogue_sentences_extended.csv',
                'sentence_score_col': 'sentence_contribution',
                'tokens_file': 'token_occlusion_dialogue_tokens_extended.csv',
                'token_score_col': 'occlusion_value'
            }
        }

        self.sentence_data = {}
        self.token_data = {}

    def load_data(self):
        """Load all extended dialogue-level XAI method results"""
        print("Loading extended dialogue-level XAI method data...")

        for method, config in self.methods_config.items():
            try:
                # Load sentence-level data
                self.sentence_data[method] = pd.read_csv(config['sentence_file'])

                # Load token-level data
                self.token_data[method] = pd.read_csv(config['tokens_file'])

                print(f"✓ {method}: {len(self.sentence_data[method])} sentences, {len(self.token_data[method])} tokens")

            except FileNotFoundError as e:
                print(f"✗ {method}: File not found - {e}")

        print(f"\nLoaded {len(self.sentence_data)} methods successfully")

        # Show dataset overview
        if self.sentence_data:
            first_method = next(iter(self.sentence_data.keys()))
            total_dialogues = len(self.sentence_data[first_method]['dialogue_id'].unique())
            total_sentences = len(self.sentence_data[first_method])
            print(f"Dataset: {total_dialogues} dialogues, {total_sentences} sentences\n")

    def get_dialogue_ids(self):
        """Get all unique dialogue IDs"""
        if not self.sentence_data:
            return []

        first_method = next(iter(self.sentence_data.keys()))
        return sorted(self.sentence_data[first_method]['dialogue_id'].unique())

    def calculate_dialogue_sentence_overlap(self):
        """Calculate sentence overlap agreement at dialogue level"""
        dialogue_ids = self.get_dialogue_ids()
        all_overlaps = []

        for dialogue_id in dialogue_ids:
            methods = list(self.sentence_data.keys())
            dialogue_overlaps = []

            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i < j:
                        overlap = self.calculate_sentence_overlap_for_dialogue(dialogue_id, method1, method2)
                        if overlap is not None:
                            dialogue_overlaps.append(overlap)

            if dialogue_overlaps:
                all_overlaps.extend(dialogue_overlaps)

        if all_overlaps:
            return {
                'mean_overlap': np.mean(all_overlaps),
                'std_overlap': np.std(all_overlaps),
                'median_overlap': np.median(all_overlaps)
            }
        return None

    def calculate_sentence_overlap_for_dialogue(self, dialogue_id, method1, method2, top_k=3):
        """Calculate overlap between top-k sentences identified by two methods"""

        # Get sentence importance scores for both methods
        df1 = self.sentence_data[method1][self.sentence_data[method1]['dialogue_id'] == dialogue_id]
        df2 = self.sentence_data[method2][self.sentence_data[method2]['dialogue_id'] == dialogue_id]

        if len(df1) < top_k or len(df2) < top_k:
            return None

        # Get top-k sentences for each method
        score_col1 = self.methods_config[method1]['sentence_score_col']
        score_col2 = self.methods_config[method2]['sentence_score_col']

        top_sentences1 = set(df1.nlargest(top_k, score_col1)['sentence_id'])
        top_sentences2 = set(df2.nlargest(top_k, score_col2)['sentence_id'])

        # Calculate overlap
        intersection = top_sentences1.intersection(top_sentences2)
        union = top_sentences1.union(top_sentences2)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)  # Jaccard similarity

    def calculate_sentence_level_correlations(self, dialogue_id):
        """Calculate correlations between methods for sentence importance scores"""
        methods = list(self.sentence_data.keys())

        # Get sentence data for each method
        method_data = {}
        for method in methods:
            df = self.sentence_data[method]
            dialogue_sentences = df[df['dialogue_id'] == dialogue_id]
            score_col = self.methods_config[method]['sentence_score_col']
            method_data[method] = dict(zip(dialogue_sentences['sentence_id'], dialogue_sentences[score_col]))

        # Calculate pairwise correlations
        correlations = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j and method1 in method_data and method2 in method_data:
                    # Find common sentences
                    sent1 = set(method_data[method1].keys())
                    sent2 = set(method_data[method2].keys())
                    common_sentences = sent1.intersection(sent2)

                    if len(common_sentences) >= 3:
                        scores1 = [method_data[method1][sent_id] for sent_id in sorted(common_sentences)]
                        scores2 = [method_data[method2][sent_id] for sent_id in sorted(common_sentences)]

                        try:
                            corr, p_val = spearmanr(scores1, scores2)
                            if not np.isnan(corr):
                                correlations[f"{method1}_vs_{method2}"] = {
                                    'correlation': corr,
                                    'p_value': p_val,
                                    'n_sentences': len(common_sentences)
                                }
                        except:
                            pass

        return correlations

    def analyze_method_agreements_across_dataset(self):
        """Analyze method agreements across all dialogues in the extended dataset"""
        dialogue_ids = self.get_dialogue_ids()
        all_correlations = defaultdict(list)

        print(f"Analyzing method agreements across {len(dialogue_ids)} dialogues...")

        for dialogue_id in dialogue_ids:
            correlations = self.calculate_sentence_level_correlations(dialogue_id)
            for pair, stats in correlations.items():
                all_correlations[pair].append(stats['correlation'])

        # Calculate summary statistics
        summary_correlations = {}
        for pair, corr_list in all_correlations.items():
            if len(corr_list) >= 1:
                summary_correlations[pair] = {
                    'mean_correlation': np.mean(corr_list),
                    'std_correlation': np.std(corr_list) if len(corr_list) > 1 else 0,
                    'median_correlation': np.median(corr_list),
                    'n_dialogues': len(corr_list),
                    'correlations': corr_list
                }

        return summary_correlations

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
        min_val = np.min(values)
        if min_val < 0:
            values = values + abs(min_val)

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

            epsilon = 1e-10
            prob1 = prob1 + epsilon
            prob2 = prob2 + epsilon
            prob1 = prob1 / np.sum(prob1)
            prob2 = prob2 / np.sum(prob2)

            js_div = jensenshannon(prob1, prob2)
            return js_div if not np.isnan(js_div) else np.nan
        except:
            return np.nan

    def calculate_pairwise_divergences_extended(self, dialogue_id):
        """Calculate divergences with multiple measures for extended analysis"""
        methods = list(self.sentence_data.keys())

        # Get sentence data for each method
        method_data = {}
        for method in methods:
            df = self.sentence_data[method]
            dialogue_sentences = df[df['dialogue_id'] == dialogue_id]
            score_col = self.methods_config[method]['sentence_score_col']
            method_data[method] = dict(zip(dialogue_sentences['sentence_id'], dialogue_sentences[score_col]))

        # Calculate pairwise divergences
        divergences = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    pair_key = f"{method1}_vs_{method2}"

                    # Find common sentences
                    sent1 = set(method_data[method1].keys())
                    sent2 = set(method_data[method2].keys())
                    common_sentences = sent1.intersection(sent2)

                    if len(common_sentences) >= 3:
                        sorted_sentences = sorted(common_sentences)
                        scores1 = np.array([method_data[method1][sent_id] for sent_id in sorted_sentences])
                        scores2 = np.array([method_data[method2][sent_id] for sent_id in sorted_sentences])

                        # Calculate multiple divergence measures
                        cosine_sim = self.compute_cosine_similarity(scores1, scores2)
                        pearson_corr = self.compute_pearson_correlation(scores1, scores2)
                        js_div = self.compute_js_divergence(scores1, scores2)

                        divergences[pair_key] = {
                            'cosine_similarity': cosine_sim,
                            'pearson_correlation': pearson_corr,
                            'jensen_shannon_divergence': js_div,
                            'n_sentences': len(common_sentences)
                        }

        return divergences

    def analyze_method_divergences_across_dataset(self):
        """Analyze divergences across all dialogues in extended dataset"""
        dialogue_ids = self.get_dialogue_ids()
        all_divergences = defaultdict(lambda: defaultdict(list))

        print(f"Analyzing divergences across {len(dialogue_ids)} dialogues...")

        for dialogue_id in dialogue_ids:
            divergences = self.calculate_pairwise_divergences_extended(dialogue_id)

            for pair, measures in divergences.items():
                for measure_name, value in measures.items():
                    if measure_name != 'n_sentences' and not np.isnan(value):
                        all_divergences[pair][measure_name].append(value)

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
                        'n_dialogues': len(values)
                    }

        return summary_divergences

    def analyze_by_dialogue_category(self):
        """Analyze method performance by dialogue category"""
        dialogue_ids = self.get_dialogue_ids()

        # Get categories from first method
        first_method = next(iter(self.sentence_data.keys()))
        category_info = {}
        for dialogue_id in dialogue_ids:
            dialogue_data = self.sentence_data[first_method][
                self.sentence_data[first_method]['dialogue_id'] == dialogue_id
                ]
            if not dialogue_data.empty:
                category_info[dialogue_id] = dialogue_data['category'].iloc[0]

        # Group by category
        category_performance = defaultdict(lambda: defaultdict(list))

        for dialogue_id in dialogue_ids:
            if dialogue_id in category_info:
                category = category_info[dialogue_id]

                # Get correlations for this dialogue
                correlations = self.calculate_sentence_level_correlations(dialogue_id)

                for pair, stats in correlations.items():
                    category_performance[category][pair].append(stats['correlation'])

        # Calculate summary by category
        category_summary = {}
        for category, pairs in category_performance.items():
            category_summary[category] = {}
            for pair, correlations in pairs.items():
                if correlations:
                    category_summary[category][pair] = {
                        'mean_correlation': np.mean(correlations),
                        'std_correlation': np.std(correlations),
                        'n_dialogues': len(correlations)
                    }

        return category_summary

    def analyze_prediction_accuracy_impact(self):
        """Analyze how method agreement relates to prediction accuracy"""
        dialogue_ids = self.get_dialogue_ids()

        # Get prediction accuracy for each dialogue
        first_method = next(iter(self.sentence_data.keys()))
        prediction_accuracy = {}

        # Check what columns are actually available
        sample_data = self.sentence_data[first_method].head(1)
        print(f"Available columns in {first_method}: {sample_data.columns.tolist()}")

        # Try different possible column names for actual and predicted labels
        actual_label_options = ['actual_label', 'dialogue_label', 'true_label', 'label']
        predicted_label_options = ['predicted_label', 'predicted_class', 'prediction']

        actual_col = None
        predicted_col = None

        for col in actual_label_options:
            if col in sample_data.columns:
                actual_col = col
                break

        for col in predicted_label_options:
            if col in sample_data.columns:
                predicted_col = col
                break

        if actual_col is None or predicted_col is None:
            print(f"Could not find label columns. Available: {sample_data.columns.tolist()}")
            return {'correct_predictions': {}, 'incorrect_predictions': {}}

        print(f"Using columns: actual='{actual_col}', predicted='{predicted_col}'")

        for dialogue_id in dialogue_ids:
            dialogue_data = self.sentence_data[first_method][
                self.sentence_data[first_method]['dialogue_id'] == dialogue_id
                ]
            if not dialogue_data.empty:
                actual = dialogue_data[actual_col].iloc[0]
                predicted = dialogue_data[predicted_col].iloc[0]
                prediction_accuracy[dialogue_id] = 1 if actual == predicted else 0

        # Group dialogues by prediction accuracy
        correct_predictions = [d for d, acc in prediction_accuracy.items() if acc == 1]
        incorrect_predictions = [d for d, acc in prediction_accuracy.items() if acc == 0]

        # Calculate method agreements for each group
        correct_agreements = defaultdict(list)
        incorrect_agreements = defaultdict(list)

        for dialogue_id in correct_predictions:
            correlations = self.calculate_sentence_level_correlations(dialogue_id)
            for pair, stats in correlations.items():
                correct_agreements[pair].append(stats['correlation'])

        for dialogue_id in incorrect_predictions:
            correlations = self.calculate_sentence_level_correlations(dialogue_id)
            for pair, stats in correlations.items():
                incorrect_agreements[pair].append(stats['correlation'])

        return {
            'correct_predictions': {
                pair: {
                    'mean_correlation': np.mean(corrs),
                    'std_correlation': np.std(corrs),
                    'n_dialogues': len(corrs)
                } for pair, corrs in correct_agreements.items() if corrs
            },
            'incorrect_predictions': {
                pair: {
                    'mean_correlation': np.mean(corrs),
                    'std_correlation': np.std(corrs),
                    'n_dialogues': len(corrs)
                } for pair, corrs in incorrect_agreements.items() if corrs
            }
        }

    def create_divergence_matrices(self):
        """Create matrices for visualization"""
        divergences = self.analyze_method_divergences_across_dataset()
        methods = list(self.sentence_data.keys())
        n_methods = len(methods)

        # Initialize matrices
        cosine_matrix = np.full((n_methods, n_methods), np.nan)
        pearson_matrix = np.full((n_methods, n_methods), np.nan)
        js_matrix = np.full((n_methods, n_methods), np.nan)

        # Fill diagonal
        for i in range(n_methods):
            cosine_matrix[i, i] = 1.0
            pearson_matrix[i, i] = 1.0
            js_matrix[i, i] = 0.0

        # Fill matrices
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    pair_key = f"{method1}_vs_{method2}"
                    reverse_pair = f"{method2}_vs_{method1}"

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

    def create_extended_visualizations(self):
        """Create comprehensive visualizations for extended dataset"""
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Extended Dialogue XAI Methods Comparison Analysis', fontsize=16, fontweight='bold')

        # 1. Method Agreement Overview - HORIZONTAL BAR CHART
        ax1 = axes[0, 0]
        correlations = self.analyze_method_agreements_across_dataset()

        if correlations:
            pairs = list(correlations.keys())
            mean_corrs = [correlations[pair]['mean_correlation'] for pair in pairs]
            std_corrs = [correlations[pair]['std_correlation'] for pair in pairs]

            # Sort by correlation value for better visualization
            sorted_data = sorted(zip(pairs, mean_corrs, std_corrs), key=lambda x: x[1], reverse=True)
            pairs = [x[0] for x in sorted_data]
            mean_corrs = [x[1] for x in sorted_data]
            std_corrs = [x[2] for x in sorted_data]

            # Create horizontal bar chart
            y_pos = range(len(pairs))
            bars = ax1.barh(y_pos, mean_corrs, xerr=std_corrs, capsize=3, alpha=0.7)

            ax1.set_title('Average Method Agreement\n(Sentence Importance Correlation)')
            ax1.set_xlabel('Mean Correlation')
            ax1.set_ylabel('Method Pairs')
            ax1.set_yticks(y_pos)

            # Clean method pair names
            clean_pair_names = []
            for pair in pairs:
                methods = pair.replace('_vs_', ' vs ').split(' vs ')
                short_methods = []
                for method in methods:
                    if 'Raw_Attention' in method:
                        short_methods.append('RawAtt')
                    elif 'Integrated_Gradients' in method:
                        short_methods.append('IntGrad')
                    elif 'Expected_Gradients' in method:
                        short_methods.append('ExpGrad')
                    elif 'Token_Occlusion' in method:
                        short_methods.append('TokenOcc')
                    else:
                        short_methods.append(method)
                clean_pair_names.append(' vs '.join(short_methods))

            ax1.set_yticklabels(clean_pair_names, fontsize=8)
            ax1.grid(axis='x', alpha=0.3)

            # Add values next to bars
            for i, (bar, mean_val) in enumerate(zip(bars, mean_corrs)):
                width = bar.get_width()
                ax1.text(width + 0.01, bar.get_y() + bar.get_height() / 2 + 0.1,
                         f'{mean_val:.3f}', ha='left', va='bottom', fontsize=12)

        # 2-4. Divergence Matrices
        matrices = self.create_divergence_matrices()
        methods = matrices['methods']

        # Cosine Similarity
        ax2 = axes[1, 0]
        cosine_matrix = matrices['cosine_similarity']
        im2 = ax2.imshow(cosine_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax2.set_title('Cosine Similarity Matrix')
        ax2.set_xticks(range(len(methods)))
        ax2.set_yticks(range(len(methods)))
        ax2.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax2.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Add values
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(cosine_matrix[i, j]):
                    color = "white" if abs(cosine_matrix[i, j]) > 0.5 else "black"
                    ax2.text(j, i, f'{cosine_matrix[i, j]:.2f}',
                             ha="center", va="center", color=color, fontsize=10)

        # Pearson Correlation
        ax3 = axes[1, 1]
        pearson_matrix = matrices['pearson_correlation']
        im3 = ax3.imshow(pearson_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3.set_title('Pearson Correlation Matrix')
        ax3.set_xticks(range(len(methods)))
        ax3.set_yticks(range(len(methods)))
        ax3.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax3.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        # Add values
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(pearson_matrix[i, j]):
                    color = "white" if abs(pearson_matrix[i, j]) > 0.5 else "black"
                    ax3.text(j, i, f'{pearson_matrix[i, j]:.2f}',
                             ha="center", va="center", color=color, fontsize=10)

        # Jensen-Shannon Divergence
        ax4 = axes[1, 2]
        js_matrix = matrices['jensen_shannon_divergence']
        im4 = ax4.imshow(js_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax4.set_title('Jensen-Shannon Divergence Matrix')
        ax4.set_xticks(range(len(methods)))
        ax4.set_yticks(range(len(methods)))
        ax4.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax4.set_yticklabels(methods, fontsize=10)
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        # Add values
        for i in range(len(methods)):
            for j in range(len(methods)):
                if not np.isnan(js_matrix[i, j]):
                    color = "white" if js_matrix[i, j] > 0.5 else "black"
                    ax4.text(j, i, f'{js_matrix[i, j]:.2f}',
                             ha="center", va="center", color=color, fontsize=10)

        # 5. Agreement by Prediction Accuracy
        ax5 = axes[0, 1]
        accuracy_analysis = self.analyze_prediction_accuracy_impact()

        if accuracy_analysis['correct_predictions'] and accuracy_analysis['incorrect_predictions']:
            # Get common pairs
            correct_pairs = set(accuracy_analysis['correct_predictions'].keys())
            incorrect_pairs = set(accuracy_analysis['incorrect_predictions'].keys())
            common_pairs = list(correct_pairs.intersection(incorrect_pairs))[:10]  # Top 10

            if common_pairs:
                correct_means = [accuracy_analysis['correct_predictions'][pair]['mean_correlation'] for pair in
                                 common_pairs]
                incorrect_means = [accuracy_analysis['incorrect_predictions'][pair]['mean_correlation'] for pair in
                                   common_pairs]

                x = np.arange(len(common_pairs))
                width = 0.35

                bars1 = ax5.bar(x - width / 2, correct_means, width, label='Correct Predictions', alpha=0.7)
                bars2 = ax5.bar(x + width / 2, incorrect_means, width, label='Incorrect Predictions', alpha=0.7)

                ax5.set_title('Method Agreement by\nPrediction Accuracy')
                ax5.set_xlabel('Method Pairs')
                ax5.set_ylabel('Mean Correlation')
                ax5.set_xticks(x)
                ax5.set_xticklabels([pair.replace('_vs_', '\nvs\n') for pair in common_pairs], rotation=45, ha='right',
                                    fontsize=8)
                ax5.legend()
                ax5.grid(axis='y', alpha=0.3)

        # 6. Top Method Pairs by Consistency
        ax6 = axes[0, 2]
        divergences = self.analyze_method_divergences_across_dataset()

        # Get top pairs by cosine similarity
        cosine_pairs = [(pair, data['cosine_similarity']['mean'])
                        for pair, data in divergences.items()
                        if 'cosine_similarity' in data]
        cosine_pairs.sort(key=lambda x: x[1], reverse=True)
        top_pairs = cosine_pairs[:10]

        if top_pairs:
            pair_names = [pair.replace('_vs_', '\nvs\n') for pair, _ in top_pairs]
            similarities = [sim for _, sim in top_pairs]

            bars = ax6.barh(range(len(pair_names)), similarities, alpha=0.7)
            ax6.set_yticks(range(len(pair_names)))
            ax6.set_yticklabels(pair_names, fontsize=8)
            ax6.set_xlabel('Cosine Similarity')
            ax6.set_title('Top 10 Method Pairs\nby Similarity')
            ax6.grid(axis='x', alpha=0.3)

            # Add values
            for i, (bar, sim) in enumerate(zip(bars, similarities)):
                ax6.text(sim + 0.01, i, f'{sim:.3f}', va='center', fontsize=8)

        plt.tight_layout()

        # Save the combined plot (your original functionality)
        plt.savefig('extended_dialogue_xai_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

        # NEW: Save individual plots for the three matrices
        print("\nSaving individual dialogue-level matrix plots...")

        # 1. Individual Cosine Similarity Matrix
        fig_cosine, ax_cosine = plt.subplots(1, 1, figsize=(8, 6))
        im_cosine = ax_cosine.imshow(cosine_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_cosine.set_title('Dialogue-Level Cosine Similarity Matrix', fontsize=14, fontweight='bold')
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
        plt.savefig('dialogue_cosine_similarity_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Individual Pearson Correlation Matrix
        fig_pearson, ax_pearson = plt.subplots(1, 1, figsize=(8, 6))
        im_pearson = ax_pearson.imshow(pearson_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        ax_pearson.set_title('Dialogue-Level Pearson Correlation Matrix', fontsize=14, fontweight='bold')
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
        plt.savefig('dialogue_pearson_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 3. Individual Jensen-Shannon Divergence Matrix
        fig_js, ax_js = plt.subplots(1, 1, figsize=(8, 6))
        im_js = ax_js.imshow(js_matrix, cmap='RdYlBu_r', vmin=0, vmax=1)
        ax_js.set_title('Dialogue-Level Jensen-Shannon Divergence Matrix', fontsize=14, fontweight='bold')
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
        plt.savefig('dialogue_jensen_shannon_divergence_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("✓ Individual dialogue-level matrix plots saved:")
        print("  - dialogue_cosine_similarity_matrix.png")
        print("  - dialogue_pearson_correlation_matrix.png")
        print("  - dialogue_jensen_shannon_divergence_matrix.png")

        return fig, matrices

    def create_method_agreement_chart(self):
        """Create and save individual method agreement chart"""
        print("Creating individual method agreement chart...")

        # Create figure for just this chart
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        correlations = self.analyze_method_agreements_across_dataset()

        if correlations:
            pairs = list(correlations.keys())
            mean_corrs = [correlations[pair]['mean_correlation'] for pair in pairs]
            std_corrs = [correlations[pair]['std_correlation'] for pair in pairs]

            # Sort by correlation value for better visualization
            sorted_data = sorted(zip(pairs, mean_corrs, std_corrs), key=lambda x: x[1], reverse=True)
            pairs = [x[0] for x in sorted_data]
            mean_corrs = [x[1] for x in sorted_data]
            std_corrs = [x[2] for x in sorted_data]

            # Create horizontal bar chart
            y_pos = range(len(pairs))
            bars = ax.barh(y_pos, mean_corrs, xerr=std_corrs, capsize=3, alpha=0.7, color='steelblue')

            ax.set_title('Average Method Agreement\n(Sentence Importance Correlation)',
                         fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Mean Correlation', fontsize=12)
            ax.set_ylabel('Method Pairs', fontsize=12)
            ax.set_yticks(y_pos)

            # Clean method pair names
            clean_pair_names = []
            for pair in pairs:
                methods = pair.replace('_vs_', ' vs ').split(' vs ')
                short_methods = []
                for method in methods:
                    if 'Raw_Attention' in method:
                        short_methods.append('Raw Attention')
                    elif 'Integrated_Gradients' in method:
                        short_methods.append('Integrated Gradients')
                    elif 'Expected_Gradients' in method:
                        short_methods.append('Expected Gradients')
                    elif 'Token_Occlusion' in method:
                        short_methods.append('Token Occlusion')
                    else:
                        short_methods.append(method)
                clean_pair_names.append(' vs '.join(short_methods))

            ax.set_yticklabels(clean_pair_names, fontsize=10)
            ax.grid(axis='x', alpha=0.3)
            ax.set_axisbelow(True)

            # Add values above bars
            for i, (bar, mean_val) in enumerate(zip(bars, mean_corrs)):
                width = bar.get_width()
                x_pos = width + 0.02 if width >= 0 else width - 0.02
                y_pos = bar.get_y() + bar.get_height() + 0.05  # Above the bar
                ha_align = 'left' if width >= 0 else 'right'
                ax.text(x_pos, y_pos, f'{mean_val:.3f}',
                        ha=ha_align, va='bottom', fontsize=10)

            # Adjust layout
            plt.tight_layout()

            # Save the chart
            plt.savefig('method_agreement_chart.png', dpi=300, bbox_inches='tight')
            plt.close()  # Close to free memory

            print("✓ Method agreement chart saved as 'method_agreement_chart.png'")

        else:
            print("✗ No correlation data available for chart")

    def generate_extended_report(self):
        """Generate comprehensive report for extended dataset"""
        print("=" * 80)
        print("EXTENDED DIALOGUE XAI METHODS COMPARISON REPORT")
        print("=" * 80)

        dialogue_ids = self.get_dialogue_ids()

        # Dataset overview
        print(f"\n1. DATASET OVERVIEW:")
        print(f"   Total dialogues analyzed: {len(dialogue_ids)}")
        print(f"   Methods compared: {len(self.sentence_data)}")
        print(f"   Methods: {', '.join(self.sentence_data.keys())}")

        # Sentence overlap analysis
        print(f"\n2. SENTENCE OVERLAP STATISTICS:")
        sentence_overlap = self.calculate_dialogue_sentence_overlap()
        if sentence_overlap:
            print(
                f"   Overall average sentence overlap: {sentence_overlap['mean_overlap']:.3f} ± {sentence_overlap['std_overlap']:.3f}")
            print(f"   Overall median sentence overlap: {sentence_overlap['median_overlap']:.3f}")
        else:
            print("   Could not calculate sentence overlap statistics")

        # Method agreement analysis
        print(f"\n2. METHOD AGREEMENT ANALYSIS:")
        correlations = self.analyze_method_agreements_across_dataset()

        print(f"   Method pairs with valid correlations: {len(correlations)}")

        # Top correlations
        sorted_correlations = sorted(correlations.items(),
                                     key=lambda x: x[1]['mean_correlation'], reverse=True)

        print(f"\n   Top 10 method pairs by correlation:")
        for i, (pair, stats) in enumerate(sorted_correlations[:10]):
            print(
                f"   {i + 1:2d}. {pair:40s}: {stats['mean_correlation']:.3f} ± {stats['std_correlation']:.3f} (n={stats['n_dialogues']})")

        print(f"\n   Bottom 5 method pairs by correlation:")
        for i, (pair, stats) in enumerate(sorted_correlations[-5:]):
            print(
                f"   {i + 1:2d}. {pair:40s}: {stats['mean_correlation']:.3f} ± {stats['std_correlation']:.3f} (n={stats['n_dialogues']})")

        # Divergence analysis
        print(f"\n3. DIVERGENCE ANALYSIS:")
        divergences = self.analyze_method_divergences_across_dataset()

        print(f"   Method pairs with divergence measures: {len(divergences)}")

        # Top similarities
        cosine_pairs = [(pair, data['cosine_similarity']['mean'])
                        for pair, data in divergences.items()
                        if 'cosine_similarity' in data]
        cosine_pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n   Top 5 method pairs by Cosine Similarity:")
        for i, (pair, sim) in enumerate(cosine_pairs[:5]):
            n_dialogues = divergences[pair]['cosine_similarity']['n_dialogues']
            std = divergences[pair]['cosine_similarity']['std']
            print(f"   {i + 1}. {pair:40s}: {sim:.3f} ± {std:.3f} (n={n_dialogues})")

        # Pearson correlations
        pearson_pairs = [(pair, data['pearson_correlation']['mean'])
                         for pair, data in divergences.items()
                         if 'pearson_correlation' in data]
        pearson_pairs.sort(key=lambda x: x[1], reverse=True)

        print(f"\n   Top 5 method pairs by Pearson Correlation:")
        for i, (pair, corr) in enumerate(pearson_pairs[:5]):
            n_dialogues = divergences[pair]['pearson_correlation']['n_dialogues']
            std = divergences[pair]['pearson_correlation']['std']
            print(f"   {i + 1}. {pair:40s}: {corr:.3f} ± {std:.3f} (n={n_dialogues})")

        # Jensen-Shannon (lowest = most similar)
        js_pairs = [(pair, data['jensen_shannon_divergence']['mean'])
                    for pair, data in divergences.items()
                    if 'jensen_shannon_divergence' in data]
        js_pairs.sort(key=lambda x: x[1])

        print(f"\n   Top 5 method pairs by Jensen-Shannon Divergence (Lower = More Similar):")
        for i, (pair, div) in enumerate(js_pairs[:5]):
            n_dialogues = divergences[pair]['jensen_shannon_divergence']['n_dialogues']
            std = divergences[pair]['jensen_shannon_divergence']['std']
            print(f"   {i + 1}. {pair:40s}: {div:.3f} ± {std:.3f} (n={n_dialogues})")

        # Prediction accuracy analysis
        print(f"\n4. PREDICTION ACCURACY IMPACT:")
        accuracy_analysis = self.analyze_prediction_accuracy_impact()

        if accuracy_analysis['correct_predictions'] and accuracy_analysis['incorrect_predictions']:
            correct_pairs = accuracy_analysis['correct_predictions']
            incorrect_pairs = accuracy_analysis['incorrect_predictions']

            # Find common pairs and compare
            common_pairs = set(correct_pairs.keys()).intersection(set(incorrect_pairs.keys()))

            print(f"   Comparing method agreement on correct vs incorrect predictions:")
            print(f"   Common method pairs: {len(common_pairs)}")

            better_on_correct = 0
            better_on_incorrect = 0

            for pair in list(common_pairs)[:10]:  # Show top 10
                correct_corr = correct_pairs[pair]['mean_correlation']
                incorrect_corr = incorrect_pairs[pair]['mean_correlation']

                if correct_corr > incorrect_corr:
                    better_on_correct += 1
                    symbol = "✓"
                else:
                    better_on_incorrect += 1
                    symbol = "✗"

                print(f"   {symbol} {pair:35s}: Correct={correct_corr:.3f}, Incorrect={incorrect_corr:.3f}")

            print(f"\n   Summary: {better_on_correct} pairs agree more on correct predictions")
            print(f"            {better_on_incorrect} pairs agree more on incorrect predictions")

        # Category analysis
        print(f"\n5. PERFORMANCE BY DIALOGUE CATEGORY:")
        category_analysis = self.analyze_by_dialogue_category()

        for category, pairs in category_analysis.items():
            print(f"\n   {category}:")
            # Show top 3 method pairs for this category
            sorted_pairs = sorted(pairs.items(), key=lambda x: x[1]['mean_correlation'], reverse=True)
            for i, (pair, stats) in enumerate(sorted_pairs[:3]):
                print(f"     {i + 1}. {pair:35s}: {stats['mean_correlation']:.3f} ± {stats['std_correlation']:.3f}")

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
            print(
                f"   {i + 1}. {pair:35s}: avg_rank={avg_rank:.1f} (CS:{cs_rank + 1}, PC:{pc_rank + 1}, JS:{js_rank + 1})")

        # Dataset statistics
        print(f"\n7. DATASET STATISTICS:")
        first_method = next(iter(self.sentence_data.keys()))
        df = self.sentence_data[first_method]

        # Check available columns
        sample_data = df.head(1)
        available_cols = sample_data.columns.tolist()
        print(f"   Available columns: {available_cols}")

        # Try to find label columns
        actual_label_options = ['actual_label', 'dialogue_label', 'true_label', 'label']
        predicted_label_options = ['predicted_label', 'predicted_class', 'prediction']

        actual_col = None
        predicted_col = None

        for col in actual_label_options:
            if col in available_cols:
                actual_col = col
                break

        for col in predicted_label_options:
            if col in available_cols:
                predicted_col = col
                break

        if actual_col and predicted_col:
            # Label distribution
            label_dist = df.groupby('dialogue_id')[actual_col].first().value_counts()
            print(f"   Label distribution:")
            print(f"     Non-manipulative (0): {label_dist.get(0, 0)} dialogues")
            print(f"     Manipulative (1): {label_dist.get(1, 0)} dialogues")

            # Prediction accuracy
            predictions = df.groupby('dialogue_id').agg({
                actual_col: 'first',
                predicted_col: 'first'
            })
            accuracy = (predictions[actual_col] == predictions[predicted_col]).mean()
            print(f"   Overall model accuracy: {accuracy:.3f} ({accuracy * 100:.1f}%)")
        else:
            print(f"   Could not find label columns in: {available_cols}")

        # Category distribution (if available)
        if 'category' in available_cols:
            category_dist = df.groupby('dialogue_id')['category'].first().value_counts()
            print(f"\n   Category distribution:")
            for category, count in category_dist.items():
                print(f"     {category}: {count} dialogues")

        print(f"\n{'=' * 80}")

    def save_extended_summary_statistics(self):
        """Save comprehensive summary statistics"""
        print("\nSaving extended summary statistics...")

        # Method agreements
        correlations = self.analyze_method_agreements_across_dataset()
        correlation_rows = []
        for pair, stats in correlations.items():
            correlation_rows.append({
                'method_pair': pair,
                'mean_correlation': stats['mean_correlation'],
                'std_correlation': stats['std_correlation'],
                'median_correlation': stats['median_correlation'],
                'n_dialogues': stats['n_dialogues']
            })

        if correlation_rows:
            correlation_df = pd.DataFrame(correlation_rows)
            correlation_df.to_csv('extended_method_agreements.csv', index=False)

        # Divergence measures
        divergences = self.analyze_method_divergences_across_dataset()
        divergence_rows = []
        for pair, measures in divergences.items():
            for measure_name, stats in measures.items():
                divergence_rows.append({
                    'method_pair': pair,
                    'measure': measure_name,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'median': stats['median'],
                    'min': stats['min'],
                    'max': stats['max'],
                    'n_dialogues': stats['n_dialogues']
                })

        if divergence_rows:
            divergence_df = pd.DataFrame(divergence_rows)
            divergence_df.to_csv('extended_method_divergences.csv', index=False)

        # Category analysis
        category_analysis = self.analyze_by_dialogue_category()
        category_rows = []
        for category, pairs in category_analysis.items():
            for pair, stats in pairs.items():
                category_rows.append({
                    'category': category,
                    'method_pair': pair,
                    'mean_correlation': stats['mean_correlation'],
                    'std_correlation': stats['std_correlation'],
                    'n_dialogues': stats['n_dialogues']
                })

        if category_rows:
            category_df = pd.DataFrame(category_rows)
            category_df.to_csv('extended_category_analysis.csv', index=False)

        # Accuracy impact analysis
        accuracy_analysis = self.analyze_prediction_accuracy_impact()
        accuracy_rows = []

        for pred_type, pairs in accuracy_analysis.items():
            for pair, stats in pairs.items():
                accuracy_rows.append({
                    'prediction_type': pred_type,
                    'method_pair': pair,
                    'mean_correlation': stats['mean_correlation'],
                    'std_correlation': stats['std_correlation'],
                    'n_dialogues': stats['n_dialogues']
                })

        if accuracy_rows:
            accuracy_df = pd.DataFrame(accuracy_rows)
            accuracy_df.to_csv('extended_accuracy_impact_analysis.csv', index=False)

        print("✓ Extended summary statistics saved to CSV files")


def main():
    """Main execution function for extended dialogue comparison"""
    print("Starting Extended Dialogue XAI Methods Comparison Analysis...")
    print("=" * 80)

    # Initialize comparator
    comparator = ExtendedDialogueXAIComparison()

    # Load data
    comparator.load_data()

    if len(comparator.sentence_data) < 2:
        print("Need at least 2 methods to compare. Please check your data files.")
        return

    # Generate comprehensive report
    comparator.generate_extended_report()

    # Create visualizations
    print("\nGenerating comprehensive visualizations...")
    fig, matrices = comparator.create_extended_visualizations()

    # CREATE AND SAVE INDIVIDUAL METHOD AGREEMENT CHART
    comparator.create_method_agreement_chart()

    # Save summary statistics
    comparator.save_extended_summary_statistics()

    print("\n" + "=" * 80)
    print("EXTENDED ANALYSIS COMPLETE!")
    print("=" * 80)
    print("Files generated:")
    print("- extended_dialogue_xai_comparison.png (combined visualizations)")
    print("- method_agreement_chart.png (individual method agreement chart)")  # NEW LINE
    print("- dialogue_cosine_similarity_matrix.png (individual matrix)")
    print("- dialogue_pearson_correlation_matrix.png (individual matrix)")
    print("- dialogue_jensen_shannon_divergence_matrix.png (individual matrix)")
    print("- extended_method_agreements.csv (correlation statistics)")
    print("- extended_method_divergences.csv (divergence measures)")
    print("- extended_category_analysis.csv (performance by category)")
    print("- extended_accuracy_impact_analysis.csv (accuracy impact analysis)")
    print("=" * 80)


if __name__ == "__main__":
    main()