import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import re

warnings.filterwarnings('ignore')


class FixedIGAttentionVisualizer:
    def __init__(self):
        self.token_data = {}

    def load_data(self):
        """Load token data for both methods"""
        print("Loading XAI method data...")

        try:
            self.token_data['Integrated_Gradients'] = pd.read_csv('gradients_token_results.csv')
            self.token_data['Raw_Attention'] = pd.read_csv('attention_token_results.csv')

            print(f"✓ Integrated Gradients: {len(self.token_data['Integrated_Gradients'])} tokens loaded")
            print(f"✓ Raw Attention: {len(self.token_data['Raw_Attention'])} tokens loaded")

        except FileNotFoundError as e:
            print(f"✗ File not found: {e}")
            return False

        print(f"\nLoaded both methods successfully\n")
        return True

    def get_sentence_ids(self):
        """Get all unique sentence IDs"""
        if not self.token_data:
            return []

        return sorted(self.token_data['Integrated_Gradients']['sentence_id'].unique())

    def reconstruct_attention_tokens(self, tokens, scores):
        """Reconstruct attention tokens to match IG tokenization better"""
        reconstructed_tokens = []
        reconstructed_scores = []

        current_word = ""
        current_score_sum = 0
        current_count = 0

        for token, score in zip(tokens, scores):
            token = str(token)

            # Skip punctuation-only tokens
            if token.strip() in ['.', ',', '!', '?', ';', ':', '"', "'", '(', ')']:
                continue

            # If token starts with space, it's a new word
            if token.startswith(' ') and current_word:
                # Finish previous word
                if current_word.strip():
                    reconstructed_tokens.append(current_word.strip())
                    reconstructed_scores.append(current_score_sum / max(current_count, 1))

                # Start new word
                current_word = token.strip()
                current_score_sum = score
                current_count = 1
            else:
                # Continue current word
                if token.startswith(' '):
                    current_word = token.strip()
                else:
                    current_word += token
                current_score_sum += score
                current_count += 1

        # Add the last word
        if current_word.strip():
            reconstructed_tokens.append(current_word.strip())
            reconstructed_scores.append(current_score_sum / max(current_count, 1))

        return reconstructed_tokens, reconstructed_scores

    def fuzzy_match_tokens(self, ig_tokens, att_tokens, ig_scores, att_scores):
        """Try to match tokens between IG and Attention using fuzzy matching"""

        # Create mappings
        matched_tokens = []
        matched_ig_scores = []
        matched_att_scores = []

        # For each IG token, try to find best match in attention tokens
        for ig_token, ig_score in zip(ig_tokens, ig_scores):
            ig_lower = ig_token.lower().strip()

            best_match = None
            best_score = 0
            best_att_score = 0

            # Try exact match first
            for att_token, att_score in zip(att_tokens, att_scores):
                att_lower = att_token.lower().strip()

                if ig_lower == att_lower:
                    best_match = ig_token
                    best_score = ig_score
                    best_att_score = att_score
                    break

            # If no exact match, try substring matching
            if best_match is None:
                for att_token, att_score in zip(att_tokens, att_scores):
                    att_lower = att_token.lower().strip()

                    # Check if one is substring of the other
                    if (len(ig_lower) >= 3 and len(att_lower) >= 3 and
                            (ig_lower in att_lower or att_lower in ig_lower)):
                        best_match = ig_token
                        best_score = ig_score
                        best_att_score = att_score
                        break

            # Add to matched list if we found a match
            if best_match is not None:
                matched_tokens.append(best_match)
                matched_ig_scores.append(best_score)
                matched_att_scores.append(best_att_score)

        return matched_tokens, matched_ig_scores, matched_att_scores

    def get_sentence_data_with_alignment(self, sentence_id):
        """Get aligned data for both methods for a sentence"""

        # Get IG data
        ig_data = self.token_data['Integrated_Gradients']
        ig_sentence = ig_data[ig_data['sentence_id'] == sentence_id]

        if ig_sentence.empty:
            return None, None, None

        ig_tokens = ig_sentence['token'].tolist()
        ig_scores = ig_sentence['gradient_value'].tolist()

        # Get Attention data
        att_data = self.token_data['Raw_Attention']
        att_sentence = att_data[att_data['sentence_id'] == sentence_id]

        if att_sentence.empty:
            return None, None, None

        att_tokens = att_sentence['token'].tolist()
        att_scores = att_sentence['attention_value'].tolist()

        # Reconstruct attention tokens
        recon_att_tokens, recon_att_scores = self.reconstruct_attention_tokens(att_tokens, att_scores)

        print(f"Sentence {sentence_id}:")
        print(f"  IG tokens ({len(ig_tokens)}): {ig_tokens[:10]}")
        print(f"  Original Att tokens ({len(att_tokens)}): {att_tokens[:10]}")
        print(f"  Reconstructed Att tokens ({len(recon_att_tokens)}): {recon_att_tokens[:10]}")

        # Try to match tokens
        matched_tokens, matched_ig_scores, matched_att_scores = self.fuzzy_match_tokens(
            ig_tokens, recon_att_tokens, ig_scores, recon_att_scores
        )

        print(f"  Matched tokens: {len(matched_tokens)}")
        print(f"  Matched tokens list: {matched_tokens[:10]}")

        if len(matched_tokens) < 3:
            print(f"  Warning: Only {len(matched_tokens)} matched tokens found")

        return matched_tokens, matched_ig_scores, matched_att_scores

    def calculate_alignment_metrics(self, sentence_id):
        """Calculate alignment metrics for a sentence"""

        matched_tokens, ig_scores, att_scores = self.get_sentence_data_with_alignment(sentence_id)

        if matched_tokens is None or len(matched_tokens) < 2:
            return None

        # Calculate metrics
        metrics = {}

        try:
            # Pearson correlation
            pearson_corr, pearson_p = pearsonr(ig_scores, att_scores)
            metrics['pearson_correlation'] = pearson_corr if not np.isnan(pearson_corr) else 0

            # Spearman correlation
            spearman_corr, spearman_p = spearmanr(ig_scores, att_scores)
            metrics['spearman_correlation'] = spearman_corr if not np.isnan(spearman_corr) else 0

            # Cosine similarity
            cosine_sim = cosine_similarity([ig_scores], [att_scores])[0][0]
            metrics['cosine_similarity'] = cosine_sim if not np.isnan(cosine_sim) else 0

            # Top-k overlap
            top_k = min(5, len(matched_tokens))

            # Get indices of top tokens for each method
            ig_top_indices = set(np.argsort(np.abs(ig_scores))[-top_k:])
            att_top_indices = set(np.argsort(np.abs(att_scores))[-top_k:])

            overlap = len(ig_top_indices.intersection(att_top_indices)) / top_k
            metrics['top5_token_overlap'] = overlap

            # Mean absolute difference
            mean_abs_diff = np.mean(np.abs(np.array(ig_scores) - np.array(att_scores)))
            metrics['mean_abs_difference'] = mean_abs_diff

            metrics['n_matched_tokens'] = len(matched_tokens)

        except Exception as e:
            print(f"Error calculating metrics for {sentence_id}: {e}")
            return None

        return metrics

    def analyze_all_sentences(self):
        """Analyze alignment for all sentences"""
        sentence_ids = self.get_sentence_ids()

        print(f"Analyzing alignment for {len(sentence_ids)} sentences...")
        print("=" * 60)

        alignment_results = []

        for sentence_id in sentence_ids:
            print(f"\nProcessing {sentence_id}:")
            metrics = self.calculate_alignment_metrics(sentence_id)

            if metrics:
                metrics['sentence_id'] = sentence_id
                alignment_results.append(metrics)
                print(f"  ✓ Pearson: {metrics['pearson_correlation']:.3f}, "
                      f"Cosine: {metrics['cosine_similarity']:.3f}, "
                      f"Matched tokens: {metrics['n_matched_tokens']}")
            else:
                print(f"  ✗ Insufficient data")

        # Convert to DataFrame
        df_results = pd.DataFrame(alignment_results)

        if df_results.empty:
            print("No valid alignment results found.")
            return df_results

        # Show rankings
        print(f"\n{'=' * 60}")
        print("RANKING BY ALIGNMENT METRICS:")
        print(f"{'=' * 60}")

        print("\nBy Pearson Correlation:")
        top_pearson = df_results.nlargest(len(df_results), 'pearson_correlation')
        for _, row in top_pearson.iterrows():
            print(f"  {row['sentence_id']}: {row['pearson_correlation']:.3f} ({row['n_matched_tokens']} tokens)")

        print("\nBy Cosine Similarity:")
        top_cosine = df_results.nlargest(len(df_results), 'cosine_similarity')
        for _, row in top_cosine.iterrows():
            print(f"  {row['sentence_id']}: {row['cosine_similarity']:.3f} ({row['n_matched_tokens']} tokens)")

        return df_results

    def normalize_scores(self, scores):
        """Normalize scores by absolute maximum"""
        scores = np.array(scores)
        max_abs = np.max(np.abs(scores))
        if max_abs != 0:
            return scores / max_abs
        else:
            return scores

    def create_heatmap_comparison(self, sentence_id, figsize=(16, 8), show_scores=True):
        """Create heatmap comparison for a sentence"""

        matched_tokens, ig_scores, att_scores = self.get_sentence_data_with_alignment(sentence_id)

        if matched_tokens is None or len(matched_tokens) < 2:
            print(f"Cannot create visualization for {sentence_id} - insufficient matched tokens")
            return None

        # Get metrics for title
        metrics = self.calculate_alignment_metrics(sentence_id)

        # Normalize scores
        norm_ig_scores = self.normalize_scores(ig_scores)
        norm_att_scores = self.normalize_scores(att_scores)

        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figsize, height_ratios=[1, 1, 0.3])

        # Create title with metrics
        if metrics:
            title = (f'XAI Methods Comparison: Integrated Gradients vs Raw Attention\n'
                     f'Sentence: {sentence_id} | Pearson: {metrics["pearson_correlation"]:.3f} | '
                     f'Cosine: {metrics["cosine_similarity"]:.3f} | Matched tokens: {metrics["n_matched_tokens"]}')
        else:
            title = f'XAI Methods Comparison: Integrated Gradients vs Raw Attention\nSentence: {sentence_id}'

        fig.suptitle(title, fontsize=12, fontweight='bold')

        # IG heatmap
        self._create_method_heatmap(ax1, matched_tokens, norm_ig_scores, ig_scores,
                                    'Integrated Gradients', show_scores)

        # Attention heatmap
        self._create_method_heatmap(ax2, matched_tokens, norm_att_scores, att_scores,
                                    'Raw Attention', show_scores)

        # Difference heatmap
        score_diff = np.array(norm_ig_scores) - np.array(norm_att_scores)
        self._create_difference_heatmap(ax3, matched_tokens, score_diff, show_scores)

        plt.tight_layout()
        return fig

    def create_ranking_comparison(self, sentence_id, top_k=8):
        """Create FIXED combined ranking comparison for a sentence"""

        matched_tokens, ig_scores, att_scores = self.get_sentence_data_with_alignment(sentence_id)

        if matched_tokens is None or len(matched_tokens) < 2:
            print(f"Cannot create ranking for {sentence_id} - insufficient matched tokens")
            return None

        # Get metrics for title
        metrics = self.calculate_alignment_metrics(sentence_id)

        # Create token-score dictionary for easy lookup
        token_scores = {}
        for token, ig_score, att_score in zip(matched_tokens, ig_scores, att_scores):
            token_scores[token] = {'ig': ig_score, 'att': att_score}

        # Get top tokens by combined importance (max of absolute values from both methods)
        combined_importance = {}
        for token in matched_tokens:
            combined_importance[token] = max(abs(token_scores[token]['ig']),
                                             abs(token_scores[token]['att']))

        # Sort by combined importance and take top k
        actual_k = min(top_k, len(matched_tokens))
        top_tokens = sorted(combined_importance.items(), key=lambda x: x[1], reverse=True)[:actual_k]
        top_token_names = [token for token, _ in top_tokens]

        # Create visualization with FIXED spacing
        fig = plt.figure(figsize=(14, 9))  # Made taller to accommodate titles
        ax = fig.add_subplot(111)

        # Create title and subtitle with PROPER spacing
        main_title = "XAI Methods Comparison: Raw Attention vs. Integrated Gradients"

        if metrics:
            # Calculate Jensen-Shannon divergence
            from scipy.spatial.distance import jensenshannon
            # Convert to probability distributions
            ig_probs = np.abs(ig_scores) / np.sum(np.abs(ig_scores)) if np.sum(np.abs(ig_scores)) > 0 else np.ones(
                len(ig_scores)) / len(ig_scores)
            att_probs = np.abs(att_scores) / np.sum(np.abs(att_scores)) if np.sum(np.abs(att_scores)) > 0 else np.ones(
                len(att_scores)) / len(att_scores)
            js_div = jensenshannon(ig_probs, att_probs)

            subtitle = (f"Sentence: {sentence_id} | "
                        f"Matched Tokens: {metrics['n_matched_tokens']} | "
                        f"Cosine: {metrics['cosine_similarity']:.3f} | "
                        f"Pearson: {metrics['pearson_correlation']:.3f} | "
                        f"Jensen-Shannon: {js_div:.3f}")
        else:
            subtitle = f"Sentence: {sentence_id}"

        # FIXED: Set titles with NO OVERLAP
        fig.suptitle(main_title, fontsize=14, fontweight='bold', y=0.97)
        ax.set_title(subtitle, fontsize=11, y=1.02)  # Move subtitle ABOVE the plot area

        # Prepare data for grouped bar chart with CLOSER bars
        y_positions = np.arange(len(top_token_names))
        bar_width = 0.4  # Thicker bars, closer together

        # Get scores for top tokens
        ig_scores_top = [abs(token_scores[token]['ig']) for token in top_token_names]
        att_scores_top = [abs(token_scores[token]['att']) for token in top_token_names]

        # FIXED: Use SIMPLE, consistent colors (no positive/negative distinction)
        ig_color = '#1f77b4'  # Standard blue
        att_color = '#87ceeb'  # Light blue

        # Create grouped horizontal bars with CLOSER spacing
        bars1 = ax.barh(y_positions - bar_width / 2, ig_scores_top, bar_width,
                        color=ig_color, alpha=0.9, edgecolor='white', linewidth=0.5)
        bars2 = ax.barh(y_positions + bar_width / 2, att_scores_top, bar_width,
                        color=att_color, alpha=0.9, edgecolor='white', linewidth=0.5)

        # Customize the plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels(top_token_names, fontsize=11)
        ax.set_xlabel('Absolute Importance Score', fontsize=12)
        ax.invert_yaxis()  # Top token at the top

        # Add score labels on bars
        max_width = max(max(ig_scores_top) if ig_scores_top else 0,
                        max(att_scores_top) if att_scores_top else 0)

        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            # IG score label
            ig_score = token_scores[top_token_names[i]]['ig']
            ax.text(bar1.get_width() + max_width * 0.02,
                    bar1.get_y() + bar1.get_height() / 2,
                    f'{ig_score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

            # Attention score label
            att_score = token_scores[top_token_names[i]]['att']
            ax.text(bar2.get_width() + max_width * 0.02,
                    bar2.get_y() + bar2.get_height() / 2,
                    f'{att_score:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')

        # Extend x-axis limit to accommodate score labels
        current_xlim = ax.get_xlim()
        ax.set_xlim(current_xlim[0], current_xlim[1] * 1.15)  # Add 15% more space on the right

        # FIXED: Create SIMPLE legend with just method names
        ax.legend(['Integrated Gradients', 'Raw Attention'],
                  loc='lower right', fontsize=12, framealpha=0.9)

        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)

        # FIXED: Adjust layout to prevent overlap
        plt.subplots_adjust(top=0.85, bottom=0.1)

        return fig

    def _create_method_heatmap(self, ax, tokens, norm_scores, raw_scores, method_name, show_scores):
        """Create heatmap for a single method"""
        n_tokens = len(tokens)

        # Create heatmap
        im = ax.imshow([norm_scores], cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)

        # Set labels
        ax.set_title(f'{method_name}', fontweight='bold', pad=20)
        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([])

        # Add score values on heatmap
        if show_scores and n_tokens <= 20:  # Only show scores if not too many tokens
            for i, (norm_score, raw_score) in enumerate(zip(norm_scores, raw_scores)):
                color = 'white' if abs(norm_score) > 0.5 else 'black'
                ax.text(i, 0, f'{raw_score:.3f}', ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)

        return im

    def _create_difference_heatmap(self, ax, tokens, score_diff, show_scores):
        """Create difference heatmap"""
        n_tokens = len(tokens)

        # Create difference heatmap
        max_diff = np.max(np.abs(score_diff)) if len(score_diff) > 0 else 1
        im = ax.imshow([score_diff], cmap='RdBu_r', aspect='auto',
                       vmin=-max_diff, vmax=max_diff)

        # Set labels
        ax.set_title('Difference (Integrated Gradients - Raw Attention)', fontweight='bold', pad=20)
        ax.set_xticks(range(n_tokens))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticks([])

        # Add difference values
        if show_scores and n_tokens <= 20:
            for i, diff in enumerate(score_diff):
                color = "white" if abs(diff) > 0.5 * max_diff else "black"
                ax.text(i, 0, f'{diff:.3f}', ha='center', va='center',
                        color=color, fontsize=12, fontweight='bold')

        # Add colorbar
        plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)

        return im


def main():
    """Main execution function"""
    print("=" * 80)
    print("FIXED IG vs RAW ATTENTION VISUALIZER")
    print("=" * 80)

    # Initialize visualizer
    visualizer = FixedIGAttentionVisualizer()

    # Load data
    if not visualizer.load_data():
        print("Failed to load data.")
        return

    # Analyze all sentences
    alignment_results = visualizer.analyze_all_sentences()

    if alignment_results.empty:
        print("No alignment results found.")
        return

    # Create visualizations for best aligned sentences
    print(f"\n{'=' * 60}")
    print("CREATING VISUALIZATIONS")
    print(f"{'=' * 60}")

    # Get top 3 sentences by different metrics
    top_pearson = alignment_results.nlargest(3, 'pearson_correlation')['sentence_id'].tolist()
    top_cosine = alignment_results.nlargest(3, 'cosine_similarity')['sentence_id'].tolist()

    # Combine and get unique
    sentences_to_visualize = list(set(top_pearson + top_cosine))[:3]

    created_files = []

    for sentence_id in sentences_to_visualize:
        print(f"\nCreating visualizations for {sentence_id}...")

        try:
            # Heatmap
            fig1 = visualizer.create_heatmap_comparison(sentence_id)
            if fig1:
                filename1 = f'Fixed_Heatmap_{sentence_id}.png'
                plt.savefig(filename1, dpi=300, bbox_inches='tight')
                created_files.append(filename1)
                print(f"  ✓ Saved: {filename1}")
                plt.show()

            # Rankings
            fig2 = visualizer.create_ranking_comparison(sentence_id)
            if fig2:
                filename2 = f'Fixed_Rankings_{sentence_id}.png'
                plt.savefig(filename2, dpi=300, bbox_inches='tight')
                created_files.append(filename2)
                print(f"  ✓ Saved: {filename2}")
                plt.show()

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'=' * 80}")
    print("ANALYSIS COMPLETE!")
    print(f"Created {len(created_files)} files:")
    for filename in created_files:
        print(f"  - {filename}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()


# Quick function to visualize a specific sentence
def visualize_sentence(sentence_id):
    """Quick function to visualize a specific sentence"""
    visualizer = FixedIGAttentionVisualizer()

    if visualizer.load_data():
        print(f"Creating visualizations for {sentence_id}...")

        # Heatmap
        fig1 = visualizer.create_heatmap_comparison(sentence_id)
        if fig1:
            plt.savefig(f'Heatmap_{sentence_id}.png', dpi=300, bbox_inches='tight')
            plt.show()

        # Rankings
        fig2 = visualizer.create_ranking_comparison(sentence_id)
        if fig2:
            plt.savefig(f'Rankings_{sentence_id}.png', dpi=300, bbox_inches='tight')
            plt.show()

# Example usage:
# visualize_sentence('M1')
# visualize_sentence('N3')