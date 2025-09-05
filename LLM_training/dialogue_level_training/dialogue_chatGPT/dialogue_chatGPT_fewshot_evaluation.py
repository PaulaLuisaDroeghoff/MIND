import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


def compare_dialogue_gpt_to_original(gpt_results_file, excel_file, dialogue_column, label_column):
    """
    Compare GPT-4 dialogue results to original annotations.

    Args:
        gpt_results_file: Path to your dialogue_manipulation_results_balanced.json
        excel_file: Path to your original Excel file with dialogues
        dialogue_column: Name of column with dialogues (e.g., 'Dialogue', 'Text')
        label_column: Name of column with labels (e.g., 'Manipulative', 'Label')
    """

    # Load GPT-4 results
    print("Loading GPT-4 dialogue results...")
    with open(gpt_results_file, 'r') as f:
        gpt_results = json.load(f)

    # Load original data
    print("Loading original dialogue annotations...")
    df_original = pd.read_excel(excel_file)

    print(f"GPT-4 results: {len(gpt_results)} dialogues")
    print(f"Original data: {len(df_original)} dialogues")

    # Extract GPT-4 predictions and dialogues
    gpt_dialogues = [result['dialogue_text'] for result in gpt_results]
    gpt_predictions = [result['is_manipulative'] for result in gpt_results]
    gpt_confidences = [result['confidence'] for result in gpt_results]

    # Create comparison lists
    true_labels = []
    matched_gpt_predictions = []
    matched_confidences = []

    # Match dialogues (in case order is different)
    print("Matching dialogues...")
    for i, gpt_dialogue in enumerate(gpt_dialogues):
        # Find this dialogue in original data
        # Clean both dialogues for comparison (remove extra whitespace)
        gpt_clean = gpt_dialogue.strip().lower()

        found = False
        for _, row in df_original.iterrows():
            original_clean = str(row[dialogue_column]).strip().lower()

            # Check if dialogues match (allowing for minor differences)
            # For dialogues, we need more flexible matching since they can be longer
            if (gpt_clean == original_clean or
                    gpt_clean in original_clean or
                    original_clean in gpt_clean or
                    # Check if first 100 characters match (for very long dialogues)
                    gpt_clean[:100] == original_clean[:100]):
                # Convert label: 1 -> True, 0 -> False
                true_label = bool(row[label_column])
                true_labels.append(true_label)
                matched_gpt_predictions.append(gpt_predictions[i])
                matched_confidences.append(gpt_confidences[i])
                found = True
                break

        if not found:
            print(f"Warning: Could not match dialogue {i + 1}: '{gpt_dialogue[:50]}...'")

    print(f"Successfully matched: {len(true_labels)} dialogues")

    if len(true_labels) == 0:
        print("No dialogues could be matched! Check your column names.")
        return

    # Calculate metrics
    accuracy = accuracy_score(true_labels, matched_gpt_predictions)
    precision = precision_score(true_labels, matched_gpt_predictions)
    recall = recall_score(true_labels, matched_gpt_predictions)
    f1 = f1_score(true_labels, matched_gpt_predictions)

    # Confusion matrix
    cm = confusion_matrix(true_labels, matched_gpt_predictions)
    tn, fp, fn, tp = cm.ravel()

    # Print results
    print("\n" + "=" * 50)
    print("GPT-4 DIALOGUE-LEVEL PERFORMANCE RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy:.3f} ({accuracy:.1%})")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print(f"Total Matched: {len(true_labels)} dialogues")

    print(f"\nConfusion Matrix:")
    print(f"True Negatives:  {tn} (Correctly identified non-manipulative)")
    print(f"False Positives: {fp} (Incorrectly said manipulative)")
    print(f"False Negatives: {fn} (Missed manipulative dialogues)")
    print(f"True Positives:  {tp} (Correctly identified manipulative)")

    # Distribution analysis
    original_manipulative = sum(true_labels)
    gpt_manipulative = sum(matched_gpt_predictions)

    print(f"\nLabel Distribution:")
    print(
        f"Original data: {original_manipulative}/{len(true_labels)} manipulative ({original_manipulative / len(true_labels):.1%})")
    print(
        f"GPT-4 predicted: {gpt_manipulative}/{len(true_labels)} manipulative ({gpt_manipulative / len(true_labels):.1%})")

    # Confidence analysis
    avg_confidence = np.mean(matched_confidences)
    confidence_correct = [matched_confidences[i] for i in range(len(true_labels))
                          if true_labels[i] == matched_gpt_predictions[i]]
    confidence_wrong = [matched_confidences[i] for i in range(len(true_labels))
                        if true_labels[i] != matched_gpt_predictions[i]]

    print(f"\nConfidence Analysis:")
    print(f"Average confidence: {avg_confidence:.3f}")
    if confidence_correct:
        print(f"Confidence when correct: {np.mean(confidence_correct):.3f}")
    if confidence_wrong:
        print(f"Confidence when wrong: {np.mean(confidence_wrong):.3f}")

    # Examples of mistakes
    print(f"\nExample Mistakes:")
    mistake_count = 0
    for i in range(len(true_labels)):
        if true_labels[i] != matched_gpt_predictions[i] and mistake_count < 3:
            true_str = "Manipulative" if true_labels[i] else "Non-manipulative"
            pred_str = "Manipulative" if matched_gpt_predictions[i] else "Non-manipulative"
            print(f"Dialogue: '{gpt_dialogues[i][:150]}...'")
            print(f"True: {true_str}, GPT-4: {pred_str} (confidence: {matched_confidences[i]:.2f})")
            print()
            mistake_count += 1

    # Save matched data for analysis
    print("Saving data for analysis...")
    matched_data = []

    for i in range(len(true_labels)):
        matched_data.append({
            'dialogue_text': gpt_dialogues[i] if i < len(gpt_dialogues) else '',
            'dialogue_id': i + 1,
            'gpt_prediction': matched_gpt_predictions[i],
            'true_label': true_labels[i],
            'confidence': matched_confidences[i],
            'is_correct': matched_gpt_predictions[i] == true_labels[i]
        })

    # Save to JSON for analysis
    with open('chatGPT_dialogue_fewshot_evaluation.json', 'w') as f:
        json.dump(matched_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(matched_data)} matched dialogues to 'chatGPT_dialogue_fewshot_evaluation.json'")
    print("You can now run analysis with this file!")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'total_matched': len(true_labels),
        'confusion_matrix': cm,
        'matched_data_file': 'chatGPT_dialogue_fewshot_evaluation.json'
    }


# Example usage - UPDATE THESE PATHS AND COLUMN NAMES
if __name__ == "__main__":
    # Update these paths and column names for your dialogue data
    gpt_file = "/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/LLM_training/dialogue_level_training/dialogue_chatGPT/dialogue_manipulation_results_fewshot_balanced.json"
    excel_file = "/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/data/dialoguelevel_mentalmanip_detailed.xlsx"
    dialogue_col = "dialogue"
    label_col = "Manipulative"

    # Run the comparison
    results = compare_dialogue_gpt_to_original(gpt_file, excel_file, dialogue_col, label_col)

    print(f"Results Summary:")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"F1-Score: {results['f1']:.3f}")
    print(f"data saved for analysis: {results['matched_data_file']}")