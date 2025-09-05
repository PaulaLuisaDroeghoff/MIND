import sys
import os
import pandas as pd
from dotenv import load_dotenv
from crew import DialogueManipulationDetectorCrew
import datetime
from sklearn.metrics import f1_score

# Load environment variables from .env file
load_dotenv()


def analyze_dataset_balance(sampled_df):
    """Analyze and report the balance of the dataset"""
    print("\n" + "=" * 70)
    print("DATASET BALANCE ANALYSIS")
    print("=" * 70)

    label_counts = sampled_df['true_label'].value_counts().sort_index()
    total = len(sampled_df)

    print(f"Total dialogues: {total}")
    print(f"Non-Manipulative (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0) / total * 100:.1f}%)")
    print(f"Manipulative (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0) / total * 100:.1f}%)")

    if label_counts.get(0, 0) == 0 or label_counts.get(1, 0) == 0:
        print("ERROR: Dataset is completely imbalanced (missing one class)!")
        return False
    else:
        ratio = label_counts.get(1, 0) / label_counts.get(0, 0)
        print(f"\nRatio (manipulative/non-manipulative): {ratio:.2f}")
        if ratio > 2 or ratio < 0.5:
            print("WARNING: Dataset is highly imbalanced!")
            return False
    return True


def save_results_for_comparison(sampled_df, results, evaluation):
    """Save results in a format that can be compared with XAI methods"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create results dataframe
    results_data = []
    for i, result in enumerate(results):
        row = {
            'dialogue_id': f'D{i + 1}',
            'dialogue_text': result['dialogue'][:200] + "..." if len(result['dialogue']) > 200 else result['dialogue'],
            'predicted_label': result['final_classification'],
            'confidence': result['confidence'],
            'iterations': result['iterations'],
            'dialogue_length': len(result['dialogue'].split()),
            'techniques_found': ', '.join(result.get('techniques_found', []))
        }

        # Add category if available from sampled_df
        if sampled_df is not None and 'category' in sampled_df.columns:
            row['category'] = sampled_df.iloc[i]['category']

        results_data.append(row)

    results_df = pd.DataFrame(results_data)

    # Save files
    results_file = f'crewai_dialogue_results_{timestamp}.csv'
    summary_file = f'crewai_dialogue_summary_{timestamp}.txt'

    results_df.to_csv(results_file, index=False)

    # Write summary
    with open(summary_file, 'w') as f:
        f.write("CrewAI Dialogue Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Dialogues: {len(results)}\n")
        f.write(f"Accuracy: {evaluation['accuracy']:.3f}\n")
        f.write(f"Precision: {evaluation['precision']:.3f}\n")
        f.write(f"Recall: {evaluation['recall']:.3f}\n")
        f.write(f"F1 Score: {evaluation['f1_score']:.3f}\n")
        f.write(f"Macro-F1 Score: {evaluation['macro_f1']:.3f}\n")
        f.write(f"Average Confidence: {evaluation['average_confidence']:.3f}\n")
        f.write(
            f"Average Dialogue Length: {sum(len(r['dialogue'].split()) for r in results) / len(results):.1f} words\n")

    print(f"\n=== RESULTS SAVED ===")
    print(f"- {results_file} (detailed results)")
    print(f"- {summary_file} (performance summary)")


def test_balanced_classification():
    """Test with both manipulative and non-manipulative dialogue examples"""
    # Clear manipulation examples (should be 1)
    manipulation_examples = [
        """Person A: I need you to help me with this project.
Person B: I'm really busy this week, maybe next week?
Person A: If you really cared about our friendship, you would make time for me. I guess I know where I stand now.
Person B: That's not fair...
Person A: I'm just saying, real friends prioritize each other.""",

        """Person A: Why didn't you call me back yesterday?
Person B: I was dealing with a family emergency.
Person A: You always have some excuse. You're just like everyone else who abandons me when I need them most.
Person B: That's not true...
Person A: Whatever, I should have known better than to trust you.""",

        """Person A: Can you lend me some money?
Person B: I'm tight on cash right now.
Person A: After everything I've done for you, this is how you repay me? I guess I'll just figure it out myself like I always do.
Person B: I feel bad but...
Person A: Don't worry about it. I'll remember this."""
    ]

    # Clear non-manipulation examples (should be 0)
    non_manipulation_examples = [
        """Person A: I disagree with your proposal for the marketing campaign.
Person B: What specifically concerns you?
Person A: I think we should focus more on digital platforms rather than traditional media.
Person B: That's a valid point. Let's discuss the budget implications.""",

        """Person A: I'm disappointed with how the meeting went today.
Person B: What happened?
Person A: I felt like my ideas weren't heard properly.
Person B: I'm sorry you felt that way. Can we schedule time to discuss them properly?""",

        """Person A: I need the report by Friday for the client meeting.
Person B: That's a tight deadline with everything else going on.
Person A: I understand it's challenging, but the client is expecting it.
Person B: Okay, I'll prioritize it and get it done."""
    ]

    # Combine with expected labels
    test_data = []
    expected_labels = []

    for dialogue in manipulation_examples:
        test_data.append(dialogue)
        expected_labels.append(1)

    for dialogue in non_manipulation_examples:
        test_data.append(dialogue)
        expected_labels.append(0)

    crew_instance = DialogueManipulationDetectorCrew()
    results = crew_instance.process_batch(test_data)

    # Evaluate results
    predictions = [r['final_classification'] for r in results]

    manip_correct = sum(1 for i in range(len(manipulation_examples))
                        if predictions[i] == 1)
    non_manip_correct = sum(1 for i in range(len(manipulation_examples), len(test_data))
                            if predictions[i] == 0)

    total_correct = sum(1 for pred, true in zip(predictions, expected_labels)
                        if pred == true)
    accuracy = total_correct / len(test_data)
    macro_f1 = f1_score(expected_labels, predictions, average='macro')

    print(f"\n=== BALANCED TEST RESULTS ===")
    print(
        f"Manipulation examples correct: {manip_correct}/{len(manipulation_examples)} ({manip_correct / len(manipulation_examples) * 100:.1f}%)")
    print(
        f"Non-manipulation examples correct: {non_manip_correct}/{len(non_manipulation_examples)} ({non_manip_correct / len(non_manipulation_examples) * 100:.1f}%)")
    print(f"Overall accuracy: {accuracy:.3f}")
    print(f"Macro-F1 Score: {macro_f1:.3f}")

    # Show detailed results
    print(f"\n=== DETAILED RESULTS ===")
    for i, (dialogue, pred, true, result) in enumerate(zip(test_data, predictions, expected_labels, results)):
        status = "✓" if pred == true else "✗"
        dialogue_preview = dialogue.replace('\n', ' ')[:80] + "..." if len(dialogue) > 80 else dialogue.replace('\n',
                                                                                                                ' ')
        print(f"{status} {dialogue_preview} | Pred: {pred}, True: {true}, Conf: {result['confidence']:.2f}")

    # Success criteria: at least 80% accuracy on both classes
    success = (manip_correct >= len(manipulation_examples) * 0.8 and
               non_manip_correct >= len(non_manipulation_examples) * 0.8)

    if success:
        print("BALANCED TEST PASSED: Good balance of precision and recall")
        return True
    else:
        print("BALANCED TEST FAILED: Need to adjust sensitivity")
        return False


def create_balanced_dataset(sampled_df, target_size=30):
    """Create a perfectly balanced dataset with equal numbers of each class"""

    print(f"\n{'=' * 70}")
    print(f"CREATING PERFECTLY BALANCED DATASET")
    print(f"{'=' * 70}")

    # Separate by label
    non_manip = sampled_df[sampled_df['true_label'] == 0]
    manip = sampled_df[sampled_df['true_label'] == 1]

    print(f"Available data:")
    print(f"  - Non-Manipulative (0): {len(non_manip)} dialogues")
    print(f"  - Manipulative (1): {len(manip)} dialogues")

    # Calculate exactly half for each class
    n_each = target_size // 2  # This ensures 50/50 split
    max_possible = min(len(non_manip), len(manip), n_each)

    if max_possible < 5:
        print(f"ERROR: Not enough examples of both classes (only {max_possible} each). Need at least 5 each.")
        return None

    if max_possible < n_each:
        print(f"WARNING: Can only use {max_possible} of each class (wanted {n_each})")
        n_each = max_possible

    # Sample exactly n_each from each class
    balanced_non_manip = non_manip.sample(n=n_each, random_state=42)
    balanced_manip = manip.sample(n=n_each, random_state=42)

    # Combine and shuffle
    balanced_df = pd.concat([balanced_non_manip, balanced_manip]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Verify perfect balance
    final_counts = balanced_df['true_label'].value_counts()
    print(f"\n✓ Created perfectly balanced dataset:")
    print(f"  - Total dialogues: {len(balanced_df)}")
    print(
        f"  - Non-Manipulative (0): {final_counts.get(0, 0)} ({final_counts.get(0, 0) / len(balanced_df) * 100:.1f}%)")
    print(f"  - Manipulative (1): {final_counts.get(1, 0)} ({final_counts.get(1, 0) / len(balanced_df) * 100:.1f}%)")
    print(f"  - Perfect 50/50 balance: {'✓' if final_counts.get(0, 0) == final_counts.get(1, 0) else '✗'}")

    return balanced_df

def debug_agent_output():
    """Debug what the agent is actually saying"""
    crew_instance = DialogueManipulationDetectorCrew()

    # Test with a dialogue that should clearly be 0
    test_dialogue = """Person A: I disagree with your proposal for the marketing campaign.
Person B: What specifically concerns you?
Person A: I think we should focus more on digital platforms rather than traditional media.
Person B: That's a valid point. Let's discuss the budget implications."""

    print(f"DEBUGGING AGENT OUTPUT")
    print(f"Testing: {test_dialogue[:100]}...")
    print(f"Expected: 0 (professional disagreement)")
    print(f"=" * 60)

    # Get raw output from classification step only
    from crewai import Crew, Process

    classification_crew = Crew(
        agents=[crew_instance.dialogue_classifier()],
        tasks=[crew_instance.classify_dialogue_task()],
        process=Process.sequential,
        verbose=False
    )

    result = classification_crew.kickoff(inputs={'dialogue': test_dialogue})

    print(f"RAW AGENT OUTPUT:")
    print(f"=" * 60)
    print(result)
    print(f"=" * 60)

    # Test parsing
    parsed = crew_instance._parse_classification(str(result))
    print(f"PARSED RESULT: {parsed}")

    # Check if the agent mentioned key phrases
    result_str = str(result).lower()
    if "disagree" in result_str:
        print("✓ Agent recognized it's about disagreement")
    else:
        print("Agent didn't recognize disagreement")

    if "not manipulative" in result_str or "non-manipulative" in result_str:
        print("✓ Agent said it's not manipulative")
    else:
        print("Agent didn't clearly say it's not manipulative")

    return str(result)


def run():
    crew_instance = DialogueManipulationDetectorCrew()
    print("\n" + "=" * 70)
    print("DIALOGUE MENTAL MANIPULATION DETECTION SYSTEM")
    print("=" * 70 + "\n")

    # CHANGE THIS PATH TO YOUR ACTUAL EXCEL FILE
    excel_file_path = '/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/data/dialoguelevel_mentalmanip_detailed.xlsx'  # Adjust path as needed

    print(f"Loading dialogues from Excel file: {excel_file_path}...")
    try:
        # Load Excel file with your specific column names
        sampled_df = pd.read_excel(excel_file_path)

        # Rename columns to match our code expectations
        if 'dialogue' in sampled_df.columns and 'Manipulative' in sampled_df.columns:
            sampled_df = sampled_df.rename(columns={'Manipulative': 'true_label'})
            print(f"✓ Found columns 'dialogue' and 'Manipulative'")
        else:
            print(f"Available columns: {list(sampled_df.columns)}")
            raise ValueError("Expected columns 'dialogue' and 'Manipulative' not found in Excel file")

        # Show basic info about the data
        print(f"✓ Loaded {len(sampled_df)} dialogues from Excel file")
        print(f"  - Columns found: {list(sampled_df.columns)}")
        print(f"  - Sample dialogue length: {len(str(sampled_df['dialogue'].iloc[0]).split())} words")

        is_balanced = analyze_dataset_balance(sampled_df)

        # Create perfectly balanced dataset
        target_size = 30  # Change this number as needed (will be split 50/50)
        print(f"\nOriginal dataset size: {len(sampled_df)}")

        balanced_df = create_balanced_dataset(sampled_df, target_size)
        if balanced_df is None:
            print("Cannot create balanced dataset. Exiting.")
            return

        sampled_df = balanced_df

        # Assuming dialogue text is in 'dialogue' column (which should exist)
        dialogues = sampled_df['dialogue'].tolist()
        labels = sampled_df['true_label'].tolist()  # Now using renamed 'Manipulative' column
        print(f"\n✓ Processing {len(dialogues)} dialogues")

        final_counts = sampled_df['true_label'].value_counts()
        print(f"Final distribution:")
        print(f"  - Non-Manipulative (0): {final_counts.get(0, 0)}")
        print(f"  - Manipulative (1): {final_counts.get(1, 0)}")

        # Show dialogue length statistics
        dialogue_lengths = [len(d.split()) for d in dialogues]
        avg_length = sum(dialogue_lengths) / len(dialogue_lengths)
        print(f"  - Average dialogue length: {avg_length:.1f} words")
        print(f"  - Length range: {min(dialogue_lengths)} - {max(dialogue_lengths)} words")

    except Exception as e:
        print(f"Error loading Excel file: {e}")
        print(f"\nTroubleshooting tips:")
        print(f"1. Make sure the Excel file path is correct")
        print(f"2. Ensure the file has columns named 'dialogue' and 'Manipulative'")
        print(f"3. Check that the file is not open in Excel")
        print(f"4. Verify you have openpyxl installed: pip install openpyxl")
        return

    results = crew_instance.process_batch(dialogues)

    # Log invalid outputs for debugging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    invalid_outputs = []
    for i, result in enumerate(results):
        # Check for invalid final classification (discussion_task)
        required_fields = ['final_classification', 'confidence', 'reasoning']
        if not all(key in result for key in required_fields):
            invalid_outputs.append({
                'dialogue': dialogues[i][:100] + "..." if len(dialogues[i]) > 100 else dialogues[i],
                'task': 'discussion_task',
                'raw_output': result.get('raw_output', 'No raw output available'),
                'error': f'Missing required fields: {", ".join([key for key in required_fields if key not in result])}'
            })

    if invalid_outputs:
        invalid_df = pd.DataFrame(invalid_outputs)
        invalid_file = f'invalid_dialogue_outputs_{timestamp}.csv'
        invalid_df.to_csv(invalid_file, index=False)
        print(f"WARNING: {len(invalid_outputs)} invalid outputs saved to {invalid_file}")

    predictions = []
    for result in results:
        if 'final_classification' in result:
            predictions.append(result['final_classification'])
        else:
            predictions.append(0)  # Fallback to avoid crashes
            print(f"WARNING: Using fallback classification 0 for dialogue: {result.get('dialogue', 'Unknown')[:50]}...")

    evaluation = crew_instance.evaluate_performance(results, labels)
    evaluation['macro_f1'] = f1_score(labels, predictions, average='macro')

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    classification_counts = {'manipulative': 0, 'non-manipulative': 0}
    for i, (dialogue, result, true_label) in enumerate(zip(dialogues, results, labels)):
        predicted_label = result.get('final_classification', 0)
        dialogue_preview = dialogue.replace('\n', ' ')[:100] + "..." if len(dialogue) > 100 else dialogue.replace('\n',
                                                                                                                  ' ')
        print(f"\nDialogue {i + 1}: {dialogue_preview}")
        print(f"True Label: {true_label}")
        print(f"Predicted: {predicted_label}")
        print(f"Confidence: {result.get('confidence', 0.5):.2f}")
        print(f"Iterations Used: {result.get('iterations', 1)}")
        print(f"Processing Time: {result.get('processing_time', 0):.1f}s")
        print(f"Techniques Found: {', '.join(result.get('techniques_found', ['None']))}")
        print(f"Correct: {'✓' if predicted_label == true_label else '✗'}")
        if predicted_label == 1:
            classification_counts['manipulative'] += 1
        else:
            classification_counts['non-manipulative'] += 1

    print(f"\n{'=' * 70}")
    print(f"CLASSIFICATION DISTRIBUTION")
    print(f"{'=' * 70}")
    print(
        f"Total Manipulative: {classification_counts['manipulative']} ({classification_counts['manipulative'] / len(dialogues) * 100:.1f}%)")
    print(
        f"Total Non-Manipulative: {classification_counts['non-manipulative']} ({classification_counts['non-manipulative'] / len(dialogues) * 100:.1f}%)")

    print("\n" + "=" * 70)
    print("PERFORMANCE METRICS")
    print("=" * 70)
    print(f"Accuracy: {evaluation['accuracy']:.3f}")
    print(f"Precision: {evaluation['precision']:.3f}")
    print(f"Recall: {evaluation['recall']:.3f}")
    print(f"F1 Score: {evaluation['f1_score']:.3f}")
    print(f"Macro-F1 Score: {evaluation['macro_f1']:.3f}")
    print(f"Average Confidence: {evaluation['average_confidence']:.3f}")
    print(f"\nEfficiency Metrics:")
    print(f"Average Processing Time: {evaluation.get('average_processing_time', 0):.1f}s per dialogue")
    print(f"Total Iterations Used: {evaluation.get('total_iterations_used', 0)}")
    print(f"Cost Efficiency: {evaluation.get('cost_efficiency', 0):.2f} iterations per dialogue")
    print("\nConfusion Matrix:")
    cm = evaluation['confusion_matrix']
    print(f"  TP: {cm['true_positive']}  FP: {cm['false_positive']}")
    print(f"  FN: {cm['false_negative']}  TN: {cm['true_negative']}")

    # Dialogue-specific insights
    print(f"\nDialogue-Specific Insights:")
    avg_length = sum(len(r['dialogue'].split()) for r in results) / len(results)
    print(f"Average dialogue length: {avg_length:.1f} words")

    technique_counts = {}
    for result in results:
        for tech in result.get('techniques_found', []):
            technique_counts[tech] = technique_counts.get(tech, 0) + 1

    if technique_counts:
        print(f"Most common manipulation techniques found:")
        sorted_techniques = sorted(technique_counts.items(), key=lambda x: x[1], reverse=True)
        for tech, count in sorted_techniques[:5]:
            print(f"  - {tech}: {count} dialogues")

    save_results_for_comparison(sampled_df, results, evaluation)


if __name__ == "__main__":
    print("Running Debug and Test Functions for Dialogue Processing")
    print("\n=== Debugging Agent Output ===")
    debug_agent_output()
    print("\n=== Testing Balanced Classification ===")
    test_balanced_classification()
    print("\n=== Running Full Dialogue Pipeline ===")
    run()