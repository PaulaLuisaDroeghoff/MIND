import sys
import os
import pandas as pd
from dotenv import load_dotenv
from crew import SentenceManipulationDetectorCrew
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

    print(f"Total sentences: {total}")
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
            'sentence_id': f'S{i + 1}',
            'text': result['sentence'],
            'predicted_label': result['final_classification'],
            'confidence': result['confidence'],
            'iterations': result['iterations']
        }

        # Add category if available from sampled_df
        if sampled_df is not None and 'category' in sampled_df.columns:
            row['category'] = sampled_df.iloc[i]['category']

        results_data.append(row)

    results_df = pd.DataFrame(results_data)

    # Save files
    results_file = f'crewai_agent_results_{timestamp}.csv'
    summary_file = f'crewai_agent_summary_{timestamp}.txt'

    results_df.to_csv(results_file, index=False)

    # Write summary
    with open(summary_file, 'w') as f:
        f.write("CrewAI Agent Analysis Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total Sentences: {len(results)}\n")
        f.write(f"Accuracy: {evaluation['accuracy']:.3f}\n")
        f.write(f"Precision: {evaluation['precision']:.3f}\n")
        f.write(f"Recall: {evaluation['recall']:.3f}\n")
        f.write(f"F1 Score: {evaluation['f1_score']:.3f}\n")
        f.write(f"Macro-F1 Score: {evaluation['macro_f1']:.3f}\n")
        f.write(f"Average Confidence: {evaluation['average_confidence']:.3f}\n")

    print(f"\n=== RESULTS SAVED ===")
    print(f"- {results_file} (detailed results)")
    print(f"- {summary_file} (performance summary)")

def test_balanced_classification():
    """Test with both manipulative and non-manipulative examples"""
    # Clear manipulation examples (should be 1)
    manipulation_examples = [
        "If you really loved me, you would do this for me.",  # Control attempt
        "Look what you made me do now.",  # Control attempt
        "You won't get any water and you won't get any rest.",  # Control attempt
        "I guess I'll just handle everything myself since nobody cares.",  # Control attempt
        "You’re too sensitive about this.",  # Control attempt
        "After everything I've done for you, this is how you repay me?",  # Control attempt
        "Fine, do whatever you want, I don't care anymore.",  # Control attempt
    ]

    # Clear non-manipulation examples (should be 0)
    non_manipulation_examples = [
        "I disagree with your proposal.",  # Direct disagreement
        "I'm disappointed with the results.",  # Honest emotion
        "That behavior is unacceptable.",  # Boundary setting
        "I need the report by Friday.",  # Direct request
        "I don't want to discuss this right now.",  # Boundary/refusal
        "The meeting is scheduled for 3 PM.",  # Factual statement
        "I prefer option A over option B.",  # Preference
    ]

    # Combine with expected labels
    test_data = []
    expected_labels = []

    for sentence in manipulation_examples:
        test_data.append(sentence)
        expected_labels.append(1)

    for sentence in non_manipulation_examples:
        test_data.append(sentence)
        expected_labels.append(0)

    crew_instance = SentenceManipulationDetectorCrew()
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
    for i, (sentence, pred, true, result) in enumerate(zip(test_data, predictions, expected_labels, results)):
        status = "✓" if pred == true else "✗"
        print(f"{status} {sentence[:50]}... | Pred: {pred}, True: {true}, Conf: {result['confidence']:.2f}")

    # Success criteria: at least 80% accuracy on both classes
    success = (manip_correct >= len(manipulation_examples) * 0.8 and
               non_manip_correct >= len(non_manipulation_examples) * 0.8)

    if success:
        print("BALANCED TEST PASSED: Good balance of precision and recall")
        return True
    else:
        print("BALANCED TEST FAILED: Need to adjust sensitivity")
        return False

def debug_agent_output():
    """Debug what the agent is actually saying"""
    crew_instance = SentenceManipulationDetectorCrew()

    # Test with a sentence that should clearly be 0
    test_sentence = "I disagree with your proposal."

    print(f"DEBUGGING AGENT OUTPUT")
    print(f"Testing: {test_sentence}")
    print(f"Expected: 0 (direct disagreement)")
    print(f"=" * 60)

    # Get raw output from classification step only
    from crewai import Crew, Process

    classification_crew = Crew(
        agents=[crew_instance.sentence_classifier()],
        tasks=[crew_instance.classify_sentence_task()],
        process=Process.sequential,
        verbose=False
    )

    result = classification_crew.kickoff(inputs={'sentence': test_sentence})

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
    crew_instance = SentenceManipulationDetectorCrew()
    print("\n" + "=" * 70)
    print("MENTAL MANIPULATION DETECTION SYSTEM")
    print("=" * 70 + "\n")

    # Note: Ensure sampled_sentences.csv is derived from MENTALMANIP dataset (e.g., sentences extracted from dialogues). Check https://github.com/audreycs/MentalManip.
    sampled_file_path = '/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/sentence_XAI/sampled_sentences.csv'

    print(f"Loading pre-sampled sentences from {sampled_file_path}...")
    try:
        sampled_df = pd.read_csv(sampled_file_path)
        is_balanced = analyze_dataset_balance(sampled_df)

        # Ensure balanced subset
        if len(sampled_df) > 50 or not is_balanced:
            print(f"\nDataset has {len(sampled_df)} sentences. Creating balanced subset of 50...")
            non_manip = sampled_df[sampled_df['true_label'] == 0]
            manip = sampled_df[sampled_df['true_label'] == 1]
            n_each = min(25, len(non_manip), len(manip))
            if n_each < 10:
                print(f"ERROR: Not enough examples (only {n_each} per class). Aborting.")
                return
            balanced_subset = pd.concat([
                non_manip.sample(n=n_each, random_state=None),
                manip.sample(n=n_each, random_state=None)
            ]).sample(frac=1).reset_index(drop=True)
            sampled_df = balanced_subset
            print(f"✓ Created balanced subset: {n_each} manipulative, {n_each} non-manipulative")

        sentences = sampled_df['text'].tolist()
        labels = sampled_df['true_label'].tolist()
        print(f"\n✓ Processing {len(sentences)} sentences")

        final_counts = sampled_df['true_label'].value_counts()
        print(f"Final distribution:")
        print(f"  - Non-Manipulative (0): {final_counts.get(0, 0)}")
        print(f"  - Manipulative (1): {final_counts.get(1, 0)}")

    except Exception as e:
        print(f"Error loading sampled sentences: {e}")
        return

    results = crew_instance.process_batch(sentences)

    # Log invalid outputs for debugging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    invalid_outputs = []
    for i, result in enumerate(results):
        # Check for invalid final classification (discussion_task)
        required_fields = ['final_classification', 'confidence', 'reasoning']
        if not all(key in result for key in required_fields):
            invalid_outputs.append({
                'sentence': sentences[i],
                'task': 'discussion_task',
                'raw_output': result.get('raw_output', 'No raw output available'),
                'error': f'Missing required fields: {", ".join([key for key in required_fields if key not in result])}'
            })
        # Check for invalid quality check output (if available)
        if 'quality_output' in result:  # Assuming crew stores quality_check_task output
            quality_output = result['quality_output']
            required_quality_fields = ['is_accurate', 'review_confidence', 'suggested_classification', 'feedback']
            if not all(key in quality_output for key in required_quality_fields):
                invalid_outputs.append({
                    'sentence': sentences[i],
                    'task': 'quality_check_task',
                    'raw_output': quality_output.get('raw_output', str(quality_output)),
                    'error': f'Missing required fields: {", ".join([key for key in required_quality_fields if key not in quality_output])}'
                })

    if invalid_outputs:
        invalid_df = pd.DataFrame(invalid_outputs)
        invalid_file = f'invalid_outputs_{timestamp}.csv'
        invalid_df.to_csv(invalid_file, index=False)
        print(f"WARNING: {len(invalid_outputs)} invalid outputs saved to {invalid_file}")

    predictions = []
    for result in results:
        if 'final_classification' in result:
            predictions.append(result['final_classification'])
        else:
            predictions.append(0)  # Fallback to avoid crashes
            print(f"WARNING: Using fallback classification 0 for sentence: {result.get('sentence', 'Unknown')}")

    evaluation = crew_instance.evaluate_performance(results, labels)
    evaluation['macro_f1'] = f1_score(labels, predictions, average='macro')

    print("\n" + "=" * 70)
    print("CLASSIFICATION RESULTS")
    print("=" * 70)
    classification_counts = {'manipulative': 0, 'non-manipulative': 0}
    for i, (sentence, result, true_label) in enumerate(zip(sentences, results, labels)):
        predicted_label = result.get('final_classification', 0)  # Fallback to 0 if missing
        print(f"\nSentence {i + 1}: {sentence[:60]}...")
        print(f"True Label: {true_label}")
        print(f"Predicted: {predicted_label}")
        print(f"Confidence: {result.get('confidence', 0.5):.2f}")
        print(f"Iterations Used: {result.get('iterations', 1)}")
        print(f"Processing Time: {result.get('processing_time', 0):.1f}s")
        print(f"Correct: {'✓' if predicted_label == true_label else '✗'}")
        if predicted_label == 1:
            classification_counts['manipulative'] += 1
        else:
            classification_counts['non-manipulative'] += 1

    print(f"\n{'=' * 70}")
    print(f"CLASSIFICATION DISTRIBUTION")
    print(f"{'=' * 70}")
    print(f"Total Manipulative: {classification_counts['manipulative']} ({classification_counts['manipulative'] / len(sentences) * 100:.1f}%)")
    print(f"Total Non-Manipulative: {classification_counts['non-manipulative']} ({classification_counts['non-manipulative'] / len(sentences) * 100:.1f}%)")

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
    print(f"Average Processing Time: {evaluation.get('average_processing_time', 0):.1f}s per sentence")
    print(f"Total Iterations Used: {evaluation.get('total_iterations_used', 0)}")
    print(f"Cost Efficiency: {evaluation.get('cost_efficiency', 0):.2f} iterations per sentence")
    print("\nConfusion Matrix:")
    cm = evaluation['confusion_matrix']
    print(f"  TP: {cm['true_positive']}  FP: {cm['false_positive']}")
    print(f"  FN: {cm['false_negative']}  TN: {cm['true_negative']}")
    save_results_for_comparison(sampled_df, results, evaluation)

if __name__ == "__main__":
    print("Running Debug and Test Functions")
    print("\n=== Debugging Agent Output ===")
    debug_agent_output()
    print("\n=== Testing Balanced Classification ===")
    test_balanced_classification()
    print("\n=== Running Full Pipeline ===")
    run()