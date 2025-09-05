import json
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.model_selection import train_test_split

# OpenAI Configuration
client = OpenAI(
    api_key=""  # Replace with your actual OpenAI API key
)

# Enhanced manipulation definition
MANIPULATION_DEFINITION = """
Mental manipulation is using language to influence, alter, or control an individual's 
psychological state or perception for the manipulator's benefit.

This includes deceptive strategies aimed at controlling or altering someone's thoughts and 
feelings to serve the speaker's personal objectives. 

IMPORTANT: Mental manipulation is more insidious, deceitful, and context-dependent than 
overt verbal toxicity. Unlike direct toxicity (profanity, hate speech), manipulation uses 
SUBTLE psychological control tactics.

Key characteristics from research:
- Often appears in conversational context rather than isolated statements
- May seem reasonable on surface but has manipulative intent underneath
- Targets victim's psychological vulnerabilities for the manipulator's benefit
- Can include techniques like: denial, evasion, feigning innocence, rationalization, 
  playing victim/servant roles, shaming, intimidation, brandishing anger, accusation, 
  and persuasion/seduction

Even seemingly mild statements can be manipulative if they:
- Undermine someone's autonomy or decision-making
- Create emotional dependency or obligation
- Distort the victim's perception of reality
- Exploit psychological vulnerabilities (naivety, low self-esteem, over-responsibility, etc.)
- Use emotional tactics to bypass logical reasoning

CRITICAL: Be highly sensitive to subtle emotional control, even in polite or reasonable-sounding language.
"""


def select_training_examples(df_train, k_manip=3, k_nonmanip=3, random_state=None):
    """
    Select actual training examples for few-shot prompting.

    Args:
        df_train: Training dataframe
        k_manip: Number of manipulative examples
        k_nonmanip: Number of non-manipulative examples
        random_state: Random seed for reproducibility
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get manipulative examples
    manip_indices = df_train[df_train['Manipulative'] == 1].index.tolist()
    if len(manip_indices) < k_manip:
        print(f"Warning: Only {len(manip_indices)} manipulative training examples available")
        k_manip = len(manip_indices)
    selected_manip_idx = np.random.choice(manip_indices, k_manip, replace=False)
    manip_examples = df_train.loc[selected_manip_idx]

    # Get non-manipulative examples
    nonmanip_indices = df_train[df_train['Manipulative'] == 0].index.tolist()
    if len(nonmanip_indices) < k_nonmanip:
        print(f"Warning: Only {len(nonmanip_indices)} non-manipulative training examples available")
        k_nonmanip = len(nonmanip_indices)
    selected_nonmanip_idx = np.random.choice(nonmanip_indices, k_nonmanip, replace=False)
    nonmanip_examples = df_train.loc[selected_nonmanip_idx]

    return manip_examples, nonmanip_examples


def create_fewshot_prompt(manip_examples, nonmanip_examples, test_sentence):
    """
    Create a few-shot prompt with actual training examples.
    """

    examples_text = "Here are some examples from this dataset:\n\n"

    # Add manipulative examples
    examples_text += "MANIPULATIVE EXAMPLES:\n"
    for idx, row in manip_examples.iterrows():
        sentence = str(row['Sentence']).strip()
        examples_text += f"- \"{sentence}\" → MANIPULATIVE\n"

    examples_text += "\nNON-MANIPULATIVE EXAMPLES:\n"
    # Add non-manipulative examples
    for idx, row in nonmanip_examples.iterrows():
        sentence = str(row['Sentence']).strip()
        examples_text += f"- \"{sentence}\" → NON-MANIPULATIVE\n"

    examples_text += f"\nNow analyze this sentence:\n\"{test_sentence}\"\n\nIs this sentence manipulative or non-manipulative?"

    return examples_text


def classify_sentence_with_fewshot(sentence_text, manip_examples, nonmanip_examples):
    """
    Classify a sentence using zero-shot prompting (changed from few-shot).
    """
    schema = {
        "type": "object",
        "properties": {
            "is_manipulative": {
                "type": "boolean",
                "description": "Whether the sentence contains mental manipulation"
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence level (0-1) in the classification"
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of the classification decision"
            }
        },
        "required": ["is_manipulative", "confidence", "reasoning"],
        "additionalProperties": False
    }

    system_prompt = f"""You are an expert at identifying mental manipulation in language.

{MANIPULATION_DEFINITION}

Analyze whether the given sentence contains mental manipulation based on the definition and characteristics provided above.

Provide your analysis in JSON format with:
1. is_manipulative: true/false
2. confidence: 0-1 score
3. reasoning: brief explanation

Be highly sensitive to subtle emotional control tactics as shown in the definition."""

    user_prompt = f"Now analyze this sentence:\n\"{sentence_text}\"\n\nIs this sentence manipulative or non-manipulative?"

    try:
        response = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],
            model="gpt-4.1-mini",
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "manipulation_classification",
                    "schema": schema
                }
            },
            max_tokens=1000,
            temperature=0.1  # Slightly more flexible than 0, like the paper
        )

        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        return {"error": f"Classification failed: {str(e)}"}


def run_improved_sentence_classification(excel_file, sentence_column, label_column, output_file,
                                         n_test_per_class=100, k_manip=3, k_nonmanip=3,
                                         train_ratio=0.7, batch_size=25):
    """
    Run improved sentence classification with few-shot training examples.

    Args:
        excel_file (str): Path to Excel file
        sentence_column (str): Name of column containing sentences
        label_column (str): Name of column containing true labels
        output_file (str): Path to save JSON results
        n_test_per_class (int): Number of test sentences per class
        k_manip (int): Number of manipulative training examples to use
        k_nonmanip (int): Number of non-manipulative training examples to use
        train_ratio (float): Ratio of data to use for training examples
        batch_size (int): Save progress every N sentences
    """

    results = []

    try:
        # Read sentences from Excel file
        df = pd.read_excel(excel_file)
        print(f"Loaded {len(df)} sentences from Excel file")
        print(f"Available columns: {list(df.columns)}")

        # Clean data
        df = df.dropna(subset=[sentence_column, label_column])
        df[sentence_column] = df[sentence_column].astype(str).str.strip()

        # Remove empty sentences
        df = df[df[sentence_column] != '']

        print(f"After cleaning: {len(df)} sentences")

        # Split into train and test
        manipulative_df = df[df[label_column] == 1]
        non_manipulative_df = df[df[label_column] == 0]

        print(f"Available: {len(manipulative_df)} manipulative, {len(non_manipulative_df)} non-manipulative")

        # Split each class into train/test
        manip_train, manip_test = train_test_split(
            manipulative_df, train_size=train_ratio, random_state=42, stratify=None
        )
        nonmanip_train, nonmanip_test = train_test_split(
            non_manipulative_df, train_size=train_ratio, random_state=42, stratify=None
        )

        # Create training and test sets
        df_train = pd.concat([manip_train, nonmanip_train]).reset_index(drop=True)
        df_test_full = pd.concat([manip_test, nonmanip_test]).reset_index(drop=True)

        print(f"Training set: {len(df_train)} sentences")
        print(f"Available for testing: {len(df_test_full)} sentences")

        # Sample balanced test set
        n_manip_test = min(n_test_per_class, len(manip_test))
        n_nonmanip_test = min(n_test_per_class, len(nonmanip_test))

        sampled_manip_test = manip_test.sample(n=n_manip_test, random_state=42)
        sampled_nonmanip_test = nonmanip_test.sample(n=n_nonmanip_test, random_state=42)

        df_test = pd.concat([sampled_manip_test, sampled_nonmanip_test])
        df_test = df_test.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"Test set: {n_manip_test} manipulative + {n_nonmanip_test} non-manipulative = {len(df_test)} total")

        # Select training examples for few-shot prompting
        manip_examples, nonmanip_examples = select_training_examples(
            df_train, k_manip=k_manip, k_nonmanip=k_nonmanip, random_state=42
        )

        print(
            f"\nUsing {len(manip_examples)} manipulative and {len(nonmanip_examples)} non-manipulative training examples")
        print("Manipulative examples:")
        for idx, row in manip_examples.iterrows():
            print(f"  - {str(row[sentence_column])[:100]}...")
        print("Non-manipulative examples:")
        for idx, row in nonmanip_examples.iterrows():
            print(f"  - {str(row[sentence_column])[:100]}...")

        print(f"\nProcessing {len(df_test)} test sentences...")

        # Process each test sentence
        for i, row in df_test.iterrows():
            sentence_text = str(row[sentence_column])
            true_label = row[label_column]

            result = classify_sentence_with_fewshot(sentence_text, manip_examples, nonmanip_examples)

            # Add metadata
            result['sentence_id'] = i + 1
            result['sentence_text'] = sentence_text
            result['true_label'] = bool(true_label)
            results.append(result)

            print(f"Processed sentence {i + 1}/{len(df_test)}: {sentence_text[:50]}...")

            # Save progress incrementally
            if (i + 1) % batch_size == 0:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Saved progress: {i + 1} sentences completed")

        # Final save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Summary statistics
        successful_results = [r for r in results if 'error' not in r]
        manipulative_count = sum(1 for r in successful_results if r.get('is_manipulative', False))
        true_manipulative = sum(1 for r in successful_results if r.get('true_label', False))
        true_non_manipulative = len(successful_results) - true_manipulative

        print(f"\nSentence Classification Complete!")
        print(f"Total processed: {len(results)}")
        print(f"Successful: {len(successful_results)}")
        print(f"\nTrue labels (balanced sample):")
        print(f"  Manipulative: {true_manipulative}")
        print(f"  Non-manipulative: {true_non_manipulative}")
        print(f"\nGPT-4 predictions:")
        print(f"  Predicted Manipulative: {manipulative_count}")
        print(f"  Predicted Non-manipulative: {len(successful_results) - manipulative_count}")

        return {
            'total': len(results),
            'successful': len(successful_results),
            'predicted_manipulative': manipulative_count,
            'predicted_non_manipulative': len(successful_results) - manipulative_count,
            'true_manipulative': true_manipulative,
            'true_non_manipulative': true_non_manipulative,
            'training_examples_used': {
                'manipulative': len(manip_examples),
                'non_manipulative': len(nonmanip_examples)
            }
        }

    except Exception as e:
        if results:
            with open(output_file.replace('.json', '_partial.json'), 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        raise e


def test_api_connection():
    """Test the API connection before running the full classification"""
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Hello, this is a test."}
            ],
            model="gpt-4.1-mini",
            max_tokens=10,
            temperature=0
        )
        print("API connection successful!")
        print(f"Response: {response.choices[0].message.content}")
        return True
    except Exception as e:
        print(f"API connection failed: {e}")
        return False


if __name__ == "__main__":
    # Test API connection first
    print("Testing API connection...")
    if not test_api_connection():
        print("Please fix API issues before proceeding.")
        exit(1)

    # Configuration
    excel_file = "/Users/pauladroghoff/Documents/UCL/COMP0177-Project/MasterThesis/Code/data/mentalmanip_detailed_sentencelevel.xlsx"
    sentence_column = "Sentence"
    label_column = "Manipulative"

    # Run improved classification with few-shot examples
    # Reduced test size to control costs while testing the approach
    run_improved_sentence_classification(
        excel_file=excel_file,
        sentence_column=sentence_column,
        label_column=label_column,
        output_file='manipulation_results_chatGPT_zeroshot.json',
        n_test_per_class=100,  # 100 per class = 200 total (adjust as needed)
        k_manip=3,  # Number of manipulative training examples
        k_nonmanip=3,  # Number of non-manipulative training examples
        train_ratio=0.7,  # 70% for training examples, 30% for testing
        batch_size=25
    )