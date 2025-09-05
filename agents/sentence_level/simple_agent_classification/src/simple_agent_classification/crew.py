from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import pandas as pd
from typing import Dict, List, Optional
import json
import re
import time
import os
import datetime

os.environ["OPENAI_MODEL_NAME"] = "gpt-4.1-mini"

# Define manipulation techniques from MentalManip paper
MANIPULATION_TECHNIQUES = {
    'DEN': 'Denial',
    'EVA': 'Evasion',
    'FEI': 'Feigning Innocence',
    'RAT': 'Rationalization',
    'VIC': 'Playing the Victim Role',
    'SER': 'Playing the Servant Role',
    'S_B': 'Shaming or Belittlement',
    'INT': 'Intimidation',
    'B_A': 'Brandishing Anger',
    'ACC': 'Accusation',
    'P_S': 'Persuasion or Seduction'
}

# Define vulnerabilities from MentalManip paper
VULNERABILITIES = {
    'NAT': 'Naivety',
    'DEP': 'Dependency',
    'O_R': 'Over-responsibility',
    'O_I': 'Over-intellectualization',
    'L_S': 'Low self-esteem'
}

@CrewBase
class SentenceManipulationDetectorCrew:
    """Enhanced Sentence Manipulation Detector Crew with Discussion Task"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self):
        self.dataset = None
        self.results = []
        self.max_iterations = 3  # Allow discussion
        self.max_discussion_rounds = 2
        self.timeout_seconds = 60
        self.processing_times = []
        print("✓ CrewAI system initialized (GPT-3.5-turbo, discussion enabled)")

    def estimate_batch_cost(self, num_sentences: int, model='gpt-3.5-turbo') -> float:
        """Estimate cost before processing batch"""
        costs = {
            'gpt-3.5-turbo': 0.005,
            'gpt-4o-mini': 0.015,
            'gpt-4.1-mini': 0.050
        }
        estimated_cost = num_sentences * costs.get(model, 0.005) * 3
        print(f"COST ESTIMATE for {num_sentences} sentences:")
        print(f"   Model: {model}")
        print(f"   Estimated cost: ${estimated_cost:.3f}")
        print(f"   Max iterations per sentence: {self.max_iterations}")
        print(f"   Discussion rounds limit: {self.max_discussion_rounds}")
        return estimated_cost

    def load_dataset(self, file_path: str, text_column: str, label_column: str):
        """Load the sentence dataset from Excel file"""
        self.dataset = pd.read_excel(file_path)
        self.text_column = text_column
        self.label_column = label_column
        print(f"Loaded {len(self.dataset)} sentences from {file_path}")
        return self.dataset

    @agent
    def sentence_classifier(self) -> Agent:
        return Agent(
            config=self.agents_config['sentence_classifier'],
            verbose=True
        )

    @agent
    def quality_checker(self) -> Agent:
        return Agent(
            config=self.agents_config['quality_checker'],
            verbose=True
        )

    @task
    def classify_sentence_task(self) -> Task:
        return Task(
            config=self.tasks_config['classify_sentence_task']
        )

    @task
    def quality_check_task(self) -> Task:
        return Task(
            config=self.tasks_config['quality_check_task']
        )

    @task
    def discussion_task(self) -> Task:
        return Task(
            config=self.tasks_config['discussion_task']
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Sentence Manipulation Detector crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )

    def needs_discussion(self, classification_data: Dict, quality_data: Dict) -> bool:
        """Balanced discussion triggers"""
        initial_class = classification_data.get('classification', 0)
        suggested_class = quality_data.get('suggested_classification', initial_class)
        initial_conf = classification_data.get('confidence', 0.5)

        if initial_class != suggested_class:
            print(f"  → Discussion needed: Disagreement ({initial_class} vs {suggested_class})")
            return True
        if not quality_data.get('is_accurate', True):
            print(f"  → Discussion needed: Flagged as inaccurate")
            return True
        if initial_conf < 0.7:  # Relaxed threshold
            print(f"  → Discussion needed: Low confidence ({initial_conf:.2f})")
            return True
        print(f"  → Skipping discussion: Agreement and high confidence ({initial_conf:.2f})")
        return False

    def classify_with_feedback(self, sentence: str, max_iterations: int = None) -> Dict:
        if max_iterations is None:
            max_iterations = self.max_iterations

        start_time = time.time()
        classification_data = {}
        quality_data = {}
        final_data = {}
        iterations_used = 0

        print(f"\n{'=' * 60}")
        print(f"Processing: {sentence[:50]}...")
        print(f"Max iterations: {max_iterations}")
        print(f"{'=' * 60}")

        try:
            # STEP 1: Initial Classification
            print("  → Step 1: Initial classification...")
            classification_crew = Crew(
                agents=[self.sentence_classifier()],
                tasks=[self.classify_sentence_task()],
                process=Process.sequential,
                verbose=True
            )
            classification_result = classification_crew.kickoff(inputs={'sentence': sentence})
            iterations_used += 1
            classification_data = self._parse_classification(str(classification_result))
            print(f"    ✓ Classification: {classification_data['classification']} (confidence: {classification_data['confidence']:.2f})")

            # STEP 2: Quality Check
            print("  → Step 2: Quality check...")
            quality_inputs = {
                'sentence': sentence,
                'classification_result': str(classification_data)
            }
            quality_crew = Crew(
                agents=[self.quality_checker()],
                tasks=[self.quality_check_task()],
                process=Process.sequential,
                verbose=True
            )
            quality_result = quality_crew.kickoff(inputs=quality_inputs)
            quality_data = self._parse_quality_check(str(quality_result))
            iterations_used += 1
            print(f"    ✓ Quality check: {'Accurate' if quality_data.get('is_accurate', True) else 'Needs revision'}")

            # STEP 3: Discussion if needed
            if self.needs_discussion(classification_data, quality_data) and iterations_used < max_iterations:
                print("  → Step 3: Discussion...")
                discussion_inputs = {
                    'sentence': sentence,
                    'classification_result': str(classification_data),
                    'quality_feedback': str(quality_data)
                }
                discussion_crew = Crew(
                    agents=[self.sentence_classifier(), self.quality_checker()],
                    tasks=[self.discussion_task()],
                    process=Process.sequential,
                    verbose=True
                )
                discussion_result = discussion_crew.kickoff(inputs=discussion_inputs)
                final_data = self._parse_discussion(str(discussion_result))
                iterations_used += 1
                print(f"    ✓ Discussion: Final classification {final_data.get('final_classification', 0)}")
            else:
                final_data = self._merge_results(classification_data, quality_data)
                print(f"    ✓ Merged results: Final classification {final_data.get('final_classification', 0)}")

        except Exception as e:
            print(f"    ✗ Error during processing: {e}")
            final_data = {
                'final_classification': 0,
                'confidence': 0.2,
                'techniques_found': [],
                'reasoning': f'Error during classification: {str(e)}'
            }
            iterations_used = max(iterations_used, 1)

        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)

        result = {
            'sentence': sentence,
            'final_classification': final_data.get('final_classification', 0),
            'confidence': final_data.get('confidence', 0.5),
            'techniques_found': final_data.get('techniques_found', []),
            'reasoning': final_data.get('reasoning', ''),
            'iterations': iterations_used,
            'processing_time': processing_time
        }

        print(f"  ✓ Final: Classification {result['final_classification']}, Confidence {result['confidence']:.2f}, Iterations {iterations_used}")
        return result

    def _parse_classification(self, result: str, task: str = 'classify_sentence_task') -> Dict:
        """Parse classification output with robust handling for different tasks"""
        # Initialize default data based on task type
        if task in ['classify_sentence_task', 'discussion_task']:
            data = {
                'classification': 0,
                'confidence': 0.5,
                'reasoning': 'No reasoning provided'
            }
        elif task == 'quality_check_task':
            data = {
                'is_accurate': 'No',
                'review_confidence': 0.5,
                'suggested_classification': 0,
                'feedback': 'No feedback provided'
            }
        else:
            raise ValueError(f"Unknown task: {task}")

        if not result:
            print(f"⚠️ Parsing warning: Empty {task} result")
            with open(f"debug_{task}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'a') as f:
                f.write(f"Empty {task} result\n\n")
            return data

        result = str(result).strip()
        result_lower = result.lower()

        # Log raw output for debugging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
            f.write(f"Raw {task} Output: {result}\n\n")

        # Handle classification tasks (classify_sentence_task, discussion_task)
        if task in ['classify_sentence_task', 'discussion_task']:
            # Extract classification
            class_key = 'Final Classification' if task == 'discussion_task' else 'Classification'
            class_match = re.search(rf'{class_key}:\s*([0-1])(?:\s|$|\n)', result, re.IGNORECASE)
            if not class_match:
                class_match = re.search(r'(?:^|\n)\s*([0-1])\s*(?:\.|:|-|\n|$)', result, re.MULTILINE)
            if class_match:
                data['classification'] = int(class_match.group(1))
            else:
                # Fallback based on keywords
                if any(phrase in result_lower for phrase in
                       ['is manipulative', 'manipulative intent', 'control or exploit']):
                    data['classification'] = 1
                    data['confidence'] = 0.75
                elif any(phrase in result_lower for phrase in
                         ['not manipulative', 'non-manipulative', 'direct communication', 'honest emotion',
                          'boundary-setting']):
                    data['classification'] = 0
                    data['confidence'] = 0.75
                else:
                    print(f"Parsing warning: No {class_key.lower()} found in {task}")
                    with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                        f.write(f"No {class_key.lower()} found: {result[:100]}...\n\n")

            # Extract confidence
            conf_match = re.search(r'Confidence:\s*([\d.]+)\s*(?:%|\b)', result, re.IGNORECASE)
            if conf_match:
                try:
                    conf_value = float(conf_match.group(1))
                    data['confidence'] = min(conf_value / 100 if '%' in conf_match.group(0) else conf_value, 1.0)
                except ValueError:
                    print(f"Parsing warning: Invalid confidence format in {task}: {conf_match.group(1)}")
                    with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                        f.write(f"Invalid confidence format: {conf_match.group(1)}\n\n")
            else:
                strong_indicators = ['clearly', 'obviously', 'definitely']
                weak_indicators = ['possibly', 'maybe', 'uncertain']
                if any(ind in result_lower for ind in strong_indicators):
                    data['confidence'] = max(data['confidence'], 0.85)
                elif any(ind in result_lower for ind in weak_indicators):
                    data['confidence'] = min(data['confidence'], 0.6)
                else:
                    print(f"Parsing warning: No confidence found in {task}, using {data['confidence']}")
                    with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                        f.write(f"No confidence found, using {data['confidence']}\n\n")

            # Extract reasoning
            reason_match = re.search(r'Reasoning:\s*(.+?)(?=\n\n|\Z|$)', result, re.IGNORECASE | re.DOTALL)
            if reason_match:
                data['reasoning'] = reason_match.group(1).strip()
            else:
                print(f"Parsing warning: No reasoning found in {task}")
                with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                    f.write(f"No reasoning found\n\n")

        # Handle quality_check_task
        elif task == 'quality_check_task':
            # Extract is_accurate
            accurate_match = re.search(r'Is Accurate:\s*(Yes|No)(?:\s|$|\n)', result, re.IGNORECASE)
            if accurate_match:
                data['is_accurate'] = accurate_match.group(1).capitalize()
            else:
                print(f"Parsing warning: No is_accurate found in {task}")
                with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                    f.write(f"No is_accurate found: {result[:100]}...\n\n")

            # Extract review_confidence
            conf_match = re.search(r'Review Confidence:\s*([\d.]+)\s*(?:%|\b)', result, re.IGNORECASE)
            if conf_match:
                try:
                    conf_value = float(conf_match.group(1))
                    data['review_confidence'] = min(conf_value / 100 if '%' in conf_match.group(0) else conf_value, 1.0)
                except ValueError:
                    print(f"Parsing warning: Invalid review_confidence format in {task}: {conf_match.group(1)}")
                    with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                        f.write(f"Invalid review_confidence format: {conf_match.group(1)}\n\n")
            else:
                print(f"Parsing warning: No review_confidence found in {task}, using {data['review_confidence']}")
                with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                    f.write(f"No review_confidence found, using {data['review_confidence']}\n\n")

            # Extract suggested_classification
            sugg_match = re.search(r'Suggested Classification:\s*([0-1])(?:\s|$|\n)', result, re.IGNORECASE)
            if sugg_match:
                data['suggested_classification'] = int(sugg_match.group(1))
            else:
                print(f"Parsing warning: No suggested_classification found in {task}")
                with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                    f.write(f"No suggested_classification found: {result[:100]}...\n\n")

            # Extract feedback
            feedback_match = re.search(r'Feedback:\s*(.+?)(?=\n\n|\Z|$)', result, re.IGNORECASE | re.DOTALL)
            if feedback_match:
                data['feedback'] = feedback_match.group(1).strip()
            else:
                print(f"Parsing warning: No feedback found in {task}")
                with open(f"debug_{task}_{timestamp}.txt", 'a') as f:
                    f.write(f"No feedback found\n\n")

        return data

    def _parse_quality_check(self, result: str) -> Dict:
        """Parse quality check with robust handling"""
        data = {
            'is_accurate': True,
            'review_confidence': 0.7,
            'suggested_classification': None,
            'feedback': 'No feedback provided'
        }

        if not result:
            print(f"Parsing warning: Empty quality check result")
            with open(f"debug_quality_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'a') as f:
                f.write(f"Empty quality check result\n\n")
            return data

        result = str(result).strip()
        result_lower = result.lower()

        # Log raw output for debugging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"debug_quality_{timestamp}.txt", 'a') as f:
            f.write(f"Raw Quality Check Output: {result}\n\n")

        # Check accuracy
        accuracy_match = re.search(r'Is Accurate:\s*(Yes|No)\s*(?:$|\n)', result, re.IGNORECASE)
        if accuracy_match:
            data['is_accurate'] = accuracy_match.group(1).lower() == 'yes'
        else:
            if any(phrase in result_lower for phrase in ['not accurate', 'inaccurate', 'disagree']):
                data['is_accurate'] = False
            print(f"Parsing warning: No accuracy field found")
            with open(f"debug_quality_{timestamp}.txt", 'a') as f:
                f.write(f"No accuracy field found\n\n")

        # Extract confidence
        conf_match = re.search(r'Review Confidence:\s*([\d.]+)\s*(?:%|\b)', result, re.IGNORECASE)
        if conf_match:
            try:
                conf_value = float(conf_match.group(1))
                data['review_confidence'] = min(conf_value / 100 if '%' in conf_match.group(0) else conf_value, 1.0)
            except ValueError:
                print(f"Parsing warning: Invalid review confidence format: {conf_match.group(1)}")
                with open(f"debug_quality_{timestamp}.txt", 'a') as f:
                    f.write(f"Invalid review confidence format: {conf_match.group(1)}\n\n")
        else:
            strong_indicators = ['confident', 'certain', 'sure']
            if any(ind in result_lower for ind in strong_indicators):
                data['review_confidence'] = 0.85
            print(f"Parsing warning: No review confidence found, using {data['review_confidence']}")
            with open(f"debug_quality_{timestamp}.txt", 'a') as f:
                f.write(f"No review confidence found, using {data['review_confidence']}\n\n")

        # Extract suggested classification
        suggest_match = re.search(r'Suggested Classification:\s*([0-1])(?:\s|$|\n)', result, re.IGNORECASE)
        if suggest_match:
            data['suggested_classification'] = int(suggest_match.group(1))
        else:
            if any(phrase in result_lower for phrase in ['should be manipulative', 'is manipulative']):
                data['suggested_classification'] = 1
            elif any(phrase in result_lower for phrase in ['should be non-manipulative', 'not manipulative']):
                data['suggested_classification'] = 0
            else:
                print(f"Parsing warning: No suggested classification found")
                with open(f"debug_quality_{timestamp}.txt", 'a') as f:
                    f.write(f"No suggested classification found\n\n")

        # Extract feedback
        feedback_match = re.search(r'Feedback:\s*(.+?)(?=\n\n|\Z|$)', result, re.IGNORECASE | re.DOTALL)
        if feedback_match:
            data['feedback'] = feedback_match.group(1).strip()
        else:
            print(f"Parsing warning: No feedback found")
            with open(f"debug_quality_{timestamp}.txt", 'a') as f:
                f.write(f"No feedback found\n\n")

        return data

    def _parse_discussion(self, result: str) -> Dict:
        """Parse discussion task output with robust handling"""
        data = {
            'final_classification': 0,
            'confidence': 0.5,
            'techniques_found': [],
            'reasoning': 'No discussion reasoning provided'
        }

        if not result:
            print(f"Parsing warning: Empty discussion result")
            with open(f"debug_discussion_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 'a') as f:
                f.write(f"Empty discussion result\n\n")
            return data

        result = str(result).strip()
        result_lower = result.lower()

        # Log raw output for debugging
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"debug_discussion_{timestamp}.txt", 'a') as f:
            f.write(f"Raw Discussion Output: {result}\n\n")

        # Extract final classification
        class_match = re.search(r'Final Classification:\s*([0-1])(?:\s|$|\n)', result, re.IGNORECASE)
        if class_match:
            data['final_classification'] = int(class_match.group(1))
        else:
            if any(phrase in result_lower for phrase in ['final manipulative', 'agreed manipulative']):
                data['final_classification'] = 1
                data['confidence'] = 0.75
            elif any(phrase in result_lower for phrase in ['final non-manipulative', 'agreed non-manipulative']):
                data['final_classification'] = 0
                data['confidence'] = 0.75
            else:
                print(f"Parsing warning: No final classification found in discussion")
                with open(f"debug_discussion_{timestamp}.txt", 'a') as f:
                    f.write(f"No final classification found\n\n")

        # Extract confidence
        conf_match = re.search(r'Confidence:\s*([\d.]+)\s*(?:%|\b)', result, re.IGNORECASE)
        if conf_match:
            try:
                conf_value = float(conf_match.group(1))
                data['confidence'] = min(conf_value / 100 if '%' in conf_match.group(0) else conf_value, 1.0)
            except ValueError:
                print(f"Parsing warning: Invalid confidence in discussion: {conf_match.group(1)}")
                with open(f"debug_discussion_{timestamp}.txt", 'a') as f:
                    f.write(f"Invalid confidence format: {conf_match.group(1)}\n\n")
        else:
            strong_indicators = ['confident', 'certain', 'agreed with high confidence']
            if any(ind in result_lower for ind in strong_indicators):
                data['confidence'] = 0.85
            print(f"Parsing warning: No confidence found in discussion, using {data['confidence']}")
            with open(f"debug_discussion_{timestamp}.txt", 'a') as f:
                f.write(f"No confidence found, using {data['confidence']}\n\n")

        # Extract reasoning
        reason_match = re.search(r'Reasoning:\s*(.+?)(?=\n\n|\Z|$)', result, re.IGNORECASE | re.DOTALL)
        if reason_match:
            data['reasoning'] = reason_match.group(1).strip()
        else:
            print(f"Parsing warning: No reasoning found in discussion")
            with open(f"debug_discussion_{timestamp}.txt", 'a') as f:
                f.write(f"No reasoning found\n\n")

        # Extract techniques
        tech_match = re.search(r'Techniques:\s*(.+?)(?=\n\n|\Z|$)', result, re.IGNORECASE | re.DOTALL)
        if tech_match:
            techniques = tech_match.group(1).strip()
            if techniques.lower() != 'none':
                for key, value in MANIPULATION_TECHNIQUES.items():
                    if value.lower() in techniques.lower():
                        data['techniques_found'].append(value)
                if not data['techniques_found'] and techniques:
                    data['techniques_found'] = [t.strip() for t in techniques.split(',')]
        else:
            print(f"Parsing warning: No techniques found in discussion")
            with open(f"debug_discussion_{timestamp}.txt", 'a') as f:
                f.write(f"No techniques found\n\n")

        return data

    def _merge_results(self, classification_data: Dict, quality_data: Dict) -> Dict:
        """Merge classification and quality check results with balanced confidence handling"""
        initial_class = classification_data.get('classification', 0)
        initial_conf = classification_data.get('confidence', 0.5)
        suggested_class = quality_data.get('suggested_classification', initial_class)
        quality_conf = quality_data.get('review_confidence', 0.5)

        MIN_CONFIDENCE_THRESHOLD = 0.7
        OVERRIDE_CONFIDENCE_THRESHOLD = 0.85

        print(f"    → Merging: Initial={initial_class}(conf:{initial_conf:.2f}), Suggested={suggested_class}(conf:{quality_conf:.2f})")

        if suggested_class != initial_class:
            if quality_conf >= OVERRIDE_CONFIDENCE_THRESHOLD:
                final_class = suggested_class
                final_conf = quality_conf * 0.95
                print(f"    → Quality override: {initial_class} → {final_class} (quality conf: {quality_conf:.2f})")
            else:
                final_class = initial_class
                final_conf = initial_conf * 0.95
                print(f"    → Kept original due to low quality confidence: {quality_conf:.2f}")
        else:
            final_class = initial_class
            final_conf = max(initial_conf, quality_conf)  # Take higher confidence when agreeing
            if initial_conf < MIN_CONFIDENCE_THRESHOLD or quality_conf < MIN_CONFIDENCE_THRESHOLD:
                final_conf *= 0.95
            print(f"    → Agents agree - final confidence: {final_conf:.2f}")

        return {
            'final_classification': final_class,
            'confidence': min(max(final_conf, 0.1), 1.0),
            'techniques_found': classification_data.get('techniques', []),
            'reasoning': f"Class: {classification_data.get('reasoning', '')} | Quality: {quality_data.get('feedback', '')}"[:200]
        }

    def debug_single_classification(self, sentence: str):
        """Debug what the agent is actually outputting"""
        print(f"DEBUGGING SINGLE CLASSIFICATION")
        print(f"Sentence: {sentence}")
        print(f"=" * 80)

        classification_crew = Crew(
            agents=[self.sentence_classifier()],
            tasks=[self.classify_sentence_task()],
            process=Process.sequential,
            verbose=True
        )
        result = classification_crew.kickoff(inputs={'sentence': sentence})

        print(f"\n" + "=" * 80)
        print(f"RAW AGENT OUTPUT:")
        print(f"=" * 80)
        print(f"Type: {type(result)}")
        print(f"Result: {result}")
        if hasattr(result, 'raw'):
            print(f"\nRaw attribute: {result.raw}")

        result_str = str(result)
        print(f"\nString conversion: {result_str}")

        # Save raw output to file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"debug_output_{timestamp}.txt", 'w') as f:
            f.write(f"Sentence: {sentence}\n")
            f.write(f"Raw Output: {result_str}\n")

        parsed = self._parse_classification(result_str)
        print(f"\n" + "=" * 80)
        print(f"PARSED RESULT: {parsed}")
        print(f"=" * 80)

        # Test regex patterns
        class_match = re.search(r'Classification:\s*([0-1])(?:\s|$|\n)', result_str, re.IGNORECASE)
        conf_match = re.search(r'Confidence:\s*([\d.]+)\s*(?:%|\b)', result_str, re.IGNORECASE)
        print(f"Classification regex match: {class_match.group(1) if class_match else 'None'}")
        print(f"Confidence regex match: {conf_match.group(1) if conf_match else 'None'}")

        return result_str

    def debug_discussion_flow(self, sentence: str) -> Dict:
        """Debug version that shows exactly what's happening in each step"""
        print(f"DEBUGGING DISCUSSION FLOW")
        print(f"Sentence: {sentence}")
        print(f"=" * 60)

        # Step 1: Classification
        print("STEP 1: Initial Classification")
        classification_crew = Crew(
            agents=[self.sentence_classifier()],
            tasks=[self.classify_sentence_task()],
            process=Process.sequential,
            verbose=True
        )
        classification_result = classification_crew.kickoff(inputs={'sentence': sentence})
        classification_data = self._parse_classification(str(classification_result))
        print(f"✓ Classification Result: {classification_data}")

        # Step 2: Quality Check
        print(f"\nSTEP 2: Quality Check")
        quality_inputs = {
            'sentence': sentence,
            'classification_result': str(classification_data)
        }
        quality_crew = Crew(
            agents=[self.quality_checker()],
            tasks=[self.quality_check_task()],
            process=Process.sequential,
            verbose=True
        )
        quality_result = quality_crew.kickoff(inputs=quality_inputs)
        quality_data = self._parse_quality_check(str(quality_result))
        print(f"✓ Quality Check Result: {quality_data}")

        # Step 3: Check if discussion is needed
        print(f"\nSTEP 3: Discussion Decision")
        discussion_needed = self.needs_discussion(classification_data, quality_data)
        print(f"Discussion needed: {discussion_needed}")

        if discussion_needed:
            print(f"RUNNING DISCUSSION TASK")
            discussion_inputs = {
                'sentence': sentence,
                'classification_result': str(classification_data),
                'quality_feedback': str(quality_data)
            }
            discussion_crew = Crew(
                agents=[self.sentence_classifier(), self.quality_checker()],
                tasks=[self.discussion_task()],
                process=Process.sequential,
                verbose=True
            )
            discussion_result = discussion_crew.kickoff(inputs=discussion_inputs)
            final_data = self._parse_discussion(str(discussion_result))
            print(f"✓ Discussion Result: {final_data}")
            iterations = 3
        else:
            print(f"SKIPPING DISCUSSION - Merging results")
            final_data = self._merge_results(classification_data, quality_data)
            print(f"✓ Merged Result: {final_data}")
            iterations = 2

        return {
            'sentence': sentence,
            'final_classification': final_data.get('final_classification', 0),
            'confidence': final_data.get('confidence', 0.5),
            'techniques_found': final_data.get('techniques_found', []),
            'reasoning': final_data.get('reasoning', ''),
            'iterations': iterations,
            'classification_data': classification_data,
            'quality_data': quality_data,
            'discussion_triggered': discussion_needed
        }

    def process_batch(self, sentences: List[str], max_iterations: int = None) -> List[Dict]:
        """Enhanced batch processing with cost controls"""
        if max_iterations is None:
            max_iterations = self.max_iterations

        estimated_cost = self.estimate_batch_cost(len(sentences))
        if estimated_cost > 2.00:
            response = input(f"\nEstimated cost: ${estimated_cost:.3f}. Continue? (y/n): ")
            if response.lower() != 'y':
                print("Processing cancelled.")
                return []

        results = []
        classification_counts = {'manipulative': 0, 'non-manipulative': 0}
        total_start_time = time.time()

        print(f"Starting enhanced batch processing:")
        print(f"   - {len(sentences)} sentences")
        print(f"   - Max {max_iterations} iterations per sentence")
        print(f"   - Discussion task enabled")

        for i, sentence in enumerate(sentences):
            print(f"\n{'#' * 70}")
            print(f"Processing sentence {i + 1}/{len(sentences)}")
            print(f"{'#' * 70}")

            try:
                result = self.classify_with_feedback(sentence, max_iterations)
                results.append(result)
                if result['final_classification'] == 1:
                    classification_counts['manipulative'] += 1
                else:
                    classification_counts['non-manipulative'] += 1

                if self.processing_times:
                    avg_time = sum(self.processing_times) / len(self.processing_times)
                    remaining = len(sentences) - (i + 1)
                    est_remaining_time = remaining * avg_time
                    print(f"Progress: {i + 1}/{len(sentences)} | Remaining time: ~{est_remaining_time:.1f}s")

            except Exception as e:
                print(f"✗ Error processing sentence {i + 1}: {e}")
                results.append({
                    'sentence': sentence,
                    'final_classification': 0,
                    'confidence': 0.3,
                    'techniques_found': [],
                    'reasoning': f'Processing error: {str(e)}',
                    'iterations': 0,
                    'processing_time': 0
                })

        total_time = time.time() - total_start_time
        total_iterations = sum(r.get('iterations', 0) for r in results)

        print(f"\n{'=' * 70}")
        print(f"CLASSIFICATION DISTRIBUTION")
        print(f"{'=' * 70}")
        print(f"Total Manipulative: {classification_counts['manipulative']} ({classification_counts['manipulative'] / len(sentences) * 100:.1f}%)")
        print(f"Total Non-Manipulative: {classification_counts['non-manipulative']} ({classification_counts['non-manipulative'] / len(sentences) * 100:.1f}%)")
        print(f"\n🔧 PROCESSING STATS:")
        print(f"   Total time: {total_time:.1f}s")
        print(f"   Average per sentence: {total_time / len(sentences):.1f}s")
        print(f"   Total iterations: {total_iterations}")
        print(f"   Average iterations per sentence: {total_iterations / len(sentences):.1f}")

        return results

    def evaluate_performance(self, results: List[Dict], true_labels: List[int]) -> Dict:
        """Enhanced evaluation with additional metrics"""
        predictions = [r['final_classification'] for r in results]
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(true_labels) if true_labels else 0

        tp = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 1)
        fp = sum(1 for pred, true in zip(predictions, true_labels) if pred == 1 and true == 0)
        fn = sum(1 for pred, true in zip(predictions, true_labels) if pred == 0 and true == 1)
        tn = sum(1 for pred, true in zip(predictions, true_labels) if pred == 0 and true == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        technique_stats = {}
        for result in results:
            for tech in result.get('techniques_found', []):
                technique_stats[tech] = technique_stats.get(tech, 0) + 1

        avg_processing_time = sum(r.get('processing_time', 0) for r in results) / len(results) if results else 0
        total_iterations = sum(r.get('iterations', 0) for r in results)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'true_positive': tp,
                'false_positive': fp,
                'false_negative': fn,
                'true_negative': tn
            },
            'total_samples': len(true_labels),
            'average_confidence': sum(r['confidence'] for r in results) / len(results) if results else 0,
            'technique_distribution': technique_stats,
            'average_processing_time': avg_processing_time,
            'total_iterations_used': total_iterations,
            'cost_efficiency': total_iterations / len(results) if results else 0
        }

if __name__ == "__main__":
    crew_instance = SentenceManipulationDetectorCrew()
    print("DEBUGGING: Let's see what the agent actually outputs...")
    test_sentence = "I disagree with your proposal."
    raw_output = crew_instance.debug_single_classification(test_sentence)
    print(f"\n" + "=" * 80)
    print(f"DIAGNOSIS:")
    print(f"=" * 80)
    print(f"If you see uniform 0.59 confidence, the problem is:")
    print(f"1. Agent not following the prompt format")
    print(f"2. Parsing regex not matching the actual output")
    print(f"3. Default values being used")
    print(f"\nExpected output format:")
    print(f"Classification: 0")
    print(f"Confidence: 0.8")
    print(f"Reasoning: [...]")
    print(f"Techniques: [...]")