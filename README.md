# MIND
MIND: Multi-Granular INterpretable Detection of Mental Manipulation

This repository contains the implementation for **"MIND: Multi-Granular INterpretable Detection of Mental Manipulation"** on detecting manipulative language at both sentence-level and dialogue-level, combining and comparing baseline machine learning models, transformer-based models (DistilRoBERTa), ChatGPT prompting, multi-agent frameworks (CrewAI), and Explainable AI (XAI) methods.

---

## Repository Structure

### `data/`
- Contains the **MentalManip dataset** in Excel format at both sentence and dialogue level.  
- Used as the input for all training, evaluation, and explainability experiments.

---

### `agents/`
- Implements **multi-agent reasoning frameworks** with CrewAI.  
- Separated into **sentence-level** and **dialogue-level** agents.  
- Each folder contains:
  - `main.py` – entry point to run the framework.  
  - `crew.py` – defines agent workflow.  
  - `agents.yaml` – agent configurations.  
  - `tasks.yaml` – task definitions.  
- **Note:** Requires an OpenAI API key in the `.env` files:  
  - `/agents/sentence_level/simple_agent_classification/.env`  
  - `/agents/dialogue_level/simple_agent_classification/.env`

---

### `LLM_training/`
- Organised by **sentence-level** and **dialogue-level**.  
- Contains:
  - **Baseline models**: Logistic Regression, Random Forest.  
  - **ChatGPT API training** scripts for classification (zero-shot and few-shot):  
    - `sentence_chatGPTrun_zeroshot.py`  
    - `sentence_chatGPTrun_fewshot.py`  
    - `dialogue_chatGPTrun_zeroshot.py`  
    - `dialogue_chatGPTrun_fewshot.py`  
    - *(Insert API key before running)*  
  - **Transformer fine-tuning**: `sentence_model_exp.py` & `dialogue_model_exp.py` fine-tune **MentalManip DistilRoBERTa**.  
  - **Radar plots**: Compare performance (precision, recall, F1) across models.

---

### `sentence_XAI/` & `dialogue_XAI/`
- Implement **explainability methods (XAI)** at sentence and dialogue levels.  
- Each folder includes:
  - **Individual method implementations**: SHAP, LIME, Raw Attention, Integrated Gradients, Expected Gradients, Token Occlusion.  
  - **Comparison code**: calculates similarity/dissimilarity metrics (cosine similarity, Pearson correlation, Jensen–Shannon divergence) to evaluate agreement between methods.  
- **Important:** Run each XAI method individually **before** running the comparison code.

---

### `requirements.txt`
- Contains all necessary Python dependencies (transformers, scikit-learn, CrewAI, shap, lime, etc.).  
- Install with:  
  ```bash
  pip install -r requirements.txt
