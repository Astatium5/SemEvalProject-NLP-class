# DNS at SemEval 2026 Task 6: Unmasking Political Question Evasions

This repository contains the code and research for our final project in the Graduate NLP course at the University of Colorado Boulder. We participated in the [SemEval 2026 Task 6: CLARITY](https://semeval.github.io/SemEval2026/tasks.html), which focuses on unmasking political question evasions.

## Abstract

Political communication inherently exhibits lower clarity and a stronger tendency toward evasive responding compared to common discourse. In this work, we extend our focus from simple clarity judgment to the classification of fine-grained evasion types in political interview question–answer datasets. Based on insights from related studies, we explored a broad spectrum of approaches from traditional statistical machine learning methods to encoder-based architectures and current LLMs. Our final system uses an ensemble of DeBERTa-large and RoBERTa-large for Subtask 1, and a seed-averaged DeBERTa-v3-large model for Subtask 2.

## Task Description

The CLARITY shared task aims to address the strategic nature of political communication by introducing a computational approach to detecting and classifying response ambiguity.

### Subtask 1: Clarity Level
A 3-label classification task focusing on response clarity. The objective is to determine how clearly an interviewee answers a question based on three categories:
- **Clear Reply**
- **Ambivalent Reply**
- **Clear Non-Reply**

### Subtask 2: Evasion Level
A 9-label classification task focusing on specific response evasion strategies:
- **Explicit**, **Implicit**, **General**, **Partial/Half-answer**, **Dodging**, **Deflection**, **Declining to answer**, **Claims ignorance**, **Clarification**.

## Methodology

Our approach progressed from simple lexical baselines to sophisticated transformer architectures:
- **Baselines:** Logistic Regression with TF-IDF features.
- **Encoder Models:** RoBERTa-large, XLNet-large, and DeBERTa-v3-large.
- **Few-Shot Learning:** SetFit (paraphrase-mpnet-base-v2).
- **LLMs:** LLaMA-3-8B with LoRA and instruction tuning.

### Key Innovations
- **Clarity-Informed Input Formatting:** Concatenating question and answer with strategic context.
- **Weighted Loss Functions:** Addressing severe class imbalance in the QEvasion dataset.
- **Ensemble Techniques:** Averaging logits from RoBERTa and DeBERTa to improve robustness.

## Results

| Subtask | Model | Macro F1 |
|---------|-------|----------|
| **Subtask 1 (Clarity)** | DeBERTa-large + RoBERTa-large Ensemble | **0.650** |
| **Subtask 2 (Evasion)** | DeBERTa-v3-large | **0.627** |

## Technologies Used

- **Frameworks:** PyTorch, Hugging Face Transformers, Datasets, Accelerate.
- **Models:** DeBERTa-v3, RoBERTa, XLNet, LLaMA 3, SetFit.
- **Tools:** Scikit-learn, Pandas, NumPy, Matplotlib (for visualization).

## Repository Structure

- `notebooks/`: Main Jupyter notebooks for training and evaluation.
  - `Subtask1_Ensemble_RoBERTa_DeBERTa.ipynb`: Best model for Subtask 1.
  - `Subtask2_Ensemble_DeBERTa.ipynb`: Best model for Subtask 2.
  - `Data_Visualization.ipynb`: Exploratory data analysis.
  - `Experiment_Runner.ipynb`: Notebook for running various experimental configurations.
- `src/`: Python scripts for training and inference.
  - `train.py`: Main training script for encoder models.
  - `inference.py`: Inference script for generating predictions.
- `experiments/`: Archive of experimental runs, failed models, and hyperparameter tuning.
- `results/`: Saved predictions and performance visualizations.
- `docs/`: Project documentation and the final LaTeX report.

## How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/semeval-2026-task6.git
   cd semeval-2026-task6
   ```

2. **Set up the environment:**
   Using conda:
   ```bash
   conda env create -f environment.yml
   conda activate nlp-semeval
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Data:**
   The project uses the `ailsntua/QEvasion` dataset from Hugging Face, which is automatically downloaded by the scripts/notebooks.

4. **Training:**
   You can run the main notebooks in the `notebooks/` directory or use the scripts in `src/`.

## Our Team

This project was a collaborative effort by:
* **Nickolaus Jackoski**
* **Sungboo Park**
* **Dima Golubenko**

*Department of Computer Science, University of Colorado Boulder*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
