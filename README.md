# Role-Aware Classroom Discourse Inequality Detection

Official PyTorch implementation for **Modeling Linguistic Power and Discourse Inequality in Classroom Interactions Using Deep Neural Networks**.

The model separately encodes teacher and student utterances, builds role-aware interaction features, adaptively fuses multiple interaction components, and reports component-level power attribution scores.

## Repository structure

```text
classroom-discourse-inequality/
├── configs/default.yaml
├── data/example.csv
├── scripts/run_ablation.sh
├── src/
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── baselines.py
├── LICENSE
└── requirements.txt
```

## Data format

Prepare a UTF-8 CSV file with the following columns:

| Column | Meaning |
|---|---|
| `teacher_text` | Teacher utterance |
| `student_text` | Temporally aligned student utterance |
| `label` | `1` for teacher-dominated or unequal discourse; `0` for relatively balanced interaction |

`data/example.csv` is only a schema illustration and must not be used to reproduce the paper's numerical results. Replace it with the public dataset used in the study and cite the dataset according to its license.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.10+, PyTorch 2.x, and Transformers 4.x. A CUDA-capable GPU is recommended.

## Training

```bash
python -m src.train \
  --data data/your_dataset.csv \
  --output_dir outputs/full_model \
  --model_name bert-base-uncased \
  --max_length 128 \
  --batch_size 8 \
  --epochs 10 \
  --learning_rate 2e-5 \
  --weight_decay 0.01 \
  --warmup_ratio 0.1 \
  --seed 42
```

The script performs a stratified 72%/8%/20% train/validation/test split, selects the checkpoint with the best validation F1-score, and evaluates the held-out test set once.

## Evaluation and attribution

```bash
python -m src.evaluate \
  --data data/your_dataset.csv \
  --checkpoint outputs/full_model/best_model.pt \
  --output_dir outputs/evaluation
```

Outputs include `metrics.json`, `predictions.csv`, `confusion_matrix.csv`, and `attributions.csv`. The attribution file contains adaptive fusion weights and gradient-times-input scores for teacher, student, absolute-difference, and element-wise-product components.

## Ablation study

```bash
bash scripts/run_ablation.sh data/your_dataset.csv
```

Supported variants are `full`, `no_role`, `no_interaction`, `no_adaptive`, and `no_role_interaction`.

## Baselines

Traditional TF-IDF baselines:

```bash
python -m src.baselines --data data/your_dataset.csv --model logistic_regression
python -m src.baselines --data data/your_dataset.csv --model svm
```

Transformer baselines can be obtained with the ablation interface: `--variant no_role_interaction` provides a shared-representation classifier. For BERT or RoBERTa, set `--model_name bert-base-uncased` or `roberta-base`.

## Reproducibility notes

- All random seeds are fixed where supported.
- Splits are stratified and saved to the output directory.
- Text normalization removes control characters and normalizes whitespace; it does not lowercase text independently of the selected tokenizer.
- The test set is not used for hyperparameter selection or early stopping.
- Reported paper values require the exact original dataset and annotation release. Because the manuscript does not identify a dataset URL, this repository does not claim that the included illustrative rows reproduce those values.

## Citation

```bibtex
@article{fu2026linguistic,
  title={Modeling Linguistic Power and Discourse Inequality in Classroom Interactions Using Deep Neural Networks},
  author={Fu, Sijing and Lu, Zongyao and Liu, Yang and Liu, Qi},
  year={2026}
}
```

## License

Code is released under the MIT License. Dataset access and use remain subject to the original dataset license.

