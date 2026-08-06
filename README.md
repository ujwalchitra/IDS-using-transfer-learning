# IoT Intrusion Detection Using Transfer Learning

A lightweight intrusion detection system (IDS) for identifying malicious network traffic in IoT environments. The project uses transfer learning to pretrain models on CIC-IDS 2017 traffic and fine-tune them on IoT-specific traffic.

The repository includes PyTorch implementations of three hybrid neural-network architectures, data preparation utilities, training and evaluation scripts, result visualizations, and a Streamlit demonstration dashboard.

## Highlights

- Binary classification of benign and malicious network flows
- Transfer-learning workflow: pretraining, fine-tuning, and evaluation
- Hybrid 1D-CNN + LSTM and bidirectional LSTM architectures
- Lightweight models intended for resource-constrained IoT deployments
- Automatic numeric feature cleaning, variance-based feature selection, scaling, and stratified splitting
- Model comparison, overfitting checks, and visualization utilities
- Interactive Streamlit demonstration with a console-compatible fallback

## Model architectures

The training pipeline compares these models:

1. `Hybrid1DCNN_LSTM` - convolutional feature extraction followed by an LSTM
2. `Hybrid1DCNN_BiLSTM` - convolutional feature extraction followed by a bidirectional LSTM
3. `BERT_CNN_LSTM` - an embedding-based hybrid CNN/LSTM model

The default configuration uses the top 50 features, a hidden dimension of 64, 32 CNN filters, dropout of 0.2, and two output classes.

## Project structure

```text
.
|-- config.py                       # Training paths and hyperparameters
|-- models/                         # PyTorch model definitions
|-- utils/data_loader.py            # Loading, cleaning, scaling, and splitting
|-- train_transfer_learning.py      # Main transfer-learning pipeline
|-- train_quick_test.py             # Short training smoke test
|-- step3_sequence_creation.py      # Time-series sequence generation
|-- step4_transfer_learning.py      # Additional transfer-learning workflow
|-- compare_three_hybrid_models.py  # Model comparison
|-- evaluate_pretrained.py          # Pretrained-model evaluation
|-- visualize_results.py            # Charts and result summaries
|-- working_demo.py                 # Streamlit IDS dashboard
|-- figures/                        # Example result visualizations
`-- dataset_report.txt              # Dataset class summary
```

Datasets, virtual environments, trained weights, and generated experiment outputs are intentionally excluded from Git because they are large or reproducible.

## Requirements

- Python 3.9 or newer
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- seaborn
- joblib
- psutil
- Streamlit (for the dashboard)
- TensorFlow (optional, for loading `.h5` models in the dashboard)

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install torch pandas numpy scikit-learn matplotlib seaborn joblib psutil streamlit
```

Install TensorFlow only if the dashboard needs to load Keras `.h5` models:

```bash
pip install tensorflow
```

## Dataset setup

The data files are not included in this repository. Prepare two CSV files:

- A source/pretraining dataset, such as CIC-IDS 2017
- A target/fine-tuning dataset containing IoT traffic

Both files must contain a `Label` column. Features are converted to numeric values, missing values are filled with zero, and multiclass attack labels are reduced to binary labels (`Benign = 0`, `Attack = 1`).

Update these paths in `config.py` before training:

```python
PRETRAIN_PATH = r"path/to/pretraining_data.csv"
FINETUNE_PATH = r"path/to/finetuning_data.csv"
```

Avoid committing the datasets; CSV and generated data files are covered by `.gitignore`.

## Training

Run the complete model comparison and transfer-learning pipeline:

```bash
python train_transfer_learning.py
```

The pipeline:

1. Loads and preprocesses the source dataset
2. Pretrains each model
3. Saves the best pretrained weights
4. Fine-tunes on the target IoT dataset
5. Evaluates each model and records comparison metrics

Generated weights and results are written to `saved_models/` and `results/`.

For a shorter validation run:

```bash
python train_quick_test.py
```

## Demo dashboard

Start the interactive dashboard with:

```bash
streamlit run working_demo.py
```

The dashboard searches `04_models/` for Keras `.h5` models. If compatible models or TensorFlow are unavailable, it can use mock predictions for demonstration purposes; those predictions are not suitable for production evaluation.

## Results and visualizations

Generate plots from saved experiment results:

```bash
python visualize_results.py
```

Example comparison charts are available in the `figures/` directory. The included dataset report summarizes 1,087,517 flows across benign, DDoS, port-scan, DoS, brute-force, web-attack, and bot traffic.

## Reproducibility notes

- `config.py` currently contains machine-specific Windows paths; replace them for your environment.
- The loader uses a stratified 80/20 training-validation split with a fixed random state of 42.
- Review the data split carefully to prevent leakage between source, target, training, validation, and test data.
- Do not treat demonstration-mode metrics as trained-model results.

## License

No license has been added yet. Add a license before redistributing or accepting external contributions.
