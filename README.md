# FactoryPulse AI
### Industrial Condition Monitoring & Intelligent Maintenance

**PyTorch · Deep Learning · LSTM Autoencoder · Time-Series Analysis · Industrial AI**

FactoryPulse AI is an industrial AI engineering project focused on analyzing real-world sensor data, learning equipment operating patterns, and developing intelligent anomaly detection capabilities for predictive maintenance applications.

The project uses the **MetroPT-3 Air Compressor dataset** and follows a modular approach from data engineering to deep learning, with future integration into a production-oriented monitoring application.

**Project Status:** LSTM Autoencoder baseline implemented and trained. Model evaluation is the next development phase.

---

## 1. Project Objectives

FactoryPulse AI aims to:

- Process and validate large-scale industrial sensor datasets.
- Handle missing time intervals and discontinuous sensor recordings.
- Prepare reliable time-series sequences for deep learning.
- Develop a PyTorch-based LSTM Autoencoder for reconstruction-based anomaly detection.
- Evaluate abnormal operating patterns against documented equipment failure periods.
- Build a modular foundation for future real-time monitoring, APIs, and cloud deployment.

## 2. Dataset

**Dataset:** MetroPT-3 — Air Compressor Monitoring Data

The current processed dataset contains:

| Property | Value |
|---|---|
| Total records | 1,516,948 |
| Processed columns | 19 |
| Time range | February – September 2020 |
| Time-series segments | 332 |
| Primary ML sensors | 4 |
| Documented failure periods | 4 |

### Selected Sensors

| Sensor | Description |
|---|---|
| `Motor_current` | Electric motor current |
| `Oil_temperature` | Compressor oil temperature |
| `TP2` | Pressure measurement |
| `TP3` | Pressure measurement |

The dataset also contains additional analog and digital signals that may be used in future operating-state analysis.

**Data availability:** The raw dataset and generated Parquet files are not included in this repository. They are stored locally and excluded through `.gitignore`.

## 3. Data Engineering Pipeline

The data processing pipeline implements:

**Data ingestion and validation**
- CSV loading and timestamp processing
- Sensor data quality checks
- Chronological ordering
- Missing-value checks

**Time-gap handling**
- Detection of sampling gaps greater than 60 seconds
- Assignment of `segment_id` values to continuous recording intervals
- Prevention of time-series sequences crossing recording discontinuities

**ML preprocessing**
- Chronological 80/20 Train/Test split
- Exclusion of documented failure intervals from training data
- StandardScaler fitted exclusively on the training subset
- Application of the fitted scaler to unseen data

The processed dataset is stored in Parquet format for efficient reuse.

## 4. Deep Learning Architecture

FactoryPulse AI currently implements an **LSTM Autoencoder using PyTorch**.

The model learns to reconstruct multivariate sensor sequences. Reconstruction errors will be investigated as potential indicators of unusual equipment behavior.

### Model Architecture

```text
Industrial Sensor Data
         |
         v
Sequence Dataset
(60 time steps × 4 sensors)
         |
         v
    LSTM Encoder
    Hidden Size: 32
         |
         v
 Latent Representation
         |
         v
    LSTM Decoder
         |
         v
     Linear Layer
         |
         v
Reconstructed Sensor Sequence
(60 time steps × 4 sensors)
         |
         v
  Reconstruction Error
         |
         v
Future Anomaly Detection
```

### Sequence Preparation

A custom PyTorch `Dataset` generates valid rolling sequences without materializing every overlapping window in memory.

The implementation ensures that sequences do not cross:

- Different `segment_id` values
- Discontinuities caused by excluded training records

Typical input batch:

```text
(batch_size, sequence_length, num_sensors)
             (32, 60, 4)
```

**Valid training sequences:** 1,168,841

## 5. Initial Training Results

The first LSTM baseline was trained on a reproducible random subset of 2,048 valid training sequences.

| Parameter | Configuration |
|---|---|
| Framework | PyTorch |
| Model | LSTM Autoencoder |
| Input features | 4 sensors |
| Sequence length | 60 |
| Hidden size | 32 |
| Batch size | 32 |
| Optimizer | Adam |
| Learning rate | 0.001 |
| Loss function | Mean Squared Error |
| Epochs | 3 |
| Training subset | 2,048 sequences |
| Device | CPU |

### Observed Training Loss

| Epoch | Training Loss |
|---|---|
| 1 | 0.805409 |
| 2 | 0.560894 |
| 3 | 0.453304 |

The training loss decreased by approximately **43.7%** between the first and third epochs.

This demonstrates that the baseline network can optimize its reconstruction objective on the selected training subset.

**Important:** These are preliminary training results, not anomaly detection accuracy metrics. Failure detection performance, generalization, false alarm rates, and model thresholds have not yet been validated.

## 6. Project Structure

```text
factorypulse-ai/
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── models/
│   ├── lstm_autoencoder_baseline.pt
│   └── sensor_scaler.joblib
│
├── scripts/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   └── train_lstm.py
│
├── src/
│   └── factorypulse/
│       ├── ingestion/
│       ├── validation/
│       │   └── data_validator.py
│       ├── ml/
│       │   ├── data_preparation.py
│       │   ├── sequence_dataset.py
│       │   └── lstm_autoencoder.py
│       ├── api/
│       ├── database/
│       ├── monitoring/
│       └── core/
│
├── tests/
│   └── test_data_validator.py
│
├── docs/
│
├── .gitignore
└── README.md
```

*The architecture includes reserved directories for future modules. Generated datasets and model artifacts are stored locally and are not version-controlled.*

## 7. Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python 3.12 |
| Data Processing | Pandas, NumPy, Parquet |
| Machine Learning Preprocessing | Scikit-learn |
| Deep Learning | PyTorch |
| Model Architecture | LSTM Autoencoder |
| Model Serialization | PyTorch, Joblib |
| Testing | Pytest |
| Version Control | Git, GitHub |
| Development Environment | PyCharm |

**Planned integrations:** FastAPI, MQTT, Database, TensorBoard, Docker, and Microsoft Azure.

## 8. Getting Started

### Clone the Repository

```bash
git clone https://github.com/amin-ghasemi-de/factorypulse-ai.git
cd factorypulse-ai
```

### Create a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install Dependencies

Example CPU-based setup:

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install pandas numpy scikit-learn pyarrow joblib pytest
```

### Prepare the Dataset

Place the MetroPT-3 CSV file at:

```text
data/raw/MetroPT3(AirCompressor).csv
```

Run:

```powershell
python scripts/prepare_data.py
```

This generates:

```text
data/processed/metropt3_processed.parquet
```

### Train the LSTM Autoencoder

Set the source directory as the Python import path:

```powershell
$env:PYTHONPATH="src"
```

Then execute:

```powershell
python scripts/train_lstm.py
```

The training script prepares the ML data, generates valid sequences, trains the baseline model, and saves the model weights and fitted scaler to the local `models/` directory.

### Run Tests

```powershell
$env:PYTHONPATH="src"
python -m pytest -q
```

## 9. Development Roadmap

- [x] Industrial data ingestion and initial validation
- [x] Time-gap identification and segmentation
- [x] Parquet data preparation
- [x] PyTorch Dataset and DataLoader
- [x] LSTM Autoencoder architecture
- [x] Initial LSTM baseline training
- [x] Model and scaler serialization
- [ ] Independent validation and temporal evaluation design
- [ ] Reconstruction-error analysis and anomaly threshold calibration
- [ ] Evaluation against documented failure periods
- [ ] Expanded model training and experiment tracking with TensorBoard
- [ ] FastAPI backend and inference endpoints
- [ ] Database and MQTT sensor stream simulation
- [ ] Industrial monitoring dashboard
- [ ] Docker containerization and Microsoft Azure deployment
- [ ] End-to-end demonstration and portfolio release

## 10. Current Limitations

FactoryPulse AI is currently a research and development prototype.

The current LSTM baseline has only been trained on a small subset of the available training sequences. The initial chronological test partition does not contain the four documented failure periods.

Therefore, reconstruction performance and failure detection capabilities still require independent temporal evaluation.

The system does not yet provide validated predictive maintenance decisions or production-ready anomaly alerts.

## 11. Engineering Focus

This project demonstrates hands-on development in:

- Industrial Data Engineering
- Multivariate Time-Series Processing
- PyTorch Deep Learning
- LSTM-based Neural Network Design
- Data Validation and Preprocessing
- Modular Python Architecture
- Reproducible Machine Learning Experimentation

The long-term objective is to evolve FactoryPulse AI into an end-to-end industrial monitoring platform combining sensor data processing, intelligent anomaly detection, backend services, visualization, and cloud deployment.

---

**Developed by [Amin Ghasemi](https://github.com/amin-ghasemi-de)**

*FactoryPulse AI — From Industrial Sensor Data to Intelligent Maintenance.*