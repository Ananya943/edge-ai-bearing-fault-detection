# ⚙️ Edge AI Predictive Maintenance: Bearing Fault Detection

An end-to-end **machine learning and signal-processing project for predictive maintenance**, designed to detect bearing faults from vibration signals and explore a lightweight workflow suitable for **edge and embedded-system deployment**.

The project combines **Python, MATLAB, signal processing, feature engineering, machine learning, model evaluation, and algorithm optimization** to classify the operating condition of industrial bearings.

---

## 📌 Project Overview

Unexpected bearing failures can result in equipment downtime, maintenance costs, and production losses.

This project develops a machine learning pipeline that analyzes vibration signals and identifies different bearing conditions:

* 🟢 Normal Bearing
* 🔴 Inner Race Fault
* 🟠 Outer Race Fault
* 🟡 Ball Fault

The objective is to transform raw vibration measurements into meaningful signal features, train and compare multiple machine learning algorithms, and investigate a lightweight model suitable for deployment closer to the data source.

### Workflow

```text
Vibration Data
      ↓
Data Preprocessing
      ↓
Signal Processing
      ↓
Feature Extraction
      ↓
Feature Selection
      ↓
Machine Learning
      ↓
Model Evaluation
      ↓
Model Optimization
      ↓
Edge / Embedded Inference
```

---

## 🎯 Project Objectives

1. Analyze vibration signals from rotating machinery.
2. Perform exploratory analysis of bearing conditions.
3. Extract meaningful time-domain and frequency-domain features.
4. Develop machine learning models for fault classification.
5. Compare different algorithms using appropriate evaluation metrics.
6. Optimize the feature set for efficient inference.
7. Implement the analytical workflow in MATLAB.
8. Explore an embedded/edge-oriented inference architecture.
9. Document the complete ML engineering workflow.

---

## 📊 Dataset

The project uses the **Case Western Reserve University (CWRU) Bearing Dataset**.

The dataset contains vibration measurements collected from bearings under different operating conditions and fault types.

### Fault categories

| Label      | Description                |
| ---------- | -------------------------- |
| Normal     | Healthy bearing            |
| Inner Race | Inner-race bearing fault   |
| Outer Race | Outer-race bearing fault   |
| Ball       | Rolling-element/ball fault |

The dataset contains different operating conditions and fault severities, allowing the model to be evaluated beyond a single fixed condition.

### Dataset Source

**Case Western Reserve University Bearing Data Center**

https://engineering.case.edu/bearingdatacenter

A Kaggle version of the CWRU dataset is used for convenient project development and experimentation.

---

# 🛠️ Technologies Used

### Programming

* Python
* MATLAB

### Python Libraries

* NumPy
* Pandas
* SciPy
* Scikit-learn
* Matplotlib
* Seaborn

### Machine Learning

* Decision Tree
* Random Forest
* Support Vector Machine
* Neural Network

### Signal Processing

* Fast Fourier Transform (FFT)
* Statistical feature extraction
* Frequency-domain analysis
* Time-domain analysis
* Signal filtering
* Windowing

### Deployment Concepts

* Edge AI
* Embedded ML
* Lightweight inference
* Feature reduction
* Model optimization

---

# 🏗️ Project Architecture

```text
                  ┌──────────────────────┐
                  │   Vibration Signal   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Preprocessing      │
                  │ Filtering / Windowing│
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Signal Processing    │
                  │ FFT / Statistics    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Engineering  │
                  │ Time + Frequency     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Selection    │
                  └──────────┬───────────┘
                             │
                             ▼
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
     ┌─────────────────┐          ┌─────────────────┐
     │ Machine Learning│          │ MATLAB Workflow │
     │ Models          │          │                 │
     └────────┬────────┘          └────────┬────────┘
              │                            │
              └──────────────┬─────────────┘
                             ▼
                  ┌──────────────────────┐
                  │ Model Evaluation     │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Model Optimization   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Edge / Embedded      │
                  │ Inference Concept    │
                  └──────────────────────┘
```

---

# 🔬 Feature Engineering

The vibration signals are transformed into numerical features that can be used by machine learning algorithms.

## Time-Domain Features

Examples include:

* Mean
* Standard deviation
* Variance
* RMS
* Maximum amplitude
* Minimum amplitude
* Peak-to-peak value
* Skewness
* Kurtosis
* Crest factor

## Frequency-Domain Features

Examples include:

* Dominant frequency
* Spectral energy
* Frequency-band energy
* Peak frequency
* FFT magnitude statistics

These features allow the model to identify differences between healthy and faulty bearing signals.

---

# 🤖 Machine Learning Models

Several algorithms are evaluated rather than relying on a single model.

### 1. Decision Tree

Provides an interpretable baseline model and helps understand how extracted features contribute to classification.

### 2. Random Forest

An ensemble approach used to capture nonlinear relationships between vibration features and bearing conditions.

### 3. Support Vector Machine

An SVM model is evaluated for its ability to separate different fault classes in feature space.

### 4. Neural Network

A neural-network-based model can additionally be evaluated for comparison with traditional machine learning approaches.

---

# 📈 Model Evaluation

The models are evaluated using multiple metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

The project avoids relying solely on accuracy because class-level performance is important in fault diagnosis.

Example evaluation workflow:

```text
Training Data
      ↓
Model Training
      ↓
Validation
      ↓
Test Data
      ↓
Predictions
      ↓
Confusion Matrix
      ↓
Precision / Recall / F1
```

> Model performance numbers in this README will be updated after the final experiments are completed.

---

# ⚡ Algorithm Optimization

For an edge/embedded application, model accuracy is not the only consideration.

The project also investigates:

* Feature reduction
* Model complexity
* Inference efficiency
* Number of input features
* Computational requirements
* Potential memory constraints

The objective is to find a practical balance between:

```text
Model Accuracy
       +
Computational Efficiency
       +
Deployment Simplicity
```

This makes the project more relevant to real-world edge ML applications.

---

# 🧠 MATLAB Implementation

A parallel MATLAB workflow is included for:

* Loading vibration data
* Signal visualization
* Signal processing
* FFT analysis
* Feature extraction
* Machine learning
* Model evaluation

Example workflow:

```matlab
data = readtable("bearing_features.csv");

X = data{:, featureColumns};
Y = categorical(data.FaultType);

model = fitcsvm(X, Y, ...
    'KernelFunction', 'rbf', ...
    'Standardize', true);

predictions = predict(model, X);
```

The MATLAB implementation demonstrates how the same machine learning workflow can be developed and analyzed using MATLAB-based engineering tools.

---

# 🔌 Edge / Embedded System Concept

The final stage of the project explores how the trained model could operate closer to the machine.

```text
     Vibration Sensor
            │
            ▼
     Microcontroller
            │
            ▼
      Signal Window
            │
            ▼
    Feature Extraction
            │
            ▼
      ML Classifier
            │
            ▼
    ┌───────┴────────┐
    │                │
 Normal           Fault
    │                │
    ▼                ▼
Continue         Alert / Maintenance
Operation          Required
```

Instead of continuously sending raw sensor data to a cloud server, a lightweight model can potentially perform inference locally.

This architecture can help reduce:

* Data transmission
* Response latency
* Dependence on cloud connectivity

Actual embedded deployment and hardware performance will depend on the selected microcontroller and model constraints.

---

# 📁 Repository Structure

```text
edge-ai-bearing-fault-detection/
│
├── data/
│   └── README.md
│
├── notebooks/
│   └── bearing_fault_analysis.ipynb
│
├── python/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate_models.py
│   └── model_optimization.py
│
├── matlab/
│   ├── preprocessing.m
│   ├── feature_extraction.m
│   ├── train_model.m
│   └── evaluation.m
│
├── embedded/
│   ├── inference.c
│   └── README.md
│
├── results/
│   ├── confusion_matrix.png
│   ├── model_comparison.png
│   └── feature_importance.png
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

# 🚀 Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/Ananya943/edge-ai-bearing-fault-detection.git

cd edge-ai-bearing-fault-detection
```

## 2. Create a virtual environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Prepare the dataset

Download the CWRU bearing dataset and place the required files inside:

```text
data/
```

Follow the instructions in:

```text
data/README.md
```

## 5. Run the analysis

Open:

```text
notebooks/bearing_fault_analysis.ipynb
```

or run the Python scripts in the following order:

```text
preprocessing.py
        ↓
feature_engineering.py
        ↓
train_models.py
        ↓
evaluate_models.py
        ↓
model_optimization.py
```

---

# 📊 Results

The final results section will contain:

* Model comparison
* Confusion matrices
* Feature importance
* Classification metrics
* Feature-reduction analysis
* Inference-time comparison

Example:

| Model          | Accuracy | Precision | Recall | F1-Score |
| -------------- | -------: | --------: | -----: | -------: |
| Decision Tree  |      TBD |       TBD |    TBD |      TBD |
| Random Forest  |      TBD |       TBD |    TBD |      TBD |
| SVM            |      TBD |       TBD |    TBD |      TBD |
| Neural Network |      TBD |       TBD |    TBD |      TBD |

**Results will be populated from the actual experiments and will not be manually estimated.**

---

# 🔮 Future Improvements

Potential extensions include:

* Real-time vibration acquisition
* Arduino/ESP32 implementation
* STM32 deployment
* TinyML model conversion
* TensorFlow Lite / Lite Micro inference
* Real-time fault alerts
* Remaining Useful Life (RUL) prediction
* Streaming sensor data
* Model quantization
* Hardware-in-the-loop testing
* Cloud + edge hybrid architecture

---

# 💼 Skills Demonstrated

This project demonstrates practical experience in:

* Machine Learning
* Python
* MATLAB
* Signal Processing
* Feature Engineering
* Classification
* Model Evaluation
* Algorithm Development
* Model Optimization
* Predictive Maintenance
* Vibration Analysis
* Edge AI
* Embedded ML Concepts
* Data Visualization

---

# 📚 Dataset Reference

**Case Western Reserve University Bearing Data Center**

The dataset is used for research and educational experimentation involving bearing fault diagnosis.

Dataset website:

https://engineering.case.edu/bearingdatacenter

---

# 👨‍💻 Author

**Ananya**

Data Science | Machine Learning | AI

Interested in developing practical machine learning solutions for **predictive analytics, automation, and intelligent systems**.

---

⭐ If you find this project useful, consider giving the repository a star.
