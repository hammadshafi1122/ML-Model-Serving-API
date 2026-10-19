# Research and Evaluation Report — ML Model Serving API

**Student:** Muhammad Hammad Shafi  
**Project:** ML Model Serving API  
**Dataset:** scikit-learn Iris  
**Date:** [insert date]

## 1. Objective
Train a classification model, save the artifact, expose it through FastAPI, validate inputs, and evaluate serving performance.

## 2. Method
- Dataset: 150 Iris records, four numeric features, three classes.
- Split: 80% training / 20% test; stratified with random seed 42.
- Pipeline: StandardScaler followed by RandomForestClassifier (200 trees, random seed 42).
- Evaluation: accuracy, confusion matrix, precision, recall, F1-score.
- Serving: FastAPI + Pydantic; `POST /predict` returns the predicted class and probabilities.

## 3. Model evaluation
Run `python train.py` and copy the measured accuracy and confusion matrix from `artifacts/evaluation.json` here. Include the per-class metrics from the same file.

## 4. Serving evaluation
Run the API, then run `python scripts/benchmark.py`. Record:
- Machine / CPU:
- RAM:
- OS and Python version:
- Number of requests:
- Successful requests:
- Mean / median / P95 latency:
- Approximate throughput:

## 5. Training vs. serving
Training fits the model using the training split and produces a serialized artifact. Serving loads that artifact and predicts on incoming requests without retraining for each request.

## 6. Limitations
This is a small educational dataset and a basic sequential benchmark. Results may not generalize to other datasets or production traffic. The benchmark values must be measured on the target machine.

## 7. Conclusion and future work
Summarize the actual evaluation results after running the scripts. Potential next steps include automated CI, monitoring, model versioning, load testing, and authentication.
