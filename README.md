# ML Model Serving API

**Individual project — Datalligence ML Model Serving API**

A small end-to-end machine-learning project that trains a classifier and serves predictions through a validated FastAPI endpoint. It uses the built-in Iris dataset from scikit-learn so the project is reproducible without downloading a separate dataset.

## Architecture

```text
Built-in Iris dataset
        |
        v
train.py -> train/test split -> StandardScaler + Random Forest
        |                              |
        v                              v
evaluation.json                 iris_model.joblib
                                       |
                                       v
                                FastAPI /predict
                                       |
                              Pydantic validation
                                       |
                          prediction + probabilities
                                       |
                                     Docker
```

## Features

- Train/test split with stratification and a fixed random seed.
- Model evaluation: accuracy, confusion matrix, per-class precision/recall/F1.
- Model persistence with `joblib`.
- `POST /predict` returns a class prediction, class probabilities and model version.
- Request validation using Pydantic; malformed feature counts and extra fields are rejected.
- Health endpoint, structured application logging and request latency header.
- Pytest API tests and a simple latency/throughput benchmark.
- Docker container support.

## Dataset and input format

The Iris dataset contains 150 observations, four measurements in centimetres, and three target classes: setosa, versicolor, virginica.

Input feature order:
1. sepal length (cm)
2. sepal width (cm)
3. petal length (cm)
4. petal width (cm)

Example request:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

Example response (the exact probabilities are model-dependent):

```json
{
  "prediction": "setosa",
  "probabilities": {
    "0": 0.98,
    "1": 0.01,
    "2": 0.01
  },
  "model_version": "1.0.0"
}
```

Probability keys are the model's numeric class labels (`0`, `1`, `2`); the prediction itself is human-readable. Do not interpret this educational demo as a high-stakes decision system.

## Setup on Windows

Open PowerShell in this project folder. Python 3.11 or newer is recommended.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train.py
```

If PowerShell blocks activation, run the commands without activating the environment using `.venv\Scripts\python.exe -m pip install -r requirements.txt` and `.venv\Scripts\python.exe train.py`.

## Run the API

```powershell
python -m uvicorn app.main:app --reload
```

Open:
- Interactive API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Prediction: `POST http://127.0.0.1:8000/predict`

Use the example JSON above in Swagger UI (`/docs`).

## Tests

```powershell
python -m pytest -v
```

## Benchmark

Start the API in one terminal, then run this in another:

```powershell
python scripts/benchmark.py
```

It sends 100 sequential requests and reports mean/median/P95 latency and approximate successful requests per second. This is a basic local benchmark, not a load test. For meaningful CPU/memory observations, record your operating system, CPU, RAM, Python version, and the benchmark output. Do not claim performance results until you have run it on your own machine.

## Docker

Install Docker Desktop, then:

```powershell
docker build -t ml-model-serving-api .
docker run --rm -p 8000:8000 ml-model-serving-api
```

The image trains the model during the build and starts the API when the container runs.

## Training vs. serving

- **Training** uses historical examples to fit model parameters and evaluate generalization. In this project, `train.py` creates and saves the model artifact.
- **Serving** loads the saved artifact and uses it to predict on new input. The API does not retrain for each request.

## Limitations and future work

- Iris is a small educational dataset; results do not imply performance on other domains.
- Benchmark results depend on hardware and environment.
- Add authentication, rate limiting, model monitoring, model/data versioning, CI, and a production-grade concurrency/load test before real deployment.
- Add a drift strategy and retraining workflow if the model is used with changing data.

## Submission checklist

- [x] Training script and saved-model workflow
- [x] FastAPI prediction endpoint and request/response schemas
- [x] Error handling and logging
- [x] Dockerfile
- [x] Automated tests
- [x] Benchmark script
- [x] Architecture diagram and setup guide
- [ ] Run tests and benchmark on your machine; add actual output/screenshots
- [ ] Create a GitHub repository and push the source code
- [ ] Review the implementation and be prepared to explain every component
