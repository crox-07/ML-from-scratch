# ML from Scratch

Implementing machine learning models from scratch using only NumPy, to understand how they work under the hood.

## Goal

Build common ML algorithms without relying on high-level libraries like scikit-learn. Each implementation focuses on the core math and mechanics — no black boxes.

## Models

| Model | Algorithm | File |
|---|---|---|
| Linear Regression | Gradient Descent | `ml_lib/models/linear_regression_GD.py` |

## Usage

```python
from ml_lib.models.linear_regression_GD import LinearRegressionGD

model = LinearRegressionGD(n_iter=2000, lr=0.01)
model.fit(X_train, y_train, plot=True)

y_pred = model.predict(X_test)
print(model.score(X_test, y_test))  # R² score
```

## Project Structure

```
ML-from-scratch/
├── data/               # Datasets
├── ml_lib/
│   └── models/         # Model implementations
├── notebooks/          # Experiments and walkthroughs
└── requirements.txt
```

## Dependencies

```
pip install -r requirements.txt
```

Only NumPy is used for model implementations. Matplotlib is used for plotting and scikit-learn only for utilities (train/test split, dataset generation).
