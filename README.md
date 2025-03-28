# Team Members

| Name             | Student ID | Email                  |
| ---------------- | ---------- | ---------------------- |
| Avinash Ashokar  | A20575257  | aashokar@hawk.iit.edu  |
| Sanjana Patel    | A20584055  | spatel226@hawk.iit.edu |
| Pranitha Chilla  | A20586430  | pchilla@hawk.iit.edu   |
| Bharathwaj Muthu | A20567309  | bmuthu1@hawk.iit.edu   |

# How to Run

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Avinash-Ashokar/Project1.git
   cd Project1/
   ```

2. Create a virtual environment:

   ```bash
   python3 -m venv project1
   ```

   **or**

   ```bash
   python -m venv project1
   ```

3. Activate Virtual Environment

   **Linux/Mac:**

   ```bash
   source project1/bin/activate
   ```

   **Windows:**

   ```bash
   project1\bin\activate
   ```

4. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

5. Run the data generation script:

   ```bash
   python3 generate_regression_data.py \
      -N 1000 \
      -m 1.0 2.0 3.0 4.0 5.0 \
      -b 5.0 \
      -scale 2.5 \
      -rnge 0.0 10.0 \
      -seed 42 \
      -output_file ./LassoHomotopy/tests/small_test.csv
   ```

   **or**

   ```bash
   python generate_regression_data.py ^
      -N 1000 ^
      -m 1.0 2.0 3.0 4.0 5.0 ^
      -b 5.0 ^
      -scale 2.5 ^
      -rnge 0.0 10.0 ^
      -seed 42 ^
      -output_file ./LassoHomotopy/tests/small_test.csv
   ```

6. Change Directory to test folder

   ```bash
   cd LassoHomotopy/tests/
   ```

7. Execute pytest for testing the code

   ```bash
   pytest -s
   ```

# Frequently Asked Questions.

1. **What does the model you have implemented do and when should it be used ?**

   - The implemented model is a LASSO (Least Absolute Shrinkage and Selection Operator) regression solver using the Homotopy Method.

   - It is used for feature selection in high-dimensional datasets, enforcing sparsity by setting some coefficients to zero.

   - It is useful when interpretability is important and when identifying the most relevant features in a dataset.

2. **How did you test your model to determine if it is working reasonably correctly ?**

   - The model is tested using unit tests implemented in test_LassoHomotopy.py.

   - Tests check that the LASSO model correctly produces sparse solutions with collinear data (collinear_data.csv).

   - The test suite validates that MSE is non-negative and that R² values fall within the expected range.

   - Additionally, visualization techniques (scatter plots of predictions vs. actual values, coefficient bar plots) help confirm correctness.

3. **What parameters have you exposed to users of your implementation in order to tune performance ?**

   - alpha: Controls the regularization strength; higher values enforce more sparsity.

   - tol: Sets the convergence tolerance; lower values improve precision but may increase computation time.

   - max_iter: Defines the maximum iterations allowed; prevents infinite loops in slow convergence cases.

4. **Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental ?**

   - Large Datasets: The Homotopy method may be computationally expensive for large feature sets.

   - Non-Standardized Data: The model assumes features are standardized; unscaled data may lead to unexpected behavior.

   - Highly Correlated Features: While LASSO is designed to handle collinearity, it may struggle when multiple features are equally important.

   - Potential Workarounds: Using dimensionality reduction (e.g., PCA) or alternative optimization techniques could improve performance.

# Lasso Homotopy Implementation

This project implements the Lasso regression algorithm using the Homotopy method, which is an efficient approach for solving the L1-regularized least squares problem. The implementation includes data generation utilities, the core Lasso Homotopy algorithm, and comprehensive test cases.

## Table of Contents

1. [Architecture](#architecture)
2. [Implementation Details](#implementation-details)
3. [Usage Examples](#usage-examples)
4. [Test Cases](#test-cases)
5. [Dependencies](#dependencies)
6. [Sample Code](#sample-code)

## Architecture

The project is organized into the following structure:

```
.
├── generate_regression_data.py    # Data generation utility
├── requirements.txt              # Project dependencies
└── LassoHomotopy/
    ├── model/
    │   └── LassoHomotopy.py     # Core implementation
    └── tests/
        ├── test_LassoHomotopy.py # Test suite
        ├── small_test.csv        # Test dataset
        └── collinear_data.csv    # Test dataset
```

### Key Components

1. **Data Generation (`generate_regression_data.py`)**

   - Generates synthetic regression data with specified parameters
   - Supports customizable noise levels and feature dimensions
   - Outputs data in CSV format

2. **Lasso Homotopy Implementation (`LassoHomotopy.py`)**

   - Implements the Homotopy algorithm for Lasso regression
   - Features standardization and numerical stability improvements
   - Provides fit and predict methods

3. **Test Suite (`test_LassoHomotopy.py`)**
   - Comprehensive test cases for model validation
   - Performance visualization and metrics calculation
   - Tests for different dataset sizes and characteristics

## Implementation Details

### Lasso Homotopy Algorithm

The implementation follows the Homotopy method for solving the Lasso problem:

```
min ||y - Xβ||²₂ + α||β||₁
```

Key features of the implementation:

1. **Initialization**

   ```python
   def __init__(self, alpha=0.1, tol=1e-6, max_iter=1000):
   ```

   - `alpha`: Regularization parameter
   - `tol`: Convergence tolerance
   - `max_iter`: Maximum iterations for optimization

2. **Data Preprocessing**

   - Standardizes features and centers target variable
   - Handles numerical stability issues
   - Prevents division by zero in standardization

3. **Core Algorithm**
   - Uses active set method for feature selection
   - Implements least squares solution on active features
   - Updates coefficients iteratively
   - Checks convergence criteria

### Data Generation

The data generation utility creates synthetic regression data with:

- Customizable number of samples
- Configurable feature dimensions
- Adjustable noise levels
- Specifiable coefficient values

## Usage Examples

### 1. Generating Synthetic Data

```python
python generate_regression_data.py -N 1000 -m 1.0 0.5 -0.3 -b 2.0 -scale 0.1 -rnge -5 5 -seed 42 -output_file data.csv
```

### 2. Training the Lasso Homotopy Model

```python
from LassoHomotopy.model.LassoHomotopy import LassoHomotopy
import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data.csv')
X = df.filter(regex='^x_').values
y = df['y'].values

# Initialize and train model
model = LassoHomotopy(alpha=0.1)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```

## Test Cases

The test suite covers multiple scenarios and generates detailed visualizations for analysis. Running `pytest -s` produces the following results:

### Test Results

1. **Small Test Dataset Performance**

   - Mean Squared Error (MSE): 6.8039
   - R² Score: 0.9846
   - Indicates excellent model performance with strong predictive accuracy

2. **Collinear Data Performance**
   - Mean Squared Error (MSE): 4.4259
   - R² Score: 0.8288
   - Shows good performance even with collinear features

### Generated Visualizations

For each test dataset (`small_test.csv` and `collinear_data.csv`), the test suite generates two types of plots:

1. **Prediction Quality Plots** (`*_test_results_plot1.png`)

   - Scatter plot of predicted vs actual values
   - Red dashed line showing perfect prediction (y=x)
   - Helps visualize prediction accuracy and bias
   - Higher density of points near the diagonal indicates better performance

2. **Feature Importance Plots** (`*_test_results_plot2.png`)
   - Bar chart showing coefficient values for each feature
   - Helps identify which features have the strongest impact
   - Zero or near-zero coefficients indicate features eliminated by LASSO
   - Useful for feature selection analysis

### Test Coverage

The test suite validates the model across different scenarios:

1. **Basic Functionality Tests**

   - Model initialization and parameter validation
   - Data standardization correctness
   - Coefficient computation accuracy
   - Prediction generation reliability

2. **Performance Tests**

   - MSE calculation and validation (ensures non-negative values)
   - R² score computation (validates range between 0 and 1)
   - Feature importance visualization and coefficient analysis

3. **Dataset Variations**

   - Small dataset (20 samples) to test basic functionality
   - Larger dataset (200 samples) to validate scalability
   - Collinear features to test feature selection
   - Different noise levels to assess robustness

4. **Visualization Tests**
   - Generates and saves prediction quality plots
   - Creates feature importance visualizations
   - Validates plot generation and saving functionality

### Test Implementation Details

The test suite uses pytest's parametrize feature to run tests with different datasets:

```python
@pytest.mark.parametrize("csv_path", ["small_test.csv", "collinear_data.csv"])
def test_lasso_model(csv_path):
    # Test implementation
```

Key testing aspects:

- Automated performance metric calculation
- Visual result generation
- Dataset size variation testing
- Error handling validation

## Dependencies

Required packages (specified versions in requirements.txt):

```
numpy
scipy
pandas
matplotlib
pytest
scikit-learn
```

## Sample Code

### Complete Training Pipeline

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from LassoHomotopy.model.LassoHomotopy import LassoHomotopy
import matplotlib.pyplot as plt

# Load and prepare data
df = pd.read_csv('data.csv')
X = df.filter(regex='^x_').values
y = df['y'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LassoHomotopy(alpha=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

print(f"MSE: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# Visualize results
plt.figure(figsize=(10, 5))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Lasso Homotopy Model - Actual vs Predicted")
plt.show()
```

### Feature Importance Analysis

```python
# Plot feature coefficients
plt.figure(figsize=(10, 5))
plt.bar(range(len(model.coef_)), model.coef_)
plt.xlabel("Feature Index")
plt.ylabel("Coefficient Value")
plt.title("Lasso Homotopy Model - Feature Importance")
plt.show()
```
