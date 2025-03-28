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

**Linux/Mac:**

```bash
python3 -m venv project1
source project1/bin/activate
```

**Windows:**

```bash
python -m venv project1
project1\bin\activate
```

3. Install requirements:

```bash
pip install -r requirements.txt
```

4. Run the data generation script:

**Linux/Mac:**

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

**Windows:**

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

5. `cd LassoHomotopy/tests/`
6. `pytest -s`

# Few Questions.

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
