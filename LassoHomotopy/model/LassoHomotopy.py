import numpy as np 

class LassoHomotopy:
    def __init__(self, alpha=0.1, tol=1e-6, max_iter=1000):
        """
        Initialize LASSO Homotopy model.
        Parameters:
        - alpha: Regularization parameter that controls the level of sparsity.
        - tol: Tolerance for convergence; determines when the algorithm stops.
        - max_iter: Maximum number of iterations allowed during optimization.
        """
        self.alpha = alpha
        self.tol = tol
        self.max_iter = max_iter
        self.coef_ = None
        self.active_set = set()

    def _standardize(self, X, y):
        """Standardizes the feature matrix X and centers the target vector y for numerical stability"""
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0)
        self.y_mean = np.mean(y)

        # Avoid division by zero
        self.X_std[self.X_std == 0] = 1.0

        X_scaled = (X - self.X_mean) / self.X_std
        y_centered = y - self.y_mean

        return X_scaled, y_centered

    def fit(self, X, y):
        """Fits the LASSO model using the Homotopy method."""
        X, y = self._standardize(X, y)
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        residuals = y.copy()
        self.active_set = set()

        for iteration in range(self.max_iter):
            # Compute the correlation between each feature and the residuals
            correlations = X.T @ residuals

            # Identify the feature with the strongest correlation
            max_idx = np.argmax(np.abs(correlations))
            max_corr = correlations[max_idx]

            # If the maximum correlation is below the regularization threshold, stop iteration
            if np.abs(max_corr) < self.alpha:
                break  
            # Add the selected feature to the active set
            self.active_set.add(max_idx)
            active_X = X[:, list(self.active_set)]

            # Solve for the least squares solution on the active set
            try:
                active_coefs = np.linalg.pinv(active_X) @ y
            except np.linalg.LinAlgError:
                active_coefs = np.linalg.lstsq(active_X, y, rcond=None)[0]

            # Update the coefficients for the active features
            for idx, coef in zip(self.active_set, active_coefs):
                self.coef_[idx] = coef

            # Compute new residuals
            residuals = y - X @ self.coef_

            # Check for convergence
            if np.linalg.norm(residuals) < self.tol:
                break

        return self

    def predict(self, X):
        """Generates predictions for new data using the trained model."""
        X_scaled = (X - self.X_mean) / self.X_std
        return X_scaled @ self.coef_ + self.y_mean
