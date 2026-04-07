import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class LinearRegressionGD:
    """
    Simple implementation of a Linear Regression model using gradient descent from scratch.
    
    Parameters
    ----------
    n_iter : int, default=1000
        Number of gradient descent iterations.
    
    lr : float, default=0.001 
        Learning rate (step size for weight updates).
    
    Attributes
    ----------
    theta : np.ndarray of shape (n_features + 1, 1)
        Matrix containing model weights, including bias term at row index 0.
        Set after calling ``fit()``.
    
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=200, n_features=1, noise=10)
    >>> model = LinearRegressionGD(n_iter=2000, lr=0.01)
    >>> model.fit(X, y)
    >>> model.score(X, y)
    0.98
    
    """
    def __init__(self, n_iter=1000, lr=0.001):
        self.theta = None
        self.n_iter = n_iter
        self.lr = lr

    def fit(self, X, Y, plot=False):
        """
        Train the model by minimising MSE via gradient descent.
        
        Parameters
        ----------
        X : np.ndarray of shape (m, n)
            Training feature matrix.
        Y : np.ndarray of shape (m,) or (m, 1)
            Target values.
        plot : bool, default=False
            If true, a plot showing loss per iteration will be displayed.
        """
        # Prepend a column of ones to the input to allow theta[0] to act as the bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
            
        self.theta = np.random.randn(X_b.shape[1], 1)
        m = X_b.shape[0]
        losses = []
        
        for _ in range(self.n_iter):
            error = X_b @ self.theta - Y
            # MSE: 1/m * Σ(error²)
            loss = np.sum(error**2) / m
            losses.append(loss)
            # Gradient of MSE w.r.t. theta = 2/m * Xᵀ(Xθ - y) (or sum of errors multiplied by X)
            gradients = 2/m * X_b.T @ error
            self.theta -= self.lr * gradients

        if plot:
            x = list(range(1, len(losses)+1))
            y = losses
            plt.scatter(x, y)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title('Training loss')
            plt.show()
    
    def predict(self, X):
        """
        Generate predictions for the given feature matrix
        
        Parameters
        ----------
        X : np.ndarray of shape (m,n)
            Input feature matrix (without bias column).
        
        Returns
        -------
        np.ndarray of shape (m, 1)
            Predicted values.
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.theta

    def score(self, X, Y):
        '''The function calculates the R-squared value for a regression model given the actual and predicted
        values.
        
        Parameters
        ----------
        X : np.ndarray of shape (m,n)
            The input feature matrix used for making predictions. 
        Y : np.ndarray of shape (m,1)
            The target values for a given feature matrix X.
        
        Returns
        -------
        float
            R-squared score of the model.
        '''
        Y_pred = self.predict(X).flatten()
        if Y.ndim > 1:
            Y = Y.flatten()
        Y_mean = np.mean(Y)
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y_mean) ** 2)
        r2 = 1 - ss_res/ss_tot
        return r2
        
        