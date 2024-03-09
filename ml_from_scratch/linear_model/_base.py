import numpy as np
from ._normalequation import DirectSolution
from ._gradientdescent import GradientDescent

class LinearRegression:
    """
    Linear Regression dengan Direct Solution atau Optimasi Gradient Descent

    Parameter
    ----------
    solver: str, default = 'direct'
        metode untuk menyelesaikan permasalahan
        'direct' untuk Direct Solution dan 'Gradient' untuk Gradient Descent
    
    Hyperparameter untuk Gradient Descent
    ----------
    learning_rate: float, default = 0.01
        Learning rate menunjukkan besar perubahan yang diinginkan
    
    max_iter : int, default = 1000000
        Maksimum itearasi untuk mencapai titik optimum / solusi konvergen
        
    tol : float, default = 1e-5
        Iterasi akan berhenti jika perubahan gradient kurang dari toleransi
        all(abs(NLL gradient)) < tol
    
    Attributes
    ----------
    weight_ : ndarray of shape (n_features,)
        Weight atau coefficient
    
    intercept_ : ndarray of shape (1,)
        Intercept atau bias

    """
    def __init__(
        self,
        solver='direct',
        learning_rate=0.01,
        max_iter=10000,
        tol=1e-5
    ):
        self.solver = solver
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """
        Fit dengan menggunakan data training
        
        Parameters
        ----------
        X : matrix berukuran (n_samples, n_features)
            Training vector, dengan `n_samples` merupakan jumlah sample dan
            `n_features` merupakan jumlah feature
        
        y : array berukuran (n_samples,)
            Target vector terhadap X
        
        Returns
        --------
        self 
            Estimasi weight dan intercept 
        """

        # solusi direct
        if self.solver == 'direct':
            
            result = DirectSolution()
            result.process(X, y)
            self.weight_ = result.weight_
            self.intercept_ = result.intercept_
        
        # solusi gradient descent
        if self.solver == 'gradient':
            
            result = GradientDescent()
            result.process(X, y, self.learning_rate, self.max_iter, self.tol)
            self.weight_ = result.weight_
            self.intercept_ = result.intercept_
            
    
    def predict(self, X):
        """
        Prediksi output mengggunakan linear model.

        Parameters
        ----------
        X : matrix berukuran (n_samples, n_features)

        Returns
        -------
        y_pred : array berukuran (n_samples,)
            Nilai prediksi
        """
        X = np.array(X)
        y_pred = np.dot(X, self.weight_) + self.intercept_
        
        return y_pred