import numpy as np

class DirectSolution:
    """
    Linear Regression dengan Direct Solution
    
    Attributes
    ----------
    weight_ : ndarray of shape (n_features,)
        Weight atau coefficient
    
    intercept_ : ndarray of shape (1,)
        Intercept atau bias

    """

    def process(self, X, y):
        """
        proses data training dengan menggunakan direct solution
        
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
        # jadian dalam bentuk np
        X = np.array(X).copy()
        y = np.array(y).copy()

        # extract size dari data
        n_samples, n_features = X.shape
            
        # cari solusi, theta
        X = np.column_stack((X, np.ones(n_samples)))
        theta = np.linalg.inv(X.T @ X) @ X.T @ y 

        # ekstrak coef_ dan intercept_
        self.weight_ = theta[:-1]
        self.intercept_ = theta[-1] 