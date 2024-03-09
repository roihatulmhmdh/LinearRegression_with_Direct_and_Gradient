import numpy as np

class GradientDescent:
    """
    Linear Regression dengan Optimasi Gradient Descent

    Parameter
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
    
    def process(self, X, y, learning_rate, max_iter, tol):
        """
        proses data training dengan menggunakan gradient descent
        
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

        # inisiasi parameter
        self.weight_ = np.zeros(n_features)
        self.intercept_ = 0

        # looping untuk update parameter
        for i in range(max_iter):

            # prediksi dengan parameter yang diketahui
            y_pred = np.dot(X, self.weight_) + self.intercept_

            # menghitung gradient
            grad_weight = (y_pred - y).dot(X) / n_samples
            grad_intercept = sum(y_pred - y) / n_samples

            # update parameter berdasarkan alpha dan gradient
            self.weight_ -= learning_rate * grad_weight
            self.intercept_ -= learning_rate * grad_intercept

            # Break iterasi
            list_grad = np.append(grad_weight, grad_intercept)
            if all(np.abs(list_grad) < tol):
                 break