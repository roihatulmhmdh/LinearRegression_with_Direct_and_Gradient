Solve Linear Regression Problems with Direct Solution and Gradient Descent

# Component of Learning

1. Cost Function : average squared error
1. Objective Optimasi : meminimalkan average squared error
1. Model Parameter : Weight dan Intercept
1. Algoritma Optimasi : Direct Solution dan Gradient Descent
1. Prediksi : Formula y = W.T X + b


# Pseudocode 

## Direct Solution

Input:
- X : input training dataset
- y : output training dataset

Output:
- weight
- intercept

```
X = matrix X (data input)
y = array y (data output)

Menyelesaikan persamaan, theta = inverse(X.T X) (X.T y)

Ekstrak model parameter, weight dan intercept, pada theta
weight = dengan menghilangan data point terakhir di theta
intercept = data point terakhir theta
```

## Gradient Descent

Input:
- X : input training dataset
- y : output training dataset
- a : learning rate
- max_iter : maximum iteration
- tol : tolerance

Output:
- weight
- intercept

Stopping Criterion:
- maksimum iterasi
- perubahan gradient kurang dari toleransi

```
X = matrix X (data input)
y = array y (data output)

Membuat kondisi looping
  Prediksi y dengan theta yang diketahui
  Hitung gradient
  Update theta
  theta baru = theta lama - alpha.gradient
  Kondisi berhenti jika 
    iterasi maksimum atau perubahan gradient kurang dari toleransi
```

## Prediction

Input :
- X : input test dataset

Ouput :
- prediksi

```
X = matrix X (nilai feature)
prediksi = weight X + intercept
```

# Run Code

Input model:
- X : matrix. Dataset input
- y : array. Dataset output
- solver : str, default = 'direct'. Metode untuk menyelesaikan permasalahan dengan 'direct' untuk Direct Solution dan 'Gradient' untuk Gradient Descent
- learning_rate : float, default = 0.01. Learning rate menunjukkan besar perubahan yang diinginkan
- max_iter : int, default = 1000000. Maksimum itearasi untuk mencapai titik optimum / solusi konvergen
-tol : float, default = 1e-5. Iterasi akan berhenti jika perubahan gradient kurang dari toleransi

```
# import library
from ml_from_scratch.linear_model import LinearRegression

# fit model dengan dataset
reg = LinearRegression()
reg.fit(X, y)

# ekstrak weight dan intercept
weight = reg.weight_
intercept = reg.intercept_

# prediction output
predict = reg.predict(X)
```

Code menghitung MSR (Mean Squared Error)
```
from sklearn.metrics import mean_squared_error

mean_squared_error(y, predict)
```

Medium : 
https://medium.com/@roihatul.mahmudah/solve-linear-regression-problems-with-direct-solution-and-gradient-descent-512bea1b7df6

Reference:
- Roger Grosse, Amir-massoud Farahmand, Juan Carrasquilla. University of Toronto lecture note: Linear Regression. https://www.cs.toronto.edu/~mren/teach/csc411_19s/lec/lec06_matt.pdf
- Antonio Ferramosca. University of Bergamo lecture note: Linear Regression. https://cal.unibg.it/wp-content/uploads/DSI/slide/Lecture-02-Linear-regression.pdf