import numpy as np 
from scipy import stats

class LocalLinearRegression:
	def fit(self,X,y,kernel,bandwidth):
		B = np.c_[np.ones(X.shape),X]
		x0 = X[-1]

		weights = np.eye(X.shape[0])
		for i in range(X.shape[0]):
			weights[i][i] = kernel( abs(X[i] - x0) / bandwidth )

		self.coefficients = np.matmul(np.linalg.inv(np.matmul(B.T, np.matmul(weights, B))), np.matmul(B.T, np.matmul(weights,y)))

	def predict(self,X):
		return np.dot(np.c_[1,X],self.coefficients)

	@staticmethod
	def epanechnikov(t):
		if (abs(t) <= 1):
			return (3.0 / 4.0) * (1.0 - (t**2))
		else:
			return 0

	@staticmethod
	def tri_cube(t):
		if (abs(t) <= 1):
			return (1.0 - (abs(t)**3))**3
		else:
			return 0

	@staticmethod
	def gaussian(t):
		return stats.norm.pdf(t)


X = np.array([1,2,3,4]) #,5,6,7,8,9,10])
y = np.array([1,20,36,50]) #,62,70,78,84,88,90])

model = LocalLinearRegression()
model.fit(X,y,model.epanechnikov,2)
print(model.coefficients)