import numpy as np 

class DMD:
	def fit(self,X,r=None,threshold=0.9,dt=1):
		'''
		Fits the model to the observed data X

		INPUTS
		X : Data that's measured through time 
			measurements (Dimension) x time 
			np.array
		r : dimension for rank reduction (if None -> class finds the ideal r)
			int 
		threshold : the threshold that has to be covered by the first r eigenvalues when finding r
			float between 0 and 1
		dt : change in time between each column in X 
			float
		'''
		Xbefore = np.asmatrix(X[:,:-1])
		Xafter = np.asmatrix(X[:,1:])
		self.dt = dt
		if not r:
			r = self.findr(X,threshold)

		# 1. SVD and rank reduce
		U,S,V = np.linalg.svd(Xbefore,full_matrices=False)
		self.r = min(r, U.shape[1])
		V = V.H
		Ur = U[:,:self.r]
		Sr = np.diag(S[:self.r])
		Vr = V[:,:self.r]

		# 2. Estimate the projection of A onto the POD modes
		Sr_inv = np.linalg.inv(Sr)
		Atilde = np.matmul(Ur.H, np.matmul(Xafter, np.matmul(Vr, Sr_inv)))

		# 3. Eigendecomposition of Atilde 
		PODeigenvalues,PODeigenvectors = np.linalg.eig(Atilde)
		PODeigenvectors = np.asmatrix(PODeigenvectors)

		# 4. Reconstuct the eigendecomposition of A from the eigenvalues and vectors
		self.eigenvectors = np.matmul(Xafter, np.matmul(Vr, np.matmul(Sr_inv, PODeigenvectors)))
		self.eigenvalues = PODeigenvalues
		self.init,_,_,_ = np.linalg.lstsq(self.eigenvectors,np.array(Xbefore[:,-1]).reshape(-1))

	def predict(self,t):
		'''
		Predicts x at time t

		INPUTS
		t : time into the future you want to predict
			float

		OUTPUTS
		x_t : prediction of x at time t
			numpy array
		'''
		t += 1;
		x_t = 0
		for i in range(self.r):
			x_t += self.init[i] * (self.eigenvalues[i]**(t/self.dt)) * self.eigenvectors[:,i]

		return x_t

	@staticmethod
	def findr(X,threshold=0.9):
		''' 
		Finds the ideal r value to rank reduce the data but maintain its structure

		INPUTS
		X : Data that's measured through time 
			measurements (Dimension) x time 
			np.array
		threshold : the threshold that has to be covered by the first r eigenvalues
			float between 0 and 1

		OUTPUTS
		r : where the input data should be rank reduced to
		'''
		U,S,V = np.linalg.svd(X,full_matrices=False)
		normalized_S = S / np.sum(S)
		cusum_norm_S = np.cumsum(normalized_S)
		return np.argmax(cusum_norm_S >= threshold) + 1



model = DMD()
X = np.transpose(np.array([[1,2,3,0.5,20],[6,4,8.5,1.5,14],[18.5,8,22.5,4.5,-4.5],[49,16,57.5,13.5,-53.5]]))
# dx1/dt = x2 + x3
# dx2/dt = x2
# dx3/dt = x2 + x3 + x4
# dx4/dt = 2*x4
# dx5/dt = -x1-x2-x3
model.fit(X,threshold=0.8,dt=1.0)
print(model.predict(2))

