import numpy as np 
from scipy import stats 
from math import sqrt
from copy import copy 
import matplotlib.pyplot as plt 

class HmmGmm:
	def __init__(self,observed):
		self.observed = observed
		self.T = len(observed)

	def fit(
		self,
		mean0,
		mean1,
		sd0,
		sd1,
		t00,
		t11,
		samples = 1000,
		burn_in = 100,
		resample = 1,
		mean_certainty0 = 10.0,
		mean_certainty1 = 10.0,
		prec_certainty0 = 10.0,
		prec_certainty1 = 10.0,
		t00_alpha = 15,
		t00_beta = 1,
		t11_alpha = 15,
		t11_beta = 1):
		'''
		Fits a 2 state hidden markov model where the states dictate 
		which normal distribution is picked to generate the data
		'''

		''' INIT PARAMS '''
		self.hidden_states = np.zeros(shape=self.T)
		self.t00 = 0
		self.t11 = 0
		self.mean0 = 0
		self.sd0 = 0
		self.mean1 = 0
		self.sd1 = 0

		for k in range(resample):
			if (resample > 1):
				print("Sample =", k)
			self.hidden = np.random.randint(2, size=self.T)
			prec0 = 1.0 / pow(sd0,2)
			prec1 = 1.0 / pow(sd1,2)

			## prior dist of mean0
			mean0_prior = copy(mean0)
			mean0_precision_prior = 1.0 / pow(sd0,2)
			# mean_certainty0
			## prior dist of mean1
			mean1_prior = copy(mean1)
			mean1_precision_prior = 1.0 / pow(sd1,2)
			# mean_certainty1
			## prior dist of precision0
			alpha0_prior = copy(prec_certainty0)
			beta0_prior = alpha0_prior / mean0_precision_prior
			## prior dist of precision1
			alpha1_prior = copy(prec_certainty1)
			beta1_prior = alpha1_prior / mean1_precision_prior
			## prior dist transition 0->0
			# t00_alpha, t00_beta
			## prior dist transition 1->1
			# t11_alpha, t11_beta
			

			''' BURN IN '''
			print("Burning In")
			for i in range(burn_in):

				### SAMPLE HIDDEN STATES
				self.hidden[0] = self.sampleHiddenState(self.observed[0],-1,self.hidden[1],mean0,mean1,prec0,prec1,t00,t11)
				for t in range(1,self.T-1):
					self.hidden[t] = self.sampleHiddenState(self.observed[t],self.hidden[t-1],self.hidden[t+1],mean0,mean1,prec0,prec1,t00,t11)
				self.hidden[self.T-1] = self.sampleHiddenState(self.observed[self.T-1],self.hidden[self.T-2],-1,mean0,mean1,prec0,prec1,t00,t11)

				## some statistics of the states:
				n0 = self._getStateSamples(0)
				n1 = self._getStateSamples(1)
				mean_state0 = self._getMeanGivenState(0)
				mean_state1 = self._getMeanGivenState(1)
				var_state0 = self._getVarGivenState(0)
				var_state1 = self._getVarGivenState(1)
				transitions00 = self._getStateTransitionSum(0)
				transitions11 = self._getStateTransitionSum(1)

				### SAMPLE TRANSITION PROBABILITIES
				t00 = self.sampleTransition(t00_alpha,t00_beta,n0,transitions00,self.hidden[-1]==0)
				t11 = self.sampleTransition(t11_alpha,t11_beta,n1,transitions11,self.hidden[-1]==1)

				### SAMPLE NORMAL 0
				mean0 = self.sampleMean(mean_state0,n0,mean0_prior,mean0_precision_prior,mean_certainty0)
				prec0 = self.samplePrecision(alpha0_prior,beta0_prior,n0,var_state0,mean_state0,mean0_prior,mean_certainty0)

				### SAMPLE NORMAL 1
				mean1 = self.sampleMean(mean_state1,n1,mean1_prior,mean1_precision_prior,mean_certainty1)
				prec1 = self.samplePrecision(alpha1_prior,beta1_prior,n1,var_state1,mean_state1,mean1_prior,mean_certainty1)


			''' SAMPLE '''
			mean0_arr = np.zeros(shape=samples)
			sd0_arr = np.zeros(shape=samples)

			mean1_arr = np.zeros(shape=samples)
			sd1_arr = np.zeros(shape=samples)

			t00_arr = np.zeros(shape=samples)
			t11_arr = np.zeros(shape=samples)

			hidden_arr = np.zeros(shape=(samples,self.T))

			print("Sampling")
			for i in range(samples):

				### SAMPLE HIDDEN STATES
				self.hidden[0] = self.sampleHiddenState(self.observed[0],-1,self.hidden[1],mean0,mean1,prec0,prec1,t00,t11)
				for t in range(1,self.T-1):
					self.hidden[t] = self.sampleHiddenState(self.observed[t],self.hidden[t-1],self.hidden[t+1],mean0,mean1,prec0,prec1,t00,t11)
				self.hidden[self.T-1] = self.sampleHiddenState(self.observed[self.T-1],self.hidden[self.T-2],-1,mean0,mean1,prec0,prec1,t00,t11)

				hidden_arr[i,:] = self.hidden[:]

				## some statistics of the states:
				n0 = self._getStateSamples(0)
				n1 = self._getStateSamples(1)
				mean_state0 = self._getMeanGivenState(0)
				mean_state1 = self._getMeanGivenState(1)
				var_state0 = self._getVarGivenState(0)
				var_state1 = self._getVarGivenState(1)
				transitions00 = self._getStateTransitionSum(0)
				transitions11 = self._getStateTransitionSum(1)

				### SAMPLE TRANSITION PROBABILITIES
				t00 = self.sampleTransition(t00_alpha,t00_beta,n0,transitions00,self.hidden[-1]==0)
				t11 = self.sampleTransition(t11_alpha,t11_beta,n1,transitions11,self.hidden[-1]==1)

				t00_arr[i] = t00
				t11_arr[i] = t11

				### SAMPLE NORMAL 0
				mean0 = self.sampleMean(mean_state0,n0,mean0_prior,mean0_precision_prior,mean_certainty0)
				prec0 = self.samplePrecision(alpha0_prior,beta0_prior,n0,var_state0,mean_state0,mean0_prior,mean_certainty0)

				mean0_arr[i] = mean0
				sd0_arr[i] = 1.0/sqrt(prec0)

				### SAMPLE NORMAL 1
				mean1 = self.sampleMean(mean_state1,n1,mean1_prior,mean1_precision_prior,mean_certainty1)
				prec1 = self.samplePrecision(alpha1_prior,beta1_prior,n1,var_state1,mean_state1,mean1_prior,mean_certainty1)

				mean1_arr[i] = mean1
				sd1_arr[i] = 1.0/sqrt(prec1)


			''' SUMMARIZE DATA '''
			self.hidden_states = self.hidden_states + hidden_arr.mean(axis=0)
			self.t00 += t00_arr.mean()
			self.t11 += t11_arr.mean()
			self.mean0 += mean0_arr.mean()
			self.sd0 += sd0_arr.mean()
			self.mean1 += mean1_arr.mean()
			self.sd1 += sd1_arr.mean()

		self.hidden_states = self.hidden_states / resample
		self.t00 /= resample
		self.t11 /= resample
		self.mean0 /= resample
		self.sd0 /= resample
		self.mean1 /= resample
		self.sd1 /= resample


	def addDataPoint(self,x):
		p0g0 = self.pState0(
			x,
			0,
			-1,
			self.mean0,
			self.mean1,
			1.0/pow(self.sd0,2),
			1.0/pow(self.sd1,2),
			self.t00,
			self.t11)
		p0g1 = self.pState0(
			x,
			1,
			-1,
			self.mean0,
			self.mean1,
			1.0/pow(self.sd0,2),
			1.0/pow(self.sd1,2),
			self.t00,
			self.t11)

		p0 = (p0g0*(1-self.hidden_states[-1])) + (p0g1*(self.hidden_states[-1]))
		self.hidden_states = np.append(self.hidden_states,1-p0)
		return 1-p0



	def sampleHiddenState(self,x,prev_state,next_state,mean0,mean1,prec0,prec1,t00,t11):
		p0 = self.pState0(x,prev_state,next_state,mean0,mean1,prec0,prec1,t00,t11)
		u = np.random.uniform()
		if (u <= p0):
			return 0
		else:
			return 1
			

	@staticmethod
	def sampleMean(x_state_mean,n_state,mean_prior,prec_prior,mean_certainty):
		new_mean = ((mean_certainty * mean_prior) + (n_state * x_state_mean)) / (mean_certainty + n_state)
		new_prec = (mean_certainty + n_state) * prec_prior 
		return np.random.normal(new_mean,1.0/sqrt(new_prec))

	@staticmethod
	def samplePrecision(alpha_prior,beta_prior,n_state,x_state_var,x_state_mean,mean_prior,mean_certainty):
		new_alpha = alpha_prior + (n_state / 2.0)

		beta_mid = 0.5 * n_state * x_state_var
		beta_end = (mean_certainty * n_state * pow((x_state_mean - mean_prior),2)) / (2 * (mean_certainty + n_state))
		new_beta = beta_prior + beta_mid + beta_end

		return np.random.gamma(new_alpha,1.0/new_beta)

	@staticmethod
	def sampleTransition(alpha_prior,beta_prior,n_state,same_to_same_sum,last_is_state=False):
		if last_is_state:
			# if last hidden state is the state we're counting
			# then do not count it in n-state b/c it doesn't 
			# have a chance to transition
			return np.random.beta(alpha_prior + same_to_same_sum, beta_prior + (n_state - 1) - same_to_same_sum)
		else:
			return np.random.beta(alpha_prior + same_to_same_sum, beta_prior + n_state - same_to_same_sum)

	def _getStateSamples(self,state):
		return (self.hidden == state).sum()

	def _getMeanGivenState(self,state):
		return self.observed[self.hidden == state].mean()

	def _getVarGivenState(self,state):
		return self.observed[self.hidden == state].var()

	def _getStateTransitionSum(self,state):
		past_state = -1
		s = 0
		for t in range(self.T):
			# if go state->state => count
			if (past_state == state):
				if (self.hidden[t] == state):
					s += 1
			past_state = self.hidden[t]
		return s

	@staticmethod
	def pState0(x,prev_state,next_state,mean0,mean1,prec0,prec1,t00,t11):
		def pGivenOtherState(prior_state,next_state):
			# pSame = T(prior state, same as prior state)
			if ((prior_state == 0) and (next_state == 0)):
				return t00
			elif ((prior_state == 0) and (next_state == 1)):
				return 1 - t00
			elif ((prior_state == 1) and (next_state == 0)):
				return 1 - t11
			elif ((prior_state == 1) and (next_state == 1)):
				return t11
			else:
				raise ValueError("Prior State and Next State must equal 0 or 1.")

		p0 = stats.norm.pdf(x,mean0,1.0/sqrt(prec0))
		p1 = stats.norm.pdf(x,mean1,1.0/sqrt(prec1))

		if (prev_state == -1):
			# first state -> only look at prob given next state
			p0 *= pGivenOtherState(0,next_state)
			p1 *= pGivenOtherState(1,next_state)

		elif (next_state == -1):
			# last state -> only look at prob given prior state
			p0 *= pGivenOtherState(prev_state,0)
			p1 *= pGivenOtherState(prev_state,1)

		else:
			p0 *= pGivenOtherState(prev_state,0) * pGivenOtherState(0,next_state)
			p1 *= pGivenOtherState(prev_state,1) * pGivenOtherState(1,next_state)

		pTotal = p0 + p1
		p0 /= pTotal
		p1 /= pTotal

		if ((p0 + p1) < 0.95) or ((p0 + p1) > 1.05):
			raise ValueError("For some reason p0 and p1 do not sum close to 1.")

		return p0

## TEST
t00_hat = 0.97
t11_hat = 0.94
mean0_hat = 0.5
sd0_hat = 1.0
mean1_hat = -0.75
sd1_hat = 2.0

states = np.zeros(shape=500)
states[0] = 0
for i in range(1,500):
	u = np.random.uniform()
	if states[i-1] == 0:
		if u <= t00_hat:
			states[i] = 0
		else:
			states[i] = 1
	elif states[i-1] == 1:
		if u <= t11_hat:
			states[i] = 1
		else:
			states[i] = 0

data = np.zeros(shape=500)
for i in range(500):
	if (states[i] == 0):
		data[i] = np.random.normal(mean0_hat,sd0_hat)
	elif (states[i] == 1):
		data[i] = np.random.normal(mean1_hat,sd1_hat)

new_states = np.zeros(shape=250)
new_states[0] = 0
for i in range(1,250):
	u = np.random.uniform()
	if new_states[i-1] == 0:
		if u <= t00_hat:
			new_states[i] = 0
		else:
			new_states[i] = 1
	elif new_states[i-1] == 1:
		if u <= t11_hat:
			new_states[i] = 1
		else:
			new_states[i] = 0

new_data = np.zeros(shape=250)
for i in range(250):
	if (new_states[i] == 0):
		new_data[i] = np.random.normal(mean0_hat,sd0_hat)
	elif (new_states[i] == 1):
		new_data[i] = np.random.normal(mean1_hat,sd1_hat)


model = HmmGmm(data)
mean = data.mean()
sd = data.std()
model.fit(
	mean0 = mean+(0.5*sd),
	mean1 = mean-(0.5*sd),
	sd0 = sd*0.8,
	sd1 = sd*1.25,
	t00 = 0.95,
	t11 = 0.95,
	burn_in = 100,
	samples = 500,
	resample = 2)
for i in range(len(new_data)):
	model.addDataPoint(new_data[i])

print("Normal 0: N(", model.mean0,",",model.sd0,")")
print("Normal 1: N(", model.mean1,",",model.sd1,")")
print("T: 0->0:",model.t00)
print("T: 1->1:",model.t11)

fig, (ax1,ax2) = plt.subplots(2,1,sharex=True)
ax1.plot(np.arange(750),model.hidden_states,c='r')
ax2.plot(np.arange(750),np.append(states,new_states),c='b')
plt.show()



