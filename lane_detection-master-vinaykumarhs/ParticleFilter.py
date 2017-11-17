'''
Class: ParticleFilter 
implements simple particle filter algorithm.
Author: Vinay
me@vany.in

'''

import numpy as np
import scipy
import scipy.stats
from numpy.random import uniform,randn
from numpy.linalg import norm

from filterpy.monte_carlo import systematic_resample




class ParticleFilter:
	def __init__(self,N,x_range,sensor_err,par_std):
		self.N = N
		self.x_range = x_range
		self.create_uniform_particles()
		self.weights = np.zeros(N)
		self.u = 0.00
		self.initial_pose = 0
		self.sensor_std_err = sensor_err
		self.particle_std = par_std

	def create_uniform_particles(self):
	    self.particles = np.empty((self.N, 1))
	    self.particles[:, 0] = uniform(self.x_range[0], self.x_range[1], size=self.N)
	    return self.particles


	def predict(self,particles, std, u, dt=1.):
	    self.N = len(particles)
	    self.particles[:,0] += u + (randn(self.N)*std)


	def update(self,particles, weights, z, R, init_var):
	    self.weights.fill(1.)

	    self.distance = np.linalg.norm(self.particles[:, 0:1] - init_var, axis=1)
	    self.weights *= scipy.stats.norm(self.distance, R).pdf(z)

	    self.weights += 1.e-300      # avoid round-off to zero
	    self.weights /= sum(self.weights) # normalize


	def estimate(self,particles, weights):
	    """returns mean and variance of the weighted particles"""
	    self.pos = self.particles[:, 0:1]
	    self.mean = np.average(self.pos, weights=self.weights, axis=0)
	    self.var  = np.average((self.pos - self.mean)**2, weights=self.weights, axis=0)
	    return self.mean, self.var


	def neff(self,weights):
	    return 1. / np.sum(np.square(self.weights)+1.e-300) #handle zero round-off


	def resample_from_index(self,particles, weights, indexes):
	    self.particles[:] = self.particles[self.indexes]
	    self.weights[:] = self.weights[self.indexes]
	    self.weights /= np.sum(self.weights)
	
	def filterdata(self, data):
	    self.predict(self.particles, u=self.u, std=self.particle_std)
	    self.update(self.particles, self.weights, z=data, R=self.sensor_std_err, init_var=self.initial_pose)
	    if self.neff(self.weights) < self.N/2: #Perform systematic resampling.
	        self.indexes = systematic_resample(self.weights)
	        self.resample_from_index(self.particles, self.weights, self.indexes)
	    mu, _ = self.estimate(self.particles, self.weights)
	    return mu


if __name__ == '__main__':
	print "ParticlFilter class implementation"
	xl_int_pf=ParticleFilter(N=10000,x_range=(0,800),ses_err=1,par_std=100)
