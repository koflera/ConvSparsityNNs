import numpy as np
import torch

def create_grad_kernel(ndims):
	
	"""
	function for creating finite-differences kernels
	"""
	
	if ndims == 1:
		dx = np.zeros((1,3))
		
		symm_diff = np.array([1,0,-1])

		dx = symm_diff
		
		filters_list = [dx]
		grad_kernel = torch.zeros(1,1,3)
		for kf in range(ndims):
			h = torch.tensor(filters_list[kf])
			grad_kernel[kf,0,...] = h
			
	elif ndims == 2:
		dx = np.zeros((3,3))
		dy = np.zeros((3,3))

		symm_diff = np.array([1,0,-1])

		dx[1,:] = symm_diff
		dy[:,1] = symm_diff

		filters_list = [dx,dy]
		grad_kernel = torch.zeros(2,1,3,3)
		for kf in range(ndims):
			h = torch.tensor(filters_list[kf])
			grad_kernel[kf,0,...] = h
		
	if ndims == 3:
		dx = np.zeros((3,3,3))
		dy = np.zeros((3,3,3))
		dt = np.zeros((3,3,3))

		symm_diff = np.array([1,0,-1])

		dx[1,:,1] = symm_diff
		dy[:,1,1] = symm_diff
		dt[1,1,:] = symm_diff

		filters_list = [dx,dy,dt]
		
		grad_kernel = torch.zeros(3,1,3,3,3)
		for kf in range(ndims):

			h = torch.tensor(filters_list[kf])
			grad_kernel[kf,0,...] = h

	return grad_kernel