"""
auxiliary functions for complex-valued tensors stored in the two-channels
format, where the last dimension is used for the real and the imaginary parts.
 
"""

import torch
import numpy as np

def cplx_conj(t):
	"""
	complex conjugate of the tensor t
	"""
	return t  * torch.tensor([1,-1]).to(t.device)
	
def cplx_pw_mult(t1, t2):
	"""
	point-wise complex multiplication of t1 and t2
	"""
	
	#real and imag parts
	res_real = t1[...,0] * t2[...,0] - t1[...,1] * t2[...,1]
	res_imag = t1[...,0] * t2[...,1] + t1[...,1] * t2[...,0]
	
	return torch.stack([res_real, res_imag], dim= -1)

def cplx_pw_div(t1, t2):
	
	"""
	point-wise complex division of t1 and t2
	"""

	t2conj = cplx_conj(t2)
	factor = torch.stack([1./(t2[...,0]**2+t2[...,1]**2),1./(t2[...,0]**2+t2[...,1]**2)],dim=-1)
	
	return factor*cplx_pw_mult(t1,t2conj)

def cplx_np2torch(x,dim):
	"""
	functon for converting a complex-valued np.array x
	to a complex-valued torch-tensor, where the 2 channels 
	for the real and imaginary parts are inserted as "dim" dimension
	"""
	
	x = torch.stack([torch.tensor(np.real(x)),torch.tensor(np.imag(x))],dim=dim)
	
	return x
	
def cplx_torch2np(x,dim):
	
	"""
	functon for a complex-valued torch-tensors to a complex-valued numpy array
	the parameter "dim" indicates which dimension is used to stre the real 
	and the imaginary part in the torch-tensor
	
	first, the tensor is transposed, such that we can access te real  and the imaginry parts
	
	the output is a numpy array where the dimension "dim" is dropped
	"""
		
	#permutes the axis "dim" and 0
	#now, the 0-th axis contains the real and imainary parts
	x = torch.transpose(x,dim,0) 
	
	#get the real and imaginry parts
	xr = x[0,...].numpy()
	xi = x[1,...].numpy()

	x = xr+1j*xi
	
	#expand dimensions in order to be able to get back to original shape
	x = np.expand_dims(x,axis=0)
	
	x = np.swapaxes(x,0,dim)
	
	#drop the dimension "dim"
	x = np.squeeze(x,axis=dim)
	
	return x