
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


from .activation_funcs import ApproxSoftShrinkAct

from .conjugate_gradient import conj_grad
from .padding_funcs import pad_circular_nd

class ConvAnaOpLearningCNN(nn.Module):

	def __init__(self, 
			    EncObj, 
				nu=4, 
				npcg=4, 
				n_filters=8, 
				kernel_size=5,
				version='version1'):

		"""
		A CNN which resembles an  method for solving the following problem:
			
			min_{x, s_k} ||Ax-y||_2^2 + \lambda/ 2* sum_k (||h_k x - s_k||_2^2)  + \alpha * \sum_k ||s_k||_1.
			
		
		One block consists of a soft/hard-thresholding operator (which yields s_k)
		The other block consists of a CG-module (which yields x)
		
		The filters can be learned by supervised end-to-end training.
		
		Further, there are two possible alternatives for the network which
		differ in the way x is updated, i.e. which system is solved:
			
			version 1:
				update x_{j+1} by solving
				Hx = b with H = A^# A + \beta * \Id,
							b = A^h + \beta * z_j
				where z_j = \sum_k h_k^T \soft_{\alpha/\lambda} h_k x_j
			
			version 2:
				update x_{j+1} by solving
				Hx = b with H = A^# A + \beta * \sum_k h_k^T \soft_{\alpha/\lambda} h_k,
							b = A^h + \beta * \sum_k h_k^T z_j
				where z_j =  x_j
		
		"""
		
		super(ConvAnaOpLearningCNN, self).__init__()
		
		self.EncObj  = EncObj
		
		#overall number of iterations  and iterations of CG
		self.nu = nu
		self.npcg = npcg

		#filter parameters
		self.n_filters=n_filters
		self.n_filters_half = np.int(self.n_filters/2)
		self.npad = np.int( np.floor(kernel_size/2))
		
		filter_shape = [np.int(self.n_filters/2), 1, kernel_size, kernel_size, kernel_size]
		filter_init  = torch.rand(filter_shape,dtype=torch.float,requires_grad=True)
		self.h_filter_half =  nn.Parameter(filter_init,requires_grad=True)
		
		#SoftPlus for constraining the regularization parameters to be strictly positve
		beta,threshold=1,20 #default values
		self.SoftPlus = nn.Softplus(beta=beta,threshold=threshold)
		
		#regularization parameters
		lambda_reg_init = -1.7133
		alpha_reg_init = -3.1488 
		self.lambda_reg = nn.Parameter( lambda_reg_init * torch.ones(1))
		self.alpha_reg = nn.Parameter( alpha_reg_init * torch.ones(1))
				
		b=0.001
		self.ThresholdingFct = ApproxSoftShrinkAct(b)
		
		#determines how the NN updates are defined
		self.version = version
		
	def apply_h(self, x):
		
		#concatenate filters
		h_filter = torch.cat(2*[self.h_filter_half],dim=0)
		
		hx = F.conv3d(pad_circular_nd(x, self.npad, dim=[2,3,4]), 
					h_filter, 
					bias=None,
					padding=self.npad, 
					groups=2,
					)
		
		hx = hx[:,:, self.npad:-self.npad, self.npad:-self.npad, self.npad:-self.npad]
		
		return hx
	
	def apply_hT(self, z):
		
		#concatenate filters
		h_filter = torch.cat(2*[self.h_filter_half],dim=0)
		
		hTz = F.conv_transpose3d(pad_circular_nd(z, self.npad, dim=[2,3,4]), 
					h_filter, 
					bias=None,
					padding=self.npad, 
					groups=2,
					)
		
		hTz = hTz[:,:, self.npad:-self.npad, self.npad:-self.npad, self.npad:-self.npad]
		
		return hTz
	
	def apply_hTh(self,x):
		
		return self.apply_hT(self.apply_h(x))
		
		
	def HOperator(self, x):
		
		if self.version == 'version1':
			
			return self.EncObj.apply_AdagA_Toeplitz(x.unsqueeze(0)).squeeze(0) + self.SoftPlus(self.lambda_reg) * x
		
		elif self.version == 'version2':
			
			return self.EncObj.apply_AdagA_Toeplitz(x.unsqueeze(0)).squeeze(0) + self.SoftPlus(self.lambda_reg) * self.apply_hTh(x)
	
	def forward(self,x):
		
		
		# initial NUFFT reconstruction; 
		# shape (1,2,Nx,Ny,Nt)
		xu = x.clone()
		
		for k in range(self.nu):
			
			#apply the filters h
			hx = self.apply_h(x)
			
			#soft-threshold
			threshold = self.SoftPlus(self.alpha_reg)/self.SoftPlus(self.lambda_reg)
			soft_hx =  self.ThresholdingFct(hx, threshold)
			
			#apply h^T
			hT_soft_hx = self.apply_hT(soft_hx)
			
 			#update x solving a system with the CG-module
 			#create rhs
			rhs = xu + self.SoftPlus(self.lambda_reg) * hT_soft_hx
			
 			# perform CG
			if self.npcg!=0:

				# shape (1,1,2,Nx,Ny,Nt)
				x = conj_grad(self.HOperator, x, rhs, niter=self.npcg)

		return x

			
def caol_unit_norm_projector(model):
	
	"""
	function for projecting the filters on the unit sphere. 
	
	"""
	with torch.no_grad():
		
		n_filters_half=model.h_filter_half.shape[0]
		for kf in range(n_filters_half):
			model.h_filter_half[kf,...].div_(torch.norm(model.h_filter_half[kf,...].flatten(), p=2, keepdim=True))
			