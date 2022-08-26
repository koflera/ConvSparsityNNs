
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


import sys
sys.path.append('../')

from utils.activation_funcs import SoftShrinkAct

from utils.conjugate_gradient import conj_grad
from utils.padding_funcs import pad_circular_nd

from utils.cplx_helper_funcs import cplx_conj,  cplx_pw_mult, cplx_pw_div

from utils.grad_kernel_fucns import create_grad_kernel


class ConvDicoLearningCNN(nn.Module):
	
	"""
		A CNN which resembles an ADMM method for solving the following problem:
			
			min_{x, s_k, u_k} ||Ax-y||_2^2 + \lambda / 2 * (||x - \sum_k d_k \conv s_k||_2^2 
									   + alpha * ||u_k||_1 + beta/2 * sum_k ||u_k - s_k||_2^2),
			
		
		The network alternates between the three following subproblems:
		The dictionary atoms are assumed to be fixed in the formulation and 
		thus can be learned by end-to-end training. 
			subproblem 1: update the sparse codes (solve a linear system in FFT domain)
			subproblem 2: update auxiliary variables (soft-thresholding)
			subproblem 3: update image x (solve linear system)
			
		"""

	def __init__(self, 
			    EncObj, 
				nu=4, 
				npcg=2, 
				n_filters=16, 
				kernel_size=[5, 5, 5],
				ndims = 3, #either 2 or 3 for 2D or 3D, respectively
				low_pass = False):

		super(ConvDicoLearningCNN, self).__init__()
		
		self.EncObj  = EncObj
		self.im_size = EncObj.im_size
		self.nu = nu
		self.npcg = npcg
		self.n_filters = n_filters
		self.ndims = ndims #has to coincide with len(kernel_size)
		self.n_filters_half = np.int(n_filters / 2 )
		self.npad = tuple([np.int( np.floor(kernel_size[k]/2)) for k in range(self.ndims)])
		
		self.low_pass = low_pass
		
		filter_shape = (self.n_filters_half, 1, ) + tuple([kernel_size[k] for k in range(ndims)])
		filter_init = 0.001*torch.randn(filter_shape)
		
		self.d_filter_half = nn.Parameter(filter_init, requires_grad= True)
		
		#SoftPlus for constraining the regularization parameters to be strictly positve
		beta,threshold=1, 20 #default values
		self.SoftPlus = nn.Softplus(beta=beta,threshold=threshold)
		
		#regularization parameters (educated guesses)
		lambda_reg = nn.Parameter(0.2 * torch.ones(1), requires_grad=True)
		alpha_reg = nn.Parameter(-1.8 * torch.ones(1), requires_grad=True)
		beta_reg = nn.Parameter(-.1* torch.ones(1), requires_grad=True)
		
		self.lambda_reg = lambda_reg 
		self.alpha_reg = alpha_reg 
		self.beta_reg = beta_reg 
		
		#param of the low-pass
		delta_reg = nn.Parameter(1. * torch.ones(1), requires_grad=True)
		self.delta_reg = delta_reg 
		
		#the Thresholding function; 
		self.ThresholdingFct = SoftShrinkAct()
		
		"""
		everything needed for the spectral operations of the filters
		"""
		
		self.n_groups  =  self.d_filter_half.shape[1] 
		kernel_size = tuple(self.d_filter_half.shape[2:])
		print(kernel_size)
		
		#create paddind and wrapping
		self.padded_size = tuple([self.im_size[k] + kernel_size[k] -1 for k in range(self.ndims)])
		self.fpad = tuple([(self.padded_size[k] - self.im_size[k])// 2 for k in range(self.ndims)])
		self.kernel_center = tuple([kernel_size[k] // 2 for k in range(self.ndims)])
		
		print(self.fpad)
		
		#create meshgrid
		if self.ndims==2:
			self.grid = torch.meshgrid(torch.arange(kernel_size[0]), 
												torch.arange(kernel_size[1])
												)
			
		if self.ndims==3:
			self.grid = torch.meshgrid(torch.arange(kernel_size[0]), 
												torch.arange(kernel_size[1]), 
												torch.arange(kernel_size[2])
												)
			
		self.rot_grid = tuple([self.grid[k] - self.kernel_center[k] % self.padded_size[k] for k in range(self.ndims)])
		
		#gradient kernel (for optional low-pass pre-processing step)
		self.grad_kernel = create_grad_kernel(ndims)
		
		#define the conv operations, and dimensions to pad
		if self.ndims == 2:
			self.conv_op_transposed = F.conv_transpose2d
			self.conv_op = F.conv2d
			self.pad_dims = [2,3]
			
		elif self.ndims == 3:
			self.conv_op_transposed = F.conv_transpose3d
			self.conv_op = F.conv3d
			self.pad_dims = [2,3,4]
			
	def apply_Dhat(self, shat, F_kernel):
		
		"""
		application of the dictionary in the FFT domain.
		shat represents the Fourier-transformed sparse codes. 		
		"""
		
		#shat.shape (mb,n_filters,Nx,Ny,Nt, 2)
		Dshat = cplx_pw_mult(shat, F_kernel.transpose(0,1))
		
		#the first n_filters/2 correspond to the real part, the others to the
		#imag part, sum up accordingly 
		Dshat = torch.stack([torch.sum(Dshat[:,:self.n_filters_half,...],dim=1),
							   torch.sum(Dshat[:,self.n_filters_half:,...],dim=1)],
					  dim=1)
		
		#output.shape (mb,2,Nxp,Nyp,Ntp,2)
		return Dshat

	def apply_DhatHerm(self, Dshat, F_kernel):
		
		"""
		hermitian of the operator Dhat; needed to construct the rhs
		for sub-problem 1.
		"""
		
		#complex-conjugate of fourier-transformed kernel
		F_kernel_conj_transposed = cplx_conj(F_kernel).transpose(0,1) # * torch.tensor([1,-1]).to(device)
		
		#separate first and second half
		Dshat_first_half = Dshat[:,0, ...]
		Dshat_second_half = Dshat[:,1, ...]
		
		if self.ndims == 2: 
			Dshat_first_half = Dshat_first_half.unsqueeze(1)
			Dshat_second_half = Dshat_second_half.unsqueeze(1)
			
		DhatHDshat = torch.cat([cplx_pw_mult(Dshat_first_half, F_kernel_conj_transposed[:,:self.n_filters_half, ...] ),
								 cplx_pw_mult(Dshat_second_half, F_kernel_conj_transposed[:,self.n_filters_half:, ...] )],
								dim=1)
			
		#output.shape (mb,n_filters,Nxp,Nyp,Ntp,2)
		return DhatHDshat
	
	def apply_DhatHermDhat(self, shat, F_kernel):
		
		"""
		the composition of the operators Dhat and DhatHerm
		"""
		return self.apply_DhatHerm(self.apply_Dhat(shat, F_kernel), F_kernel)	
	
	def sherman_morrison(self, b, F_kernel):
		
		""" 
		Applying the SherrmanMorrison formula to solve an appropriate linear equation. 
		See equation (51) in  
			"Efficient Algorithms for Convolutional Sparse Representations"
		by B. Wohlberg, IEEE TIP 2016.
		
		"""
		
		device = b.device
		gamma = self.SoftPlus(self.beta_reg) / self.SoftPlus(self.lambda_reg)
	
		#inner product of the filters and rhs b
		F_kernel_conj = cplx_conj(F_kernel)
		dHb = cplx_pw_mult(F_kernel, b)
		
		dHb_first_half = torch.sum(dHb[:,:self.n_filters_half,...],dim=1)
		dHb_second_half = torch.sum(dHb[:,self.n_filters_half:,...],dim=1)
		
		if self.ndims == 2:
			dHb_first_half = dHb_first_half.unsqueeze(1)
			dHb_second_half = dHb_second_half.unsqueeze(1)
		
		#inner product of the filters
		dHd = cplx_pw_mult(F_kernel_conj, F_kernel)
		
		dHd_first_half = torch.sum(dHd[:,:self.n_filters_half,...],dim=1)
		dHd_second_half = torch.sum(dHd[:,self.n_filters_half:,...],dim=1)
		
		#reg parameter
		gammat = torch.tensor([gamma,0]).to(device)
		factor_first_half = cplx_pw_div(dHb_first_half, gammat + dHd_first_half)
		factor_second_half = cplx_pw_div(dHb_second_half, gammat + dHd_second_half)
		
		gamma_inv =  torch.tensor([gamma**(-1),0]).to(device)
		b_factor_a_first_half = b[:,:self.n_filters_half,...] - cplx_pw_mult(factor_first_half, F_kernel_conj[:,:self.n_filters_half,...])
		b_factor_a_second_half = b[:,self.n_filters_half:,...] - cplx_pw_mult(factor_second_half, F_kernel_conj[:,self.n_filters_half:,...])
		
		return cplx_pw_mult(gamma_inv, torch.cat([b_factor_a_first_half, b_factor_a_second_half],dim=1) )
	
	def apply_HOperator(self, x):
		
		"""
		The operator for constructing the system Hx = b for sub-problem 3
		i.e. H = A^# A + \lambda*Id
			 b = xu + \lambda*\sum_k d_k \conv s_k, where xu:=A^# y
		"""
		x = self.EncObj.apply_AdagA_Toeplitz(x.unsqueeze(0)).squeeze(0) +  self.SoftPlus(self.lambda_reg)*x
		return x
	
	def apply_low_pass_op(self, x):
	
		device = x.device
		grad_kernel = torch.cat(2*[self.grad_kernel],dim=0).to(device)
		
		#apply the adjoint and the original finite differences filter
		npads = [1 for kp in range(len(self.pad_dims))]
		GTGx = pad_circular_nd(x, npads, dim=self.pad_dims)
		GTGx = self.conv_op(GTGx, grad_kernel, groups= 2, padding = 1)
		
		if self.ndims == 2:
			GTGx = GTGx[:,:,1:-1,1:-1]
		elif self.ndims == 3:
			GTGx = GTGx[:,:,1:-1,1:-1,1:-1]
		
		GTGx = pad_circular_nd(GTGx, npads, dim=self.pad_dims)
		GTGx = self.conv_op_transposed(GTGx, grad_kernel, groups= 2, padding = 1)
		
		if self.ndims == 2:
			GTGx = GTGx[:,:,1:-1,1:-1]
		elif self.ndims == 3:
			GTGx = GTGx[:,:,1:-1,1:-1,1:-1]
			
		return x + self.SoftPlus(self.delta_reg) * GTGx
	
	def batch_temporal_dim(self, x):
		"""
		stack the different cardiac phases along the time dimension
		--> (mb, 2, Nx, Ny, Nt) --> (mb * Nt, 2, Nx, Ny)
		"""
		mb,nch,Nx,Ny,Nt = x.shape
		x = x.permute(0,4,1,2,3).contiguous().view(mb*Nt,nch,Nx,Ny)
		
		return x
	
	def unbatch_temporal_dim(self, x, mb):
		"""
		stack the different cardiac phases along the time dimension
		--> (mb * Nt, 2, Nx, Ny) --> (mb , Nt, 2, Nx, Ny, Nt)
		"""
		mb_Nt,nch,Nx,Ny = x.shape
		Nt = np.int(mb_Nt / mb) # inferNt
		x = x.view(mb,Nt,nch,Nx,Ny).permute(0,2,3,4,1)
		
		return x
	
	def forward(self,x):
		
		#get shape
		mb,n_ch,Nx,Ny,Nt = x.shape
		device = x.device
		
		# initial NUFFT reconstruction;  # shape (1,2,Nx,Ny,Nt)
		xu = x.clone()
		
		#perform a few steps of iterative-SENSE
		it_SENSE_pre_processing = 1
		if it_SENSE_pre_processing:
			#print('iterative SENSE')
			x = conj_grad(self.EncObj.apply_AdagA_Toeplitz, x.unsqueeze(0), xu.unsqueeze(0), niter=6).squeeze(0)
		
		#define domain size for the conv operations
		if self.ndims == 2:
			conv_dmn_size = tuple([Nx,Ny])
		elif self.ndims == 3:
			conv_dmn_size = tuple([Nx,Ny,Nt])
		
		"""
		prepare convolutional filter and 
		filter in FFT domain
		"""
		#concatenate the filter 
		d = torch.cat(2*[self.d_filter_half],dim=0).to(device)
		
		#create FFT filter
		kernel_padded = torch.zeros((self.n_filters,self.n_groups)+ self.padded_size).to(device)
		for kf in range(self.n_filters):
			kernel_padded[(kf, 0,)+ self.rot_grid] = d[(kf, 0,)+ self.grid]
		
		#Fourier-transformed kernel
		F_kernel = torch.rfft(kernel_padded, signal_ndim=self.ndims, onesided=False)
		
		#auxiliary variable u
		if self.ndims == 2:
			u = 0.001*torch.zeros( (Nt * mb,self.n_filters,) + conv_dmn_size).to(device)
		if self.ndims == 3:
			u = 0.001*torch.zeros( (mb, self.n_filters,) + conv_dmn_size).to(device) #stack all time points in the batch dimension
			
		#initialize auxiliaray variable z
		z = u.clone()
	
		for k in range(self.nu):
			
			"""
			pre-processing step, low-pass filter the image and learn the 
			sparse representation on the low-pass-filtered image.
			we compute the lowpass-filtered imageas the solution of 
			
			min_x_low 1/2||x_low- x||_2^2 + \delta/2 * \sum_i ||G_i x_low ||_2^2,
			
			see https://pdfs.semanticscholar.org/4ec3/62863c7e1391f6892ea1efbe2738cfb35ec9.pdf
			page 3 for a reference.
			
			"""
			
			#batch the time points
			if self.ndims == 2:
				x = self.batch_temporal_dim(x)
				
			if self.low_pass:
					
				#the lowpass filtered image (maybe there is a close-form solution)
				x = conj_grad(self.apply_low_pass_op, x, x, niter=4)
				
				#copy the low-pass component of the image
				xlow = x.clone()
				
				#calculate the residual
				x = x - xlow
				
			#SUB-PROBLEM 1: solve for the sparse codes s_k given u_k and x,
			#prepare x and u by properly padding before going to FFT domain
			fpad = tuple(self.fpad)+tuple(self.fpad)
			
			xp = F.pad(x, fpad, mode='constant', value=0)
			up = F.pad(u, fpad, mode='constant', value=0)
			
			#apply FFT
			xhat = torch.rfft(xp, signal_ndim=self.ndims, onesided=False)
			uhat = torch.rfft(up, signal_ndim=self.ndims, onesided=False)
			
			#solve the system (D shat = c)  with the SM formula
			gamma = self.SoftPlus(self.beta_reg) / self.SoftPlus(self.lambda_reg)
			
			#pad and FFT z
			zp = F.pad(z, fpad, mode='constant', value=0)
			zhat = torch.rfft(zp, signal_ndim=self.ndims, onesided=False)
			c = self.apply_DhatHerm(xhat, F_kernel) + gamma * (uhat + zhat)
			
			if self.ndims == 2:
				permuted_F_kernel = F_kernel.permute(1,0,2,3,4)
			elif self.ndims == 3:
				permuted_F_kernel = F_kernel.permute(1,0,2,3,4,5)
				
			#apply Sherman-Morrison formula
			shat = self.sherman_morrison(c,  permuted_F_kernel) 
			
			#obtain sparse codes by IFFT:
			s = torch.irfft(shat, signal_ndim=self.ndims, onesided=False)
			
			#crop s to original shape
			if self.ndims == 2:
				fpad0, fpad1 = self.fpad
				s = s[:, :, fpad0:-fpad0, fpad1:-fpad1]
			elif self.ndims == 3:
				fpad0, fpad1, fpad2 = self.fpad
				s = s[:, :, fpad0:-fpad0, fpad1:-fpad1, fpad2:-fpad2]
			
			#SUB-PROBLEM 2: solve for aux variable u_k given s_k and x
			#solution given by soft-thresholding:
			u =  self.ThresholdingFct(z - s ,self.SoftPlus(self.alpha_reg)/self.SoftPlus(self.beta_reg))
				
			#update ADMM aux variable
			z = z + u - s
			
			#SUB-PROBLEM 3: solve for image x given s_k and u_k
			#pad s to ensure circular boundary conditions
			s = pad_circular_nd(s, self.npad, dim=self.pad_dims)
		
			#apply the filter d to the sparse codes s
			ds = self.conv_op_transposed(s, d, groups= 2, padding = self.npad)
				
			#crop again
			if self.ndims == 2:
				npad0, npad1 = self.npad
				ds = ds[:,:,npad0:-npad0,npad1:-npad1] 
				s = s[:,:,npad0:-npad0,npad1:-npad1] 
				
			elif self.ndims == 3:
				npad0, npad1, npad2 = self.npad
				ds = ds[:,:,npad0:-npad0,npad1:-npad1,npad2:-npad2] 
				s = s[:,:,npad0:-npad0,npad1:-npad1,npad2:-npad2]
			
			if self.low_pass:
				#add the low-pass component after the sparse approximation
				ds = ds + xlow
				x = x + xlow
			
			if self.ndims == 2:
				ds = self.unbatch_temporal_dim(ds, mb)
				x = self.unbatch_temporal_dim(x, mb)

			#las sub-problem: solve for image x; 
			#construct rhs of system
			rhs = xu + self.SoftPlus(self.lambda_reg) * ds
			
			#solve sysstem with CG
			x = conj_grad(self.apply_HOperator, x, rhs, niter=self.npcg)
		
		return x
		