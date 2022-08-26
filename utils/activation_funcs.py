import torch
import torch.nn as nn

class SoftShrinkAct(nn.Module):
	"""
	soft thresholding operator with learnable threshold
	soft-thresholding can be expressed as 
	#S_t(x) = ReLU(x-t) - ReLU(-x -t)
	"""
	
	def __init__(self):
		super(SoftShrinkAct, self).__init__()
		
	def forward(self, x, threshold):
		
		return nn.ReLU()(x-threshold) - nn.ReLU()(-x -threshold)


class ApproxSoftShrinkAct(nn.Module):
	
	"""
	smooth approximation of the soft thresholding operator according to
	https://ieeexplore.ieee.org/document/925559
			
	"""
	
	def __init__(self,b=0.001):
		super(ApproxSoftShrinkAct, self).__init__()
		
		self.b = b
	
	def forward(self, x, threshold):
		
		x + torch.tensor(1./2)*( torch.sqrt(torch.pow(x-threshold,2) + self.b) -
										torch.sqrt(torch.pow(x+threshold,2) + self.b)
								)
		return  x