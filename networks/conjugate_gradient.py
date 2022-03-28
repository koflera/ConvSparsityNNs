import torch

def conj_grad(H, x, b, niter=4):
		
	#x is the starting value, b the rhs;
	r = H(x)
	r = b-r
	
	#initialize p
	p = r.clone()
	
	#old squared norm of residual
	sqnorm_r_old = torch.dot(r.flatten(),r.flatten())
	
	for kiter in range(niter):
	
		#calculate Hp;
		d = H(p);

		#calculate step size alpha;
		inner_p_d = torch.dot(p.flatten(),d.flatten())
		alpha = sqnorm_r_old / inner_p_d

		#perform step and calculate new residual;
		x = torch.add(x,p,alpha= alpha.item())
		r = torch.add(r,d,alpha= -alpha.item())
		
		#new residual norm
		sqnorm_r_new = torch.dot(r.flatten(),r.flatten())
		
		#calculate beta and update the norm;
		beta = sqnorm_r_new / sqnorm_r_old
		sqnorm_r_old = sqnorm_r_new

		p = torch.add(r,p,alpha=beta.item())

	return x