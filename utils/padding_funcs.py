import torch

def pad_circular_nd(x, npad, dim):
	
	"""
	function for circular padding
	"""

	if isinstance(dim, int):
		dim = [dim]
		#npad = [npad]
	else:
		dim=dim

	#pad 
	dim_counter=0
	for d in dim:
		
		pad = npad[dim_counter]
		if d >= len(x.shape):
			raise IndexError(f"dim {d} out of range")

		
		idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
		x = torch.cat([x, x[idx]], dim=d)
		
		idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))

		x = torch.cat([x[idx], x], dim=d)
		dim_counter+=1
		pass

	return x