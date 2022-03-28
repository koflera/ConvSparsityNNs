import torch

def pad_circular_nd(x, pad, dim):
	
	"""
	function for circular padding
	"""

	if isinstance(dim, int):
		dim = [dim]

	for d in dim:
		if d >= len(x.shape):
			raise IndexError(f"dim {d} out of range")

		
		idx = tuple(slice(0, None if s != d else pad, 1) for s in range(len(x.shape)))
		x = torch.cat([x, x[idx]], dim=d)
		
		idx = tuple(slice(None if s != d else -2 * pad, None if s != d else -pad, 1) for s in range(len(x.shape)))

		x = torch.cat([x[idx], x], dim=d)
		pass

	return x