import torch


def make_color_wheel():
	"""
	Generate color wheel according Middlebury color code
	:return: Color wheel
	"""
	RY = 15
	YG = 6
	GC = 4
	CB = 11
	BM = 13
	MR = 6

	ncols = RY + YG + GC + CB + BM + MR

	colorwheel = torch.zeros([3, ncols])

	col = 0

	# RY
	colorwheel[0, 0:RY] = 255
	colorwheel[1, 0:RY] = torch.floor(255 * torch.arange(0, RY) / RY)
	col += RY

	# YG
	colorwheel[0, col:col + YG] = 255 - torch.floor(255 * torch.arange(0, YG) / YG)
	colorwheel[1, col:col + YG] = 255
	col += YG

	# GC
	colorwheel[1, col:col + GC] = 255
	colorwheel[2, col:col + GC] = torch.floor(255 * torch.arange(0, GC) / GC)
	col += GC

	# CB
	colorwheel[1, col:col + CB] = 255 - torch.floor(255 * torch.arange(0, CB) / CB)
	colorwheel[2, col:col + CB] = 255
	col += CB

	# BM
	colorwheel[2, col:col + BM] = 255
	colorwheel[0, col:col + BM] = torch.floor(255 * torch.arange(0, BM) / BM)
	col += + BM

	# MR
	colorwheel[2, col:col + MR] = 255 - torch.floor(255 * torch.arange(0, MR) / MR)
	colorwheel[0, col:col + MR] = 255

	return colorwheel


colorwheel = make_color_wheel().cuda()


def flow2img(flow_data: torch.Tensor):
	"""
	convert optical flow into color image
	:param flow_data:
	:return: color image
	"""
	# print(flow_data.shape)
	# print(type(flow_data))
	u = flow_data[:, 0:1, :, :]
	v = flow_data[:, 1:2, :, :]

	UNKNOW_FLOW_THRESHOLD = 1e7
	pr1 = torch.abs(u) > UNKNOW_FLOW_THRESHOLD
	pr2 = torch.abs(v) > UNKNOW_FLOW_THRESHOLD
	idx_unknown = (pr1 | pr2)
	u[idx_unknown] = 0
	v[idx_unknown] = 0
	idx_unknown = idx_unknown.repeat(1, 3, 1, 1)

	rad = torch.sqrt(u ** 2 + v ** 2)
	maxrad = max(-1, torch.max(rad).item())
	u = u / maxrad + torch.finfo(float).eps
	v = v / maxrad + torch.finfo(float).eps

	img = compute_color(u, v)

	img[idx_unknown] = 0

	return img / 255.


def compute_color(u, v):
	"""
	compute optical flow color map
	:param u: horizontal optical flow
	:param v: vertical optical flow
	:return:
	"""

	B, _, H, W = u.shape
	img = torch.zeros((B, 3, H, W), device=torch.device('cuda'))

	NAN_idx = torch.isnan(u) | torch.isnan(v)
	u[NAN_idx] = v[NAN_idx] = 0
	ncols = colorwheel.shape[1]

	rad = torch.sqrt(u ** 2 + v ** 2)

	a = torch.arctan2(-v, -u) / torch.pi

	fk = (a + 1) / 2 * (ncols - 1) + 1

	k0 = torch.floor(fk).to(int)

	k1 = k0 + 1
	k1[k1 == ncols + 1] = 1
	f = fk - k0

	for i in range(0, colorwheel.shape[0]):
		tmp = colorwheel[i, :]
		col0 = tmp[k0 - 1] / 255
		col1 = tmp[k1 - 1] / 255
		col = (1 - f) * col0 + f * col1

		idx = rad <= 1
		col[idx] = 1 - rad[idx] * (1 - col[idx])
		notidx = torch.logical_not(idx)

		col[notidx] *= 0.75
		img[:, i:i+1, :, :] = torch.floor(255 * col * (~NAN_idx)).to(torch.uint8)

	return img

