import pdb

import torch
import nibabel as nib
import numpy as np



def fast_3D_interp_torch(X, II, JJ, KK, mode, pad='zeros'):
    if mode == 'nearest':
        IIr = torch.round(II).long()
        JJr = torch.round(JJ).long()
        KKr = torch.round(KK).long()
        IIr[IIr < 0] = 0
        JJr[JJr < 0] = 0
        KKr[KKr < 0] = 0
        IIr[IIr > (X.shape[0] - 1)] = (X.shape[0] - 1)
        JJr[JJr > (X.shape[1] - 1)] = (X.shape[1] - 1)
        KKr[KKr > (X.shape[2] - 1)] = (X.shape[2] - 1)
        Y = X[IIr, JJr, KKr]

    elif mode == 'linear':
        ok = (II>0) & (JJ>0) & (KK>0) & (II<=X.shape[0]-1) & (JJ<=X.shape[1]-1) & (KK<=X.shape[2]-1)
        IIv = II[ok]
        JJv = JJ[ok]
        KKv = KK[ok]

        fx = torch.floor(IIv).long()
        cx = fx + 1
        cx[cx > (X.shape[0] - 1)] = (X.shape[0] - 1)
        wcx = IIv - fx
        wfx = 1 - wcx

        fy = torch.floor(JJv).long()
        cy = fy + 1
        cy[cy > (X.shape[1] - 1)] = (X.shape[1] - 1)
        wcy = JJv - fy
        wfy = 1 - wcy

        fz = torch.floor(KKv).long()
        cz = fz + 1
        cz[cz > (X.shape[2] - 1)] = (X.shape[2] - 1)
        wcz = KKv - fz
        wfz = 1 - wcz

        c000 = X[fx, fy, fz]
        c100 = X[cx, fy, fz]
        c010 = X[fx, cy, fz]
        c110 = X[cx, cy, fz]
        c001 = X[fx, fy, cz]
        c101 = X[cx, fy, cz]
        c011 = X[fx, cy, cz]
        c111 = X[cx, cy, cz]

        c00 = c000 * wfx + c100 * wcx
        c01 = c001 * wfx + c101 * wcx
        c10 = c010 * wfx + c110 * wcx
        c11 = c011 * wfx + c111 * wcx

        c0 = c00 * wfy + c10 * wcy
        c1 = c01 * wfy + c11 * wcy

        c = c0 * wfz + c1 * wcz

        if pad == 'zeros':
            Y = torch.zeros(II.shape, device='cpu')
        elif pad == 'ct':
            Y = -2000 * torch.ones(II.shape, device='cpu')
        else:
            Y = torch.zeros(II.shape, device='cpu')


        Y[ok] = c.float()

    else:
        raise Exception('mode must be linear or nearest')

    return Y



def vol_resample(ref_proxy, flo_proxy, M, mode='linear'):
    Rshape = ref_proxy.shape
    Raff = ref_proxy.affine
    Faff = flo_proxy.affine
    F = np.array(flo_proxy.dataobj)
    Fdtype = F.dtype

    II, JJ, KK = np.meshgrid(np.arange(Rshape[0]), np.arange(Rshape[1]), np.arange(Rshape[2]), indexing='ij')
    II = torch.tensor(II)
    JJ = torch.tensor(JJ)
    KK = torch.tensor(KK)

    # Reference
    affine = torch.tensor(np.matmul(np.linalg.inv(Faff), np.matmul(M, Raff)))
    II2 = affine[0, 0] * II + affine[0, 1] * JJ + affine[0, 2] * KK + affine[0, 3]
    JJ2 = affine[1, 0] * II + affine[1, 1] * JJ + affine[1, 2] * KK + affine[1, 3]
    KK2 = affine[2, 0] * II + affine[2, 1] * JJ + affine[2, 2] * KK + affine[2, 3]
    Rlin = fast_3D_interp_torch(torch.tensor(F.astype('float').copy()), II2, JJ2, KK2, mode)

    return nib.Nifti1Image(Rlin.numpy().astype(Fdtype), Raff)

