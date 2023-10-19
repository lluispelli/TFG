import pdb

import nibabel
import torch
from torch import nn
import numpy as np

from utils import def_utils, fn_utils

class InstanceModelClassic(nn.Module):

    def __init__(self, image_shape, ref_v2r=None, flo_v2r=None, batchsize=1, device='cpu', **kwargs):
        super().__init__()

        self.device = device
        self.batchsize = batchsize
        self.image_shape = image_shape
        self.ndims = 3
        self.ref_v2r = torch.from_numpy(ref_v2r)
        self.flo_v2r = torch.from_numpy(flo_v2r)

        vectors = [torch.arange(0, s) for s in self.image_shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        self.grid = torch.stack(grids).to(device)  # y, x, z

    @NotImplementedError
    def _compute_ras_matrix(self):
        return None

    @NotImplementedError
    def set_params(self):
        return None

    @NotImplementedError
    def get_params(self):
        return None

    def _compute_rotation(self, rotation):
        shape = rotation[..., 0].shape + (1,)

        Rx_row0 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([1., 0., 0.])), 0), shape),
                                  axis=1).to(self.device)
        Rx_row1 = torch.stack([torch.zeros(shape).to(self.device), torch.unsqueeze(torch.cos(rotation[..., 0]), -1),
                               torch.unsqueeze(-torch.sin(rotation[..., 0]), -1)], axis=-1)
        Rx_row2 = torch.stack([torch.zeros(shape).to(self.device), torch.unsqueeze(torch.sin(rotation[..., 0]), -1),
                               torch.unsqueeze(torch.cos(rotation[..., 0]), -1)], axis=-1)
        Rx = torch.cat([Rx_row0, Rx_row1, Rx_row2], axis=1)

        Ry_row0 = torch.stack([torch.unsqueeze(torch.cos(rotation[..., 1]), -1), torch.zeros(shape).to(self.device),
                               torch.unsqueeze(torch.sin(rotation[..., 1]), -1)], axis=-1)
        Ry_row1 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([0., 1., 0.])), 0), shape),
                                  axis=1).to(self.device)
        Ry_row2 = torch.stack([torch.unsqueeze(-torch.sin(rotation[..., 1]), -1), torch.zeros(shape).to(self.device),
                               torch.unsqueeze(torch.cos(rotation[..., 1]), -1)], axis=-1)
        Ry = torch.cat([Ry_row0, Ry_row1, Ry_row2], axis=1)

        Rz_row0 = torch.stack(
            [torch.unsqueeze(torch.cos(rotation[..., 2]), -1), torch.unsqueeze(-torch.sin(rotation[..., 2]), -1),
             torch.zeros(shape).to(self.device)], axis=-1)
        Rz_row1 = torch.stack(
            [torch.unsqueeze(torch.sin(rotation[..., 2]), -1), torch.unsqueeze(torch.cos(rotation[..., 2]), -1),
             torch.zeros(shape).to(self.device)], axis=-1)
        Rz_row2 = torch.unsqueeze(torch.tile(torch.unsqueeze(torch.from_numpy(np.array([0., 0., 1.])), 0), shape),
                                  axis=1).to(self.device)
        Rz = torch.cat([Rz_row0, Rz_row1, Rz_row2], axis=1)

        T_rot = torch.matmul(torch.matmul(Rx, Ry), Rz)

        return T_rot

    def _compute_matrix(self):

        T_rig = self._compute_ras_matrix()
        T_rig_list = torch.unbind(T_rig, dim=0)
        T_rig_ras_list = [torch.linalg.inv(self.flo_v2r) @ T @ self.ref_v2r for T in T_rig_list]
        T = torch.stack(T_rig_ras_list, 0)

        return T.to(self.device)

    def get_ras_matrix(self):
        return self._compute_ras_matrix()

    def get_matrix(self):
        return self._compute_matrix()

    def forward(self, image_targ, **kwargs):
        T = self._compute_matrix()

        T = T[0]
        di = T[0, 0] * self.grid[0] + T[0, 1] * self.grid[1] + T[0, 2] * self.grid[2] + T[0, 3]
        dj = T[1, 0] * self.grid[0] + T[1, 1] * self.grid[1] + T[1, 2] * self.grid[2] + T[1, 3]
        dk = T[2, 0] * self.grid[0] + T[2, 1] * self.grid[1] + T[2, 2] * self.grid[2] + T[2, 3]

        image_reg = def_utils.fast_3D_interp_torch(image_targ[0, 0], di, dj, dk, mode='linear')

        return torch.unsqueeze(torch.unsqueeze(image_reg, 0), 0)

class InstanceRigidModelClassic(InstanceModelClassic):

    def __init__(self, image_shape, ref_v2r=None, flo_v2r=None, batchsize=1, device='cpu', cog=None,
                 tx_init=None, angle_init=None, tx_factor=1, angle_factor=1):
        super().__init__(image_shape, ref_v2r=ref_v2r, flo_v2r=flo_v2r, batchsize=batchsize, device=device)

        self.tx_factor = tx_factor
        self.angle_factor = angle_factor
        self.cog = cog

        # Parameters
        if angle_init is not None:
            if torch.is_tensor(angle_init):
                self.angle = torch.nn.Parameter(angle_init/angle_factor)
            else:
                self.angle = torch.nn.Parameter(torch.from_numpy(angle_init/angle_factor))

        else:
            self.angle = torch.nn.Parameter(torch.randn(self.batchsize, 3))

        if tx_init is not None:
            if torch.is_tensor(tx_init):
                self.translation = torch.nn.Parameter(tx_init/tx_factor)
            else:
                self.translation = torch.nn.Parameter(torch.from_numpy(tx_init)/tx_factor)
        else:
            self.translation = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        self.angle.requires_grad = True
        self.translation.requires_grad = True

    def set_params(self, params):
        self.angle = torch.nn.Parameter(params[0]/self.angle_factor)
        self.translation = torch.nn.Parameter(params[1]/self.tx_factor)

    def get_params(self):
        return self.angle * self.angle_factor, self.translation * self.tx_factor

    def _compute_ras_matrix(self):

        angle, tx = self.get_params()

        T_center = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center[:, 0, 0] = 1
        T_center[:, 1, 1] = 1
        T_center[:, 2, 2] = 1
        T_center[:, 3, 3] = 1
        if self.cog is None:
            T_center[:, :3, 3] = (self.ref_v2r @ torch.from_numpy(np.asarray([-i/2 for i in self.image_shape[2:]] + [1])).float())[:3]
        else:
            T_center[:, :3, 3] = torch.from_numpy(-self.cog)

        T_center_inv = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center_inv[:, 0, 0] = 1
        T_center_inv[:, 1, 1] = 1
        T_center_inv[:, 2, 2] = 1
        T_center_inv[:, 3, 3] = 1
        if self.cog is None:
            T_center_inv[:, :3, 3] = -(self.ref_v2r @ torch.from_numpy(np.asarray([-i/2 for i in self.image_shape[2:]] + [1])).float())[:3]
        else:
            T_center_inv[:, :3, 3] = torch.from_numpy(self.cog)

        T_trans = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_trans[:, :3, 3] = tx
        T_trans[:, 0, 0] = 1
        T_trans[:, 1, 1] = 1
        T_trans[:, 2, 2] = 1
        T_trans[:, 3, 3] = 1

        T_rot = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_rot[:, :3, :3] = self._compute_rotation(angle)
        T_rot[:, 3, 3] = 1

        T_rig = T_center_inv @ T_trans @ T_rot @ T_center

        return T_rig

class InstanceSimilarityModelClassic(InstanceModelClassic):

    def __init__(self, image_shape, ref_v2r=None, flo_v2r=None, batchsize=1, device='cpu', cog=None,
                 tx_init=None, angle_init=None, sc_init=None, tx_factor=1, angle_factor=1, sc_factor=1):
        super().__init__(image_shape, ref_v2r=ref_v2r, flo_v2r=flo_v2r, batchsize=batchsize, device=device)

        self.tx_factor = tx_factor
        self.angle_factor = angle_factor
        self.sc_factor = sc_factor
        self.cog = cog

        # Parameters
        if angle_init is not None:
            if torch.is_tensor(angle_init):
                self.angle = torch.nn.Parameter(angle_init/angle_factor)
            else:
                self.angle = torch.nn.Parameter(torch.from_numpy(angle_init/angle_factor))

        else:
            self.angle = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        if tx_init is not None:
            if torch.is_tensor(tx_init):
                self.translation = torch.nn.Parameter(tx_init/tx_factor)
            else:
                self.translation = torch.nn.Parameter(torch.from_numpy(tx_init)/tx_factor)
        else:
            self.translation = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        if sc_init is not None:
            if torch.is_tensor(sc_init):
                self.scaling = torch.nn.Parameter(torch.log(sc_init / sc_factor))
            else:
                self.scaling = torch.nn.Parameter(torch.from_numpy(np.log(sc_init / sc_factor)))
        else:
            self.scaling = torch.nn.Parameter(torch.zeros(self.batchsize, 3))

        self.angle.requires_grad = True
        self.scaling.requires_grad = True
        self.translation.requires_grad = True

    def _compute_ras_matrix(self):

        angle, scaling, tx = self.get_params()

        T_center = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center[:, 0, 0] = 1
        T_center[:, 1, 1] = 1
        T_center[:, 2, 2] = 1
        T_center[:, 3, 3] = 1
        if self.cog is None:
            T_center[:, :3, 3] = (self.ref_v2r @ torch.from_numpy(np.asarray([-i/2 for i in self.image_shape[2:]] + [1])).float())[:3]
        else:
            T_center[:, :3, 3] = torch.from_numpy(-self.cog)


        T_center_inv = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_center_inv[:, 0, 0] = 1
        T_center_inv[:, 1, 1] = 1
        T_center_inv[:, 2, 2] = 1
        T_center_inv[:, 3, 3] = 1
        if self.cog is None:
            T_center_inv[:, :3, 3] = -(self.ref_v2r @ torch.from_numpy(np.asarray([-i/2 for i in self.image_shape[2:]] + [1])).float())[:3]
        else:
            T_center_inv[:, :3, 3] = torch.from_numpy(self.cog)


        T_sc = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_sc[:, 0, 0] = scaling[:, 0]
        T_sc[:, 1, 1] = scaling[:, 1]
        T_sc[:, 2, 2] = scaling[:, 2]
        T_sc[:, 3, 3] = 1

        T_trans = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_trans[:, :3, 3] = tx
        T_trans[:, 0, 0] = 1
        T_trans[:, 1, 1] = 1
        T_trans[:, 2, 2] = 1
        T_trans[:, 3, 3] = 1

        T_rot = torch.zeros((self.batchsize, 4, 4)).to(self.device)
        T_rot[:, :3, :3] = self._compute_rotation(angle)
        T_rot[:, 3, 3] = 1

        T_rig = T_center_inv @ T_trans @ T_rot @ T_sc @ T_center

        return T_rig

    def set_params(self, params):
        self.angle = torch.nn.Parameter(params[0] / self.angle_factor)
        self.translation = torch.nn.Parameter(params[1] / self.tx_factor)
        self.scaling = torch.nn.Parameter(torch.log(params[2] / self.sc_factor))

    def get_params(self):
        return self.angle * self.angle_factor, torch.exp(self.scaling * self.sc_factor), self.translation * self.tx_factor

class DiceLoss(nn.Module):

    def forward(self, y_pred, y_true, eps=1e-5, **kwargs):
        y_true = y_true > 0 # if distance, positive is inside, if onehot is 1 or 0

        mask = kwargs['mask'] if 'mask' in kwargs else 1
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (mask * y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((mask * (y_true + y_pred)).sum(dim=vol_axes), min=eps)
        dice = torch.mean(top / bottom)
        return 1 - torch.mean(dice)

def compute_gradient(image):
    gradient_map = np.zeros(image.shape)

    dx = image[2:, :, :] - image[:-2, :, :]
    dy = image[:, 2:, :] - image[:, :-2, :]
    dz = image[:, :, 2:] - image[:, :, :-2]

    gradient_map[1:-1, :, :] = dx**2
    gradient_map[:, 1:-1, :] = dy**2
    gradient_map[:, :, 1:-1] = dz**2


    return np.sqrt(gradient_map)

def register(proxyref, proxyflo, loss_fn, model_type='rigid', max_iter=20, lr=1e-3, initialise_with_dist=False):

    pixdim = np.sqrt(np.sum(proxyref.affine * proxyref.affine, axis=0))[:-1]
    ref_image = np.array(proxyref.dataobj)
    flo_image = np.array(proxyflo.dataobj)
    flo_image, aff_flo = fn_utils.rescale_voxel_size(flo_image.astype('float32'), proxyflo.affine, pixdim)

    aux_xyz = np.where(ref_image > 0)
    ref_COG = {}
    for it_z in np.sort(np.unique(aux_xyz[2])):
        aux_xy = np.where(ref_image[..., it_z] > 0)
        cx, cy = np.median(aux_xy[0]), np.median(aux_xy[1])
        if ref_image[int(cx), int(cy), it_z] == 0: continue
        ref_COG[it_z] = proxyref.affine @ np.array([cx, cy, it_z, 1])
    min_z = min(ref_COG.keys())
    ref_COG = ref_COG[min_z]

    aux_xyz = np.where(flo_image > 0)
    flo_COG = {}
    for it_z in np.sort(np.unique(aux_xyz[2])):
        aux_xy = np.where(flo_image[..., it_z] > 0)
        cx, cy = np.median(aux_xy[0]), np.median(aux_xy[1])
        if flo_image[int(cx), int(cy), it_z] == 0: continue
        flo_COG[it_z] = aff_flo @ np.array([cx, cy, it_z, 1])
    min_z = min(flo_COG.keys())
    flo_COG = flo_COG[min_z]

    T_ras = np.eye(4)
    T_ras[0, 3] = (-ref_COG[0] + flo_COG[0])
    T_ras[1, 3] = (-ref_COG[1] + flo_COG[1])
    T_ras[2, 3] = (-ref_COG[2] + flo_COG[2])

    ref_image = np.array(proxyref.dataobj)
    ref_tensor = torch.from_numpy(ref_image[np.newaxis, np.newaxis]).float()
    flo_image = np.array(proxyflo.dataobj)
    flo_tensor = torch.from_numpy(flo_image[np.newaxis, np.newaxis]).float()




    model = InstanceSimilarityModelClassic(ref_image.shape,
                                           ref_v2r=proxyref.affine.astype('float32'),
                                           flo_v2r=proxyflo.affine.astype('float32'),
                                           device="cpu",
                                           tx_init=np.array([[T_ras[0, 3], T_ras[1, 3], T_ras[2, 3]]]),
                                           cog=np.array([0, 0, 0]))
    optimizer = torch.optim.LBFGS(params=model.parameters(), lr=lr, max_iter=10, line_search_fn='strong_wolfe')
    if initialise_with_dist:
        print('   Initialisation with distance map.')
        ref_image_dist = fn_utils.binary_distance_map(ref_image)
        flo_image_dist = fn_utils.binary_distance_map(flo_image)
        ref_tensor_dist = torch.from_numpy(ref_image_dist[np.newaxis, np.newaxis]).float()
        flo_tensor_dist = torch.from_numpy(flo_image_dist[np.newaxis, np.newaxis]).float()
        mse_loss = torch.nn.MSELoss()
        def closure_dist():
            optimizer.zero_grad()

            reg_flo_tensor = model(flo_tensor_dist)
            loss = mse_loss(reg_flo_tensor, ref_tensor_dist)
            loss.backward()

            return loss

        optimizer.step(closure=closure_dist)
        with torch.no_grad():
            reg_flo_tensor = model(flo_tensor_dist)
            loss = torch.nn.MSELoss()(reg_flo_tensor, ref_tensor_dist)
            print('   Loss: ' + str(loss.item()))
    else:
        print('   Initialisation with image COG.')

    print('   Refinement with Dice.')
    last_loss = 1000000
    for it in range(max_iter):
        def closure():
            optimizer.zero_grad()

            reg_flo_tensor = model(flo_tensor)
            loss = loss_fn(reg_flo_tensor, ref_tensor)
            loss.backward()

            return loss

        optimizer.step(closure=closure)
        with torch.no_grad():
            reg_flo_tensor = model(flo_tensor)
            loss = loss_fn(reg_flo_tensor, ref_tensor)
            print('   Loss: ' + str(loss.item()))
            if (last_loss - loss.item())/last_loss < 0.025:
                break
            else:
                last_loss = loss.item()
    with torch.no_grad():
        aff_T = model.get_ras_matrix().detach().numpy()[0]

    vectors = [torch.arange(0, s) for s in ref_image.shape]
    grids = torch.meshgrid(vectors, indexing='ij')
    grid = torch.stack(grids)

    T = torch.from_numpy(np.linalg.inv(proxyflo.affine) @ aff_T @ proxyref.affine)
    di = T[0, 0] * grid[0] + T[0, 1] * grid[1] + T[0, 2] * grid[2] + T[0, 3]
    dj = T[1, 0] * grid[0] + T[1, 1] * grid[1] + T[1, 2] * grid[2] + T[1, 3]
    dk = T[2, 0] * grid[0] + T[2, 1] * grid[1] + T[2, 2] * grid[2] + T[2, 3]
    flo_tensor = torch.from_numpy(flo_image.astype('float32'))
    image_reg = def_utils.fast_3D_interp_torch(flo_tensor, di, dj, dk, mode='linear')

    if DiceLoss()(torch.unsqueeze(torch.unsqueeze(image_reg, 0), 0), ref_tensor) > 0.4:
        good_flag = False
    else:
        good_flag = True

    return image_reg.numpy(), aff_T, good_flag