import pdb

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from scipy.interpolate import RegularGridInterpolator as rgi

def rescale_voxel_size(volume, aff, new_vox_size, not_aliasing=False, method='linear'):
    """This function resizes the voxels of a volume to a new provided size, while adjusting the header to keep the RAS
    :param volume: a numpy array
    :param aff: affine matrix of the volume
    :param new_vox_size: new voxel size (3 - element numpy vector) in mm
    :return: new volume and affine matrix
    """

    pixdim = np.sqrt(np.sum(aff * aff, axis=0))[:-1]
    new_vox_size = np.array(new_vox_size)
    factor = pixdim / new_vox_size
    sigmas = 0.25 / factor
    sigmas[factor > 1] = 0  # don't blur if upsampling

    if len(volume.shape) > 3:
        sigmas = np.concatenate((sigmas, [0]))

    if all(sigmas == 0) or not_aliasing:
        volume_filt = volume
    else:
        volume_filt = gaussian_filter(volume, sigmas)

    # volume2 = zoom(volume_filt, factor, order=1, mode='reflect', prefilter=False)
    x = np.arange(0, volume_filt.shape[0])
    y = np.arange(0, volume_filt.shape[1])
    z = np.arange(0, volume_filt.shape[2])

    my_interpolating_function = rgi((x, y, z), volume_filt, method)

    start = - (factor - 1) / (2 * factor)
    step = 1.0 / factor
    stop = start + step * np.ceil(volume_filt.shape[:3] * factor)

    xi = np.arange(start=start[0], stop=stop[0], step=step[0])
    yi = np.arange(start=start[1], stop=stop[1], step=step[1])
    zi = np.arange(start=start[2], stop=stop[2], step=step[2])
    xi[xi < 0] = 0
    yi[yi < 0] = 0
    zi[zi < 0] = 0
    xi[xi > (volume_filt.shape[0] - 1)] = volume_filt.shape[0] - 1
    yi[yi > (volume_filt.shape[1] - 1)] = volume_filt.shape[1] - 1
    zi[zi > (volume_filt.shape[2] - 1)] = volume_filt.shape[2] - 1

    xig, yig, zig = np.meshgrid(xi, yi, zi, indexing='ij', sparse=True)
    volume2 = my_interpolating_function((xig, yig, zig))

    aff2 = aff.copy()
    for c in range(3):
        aff2[:-1, c] = aff2[:-1, c] / factor[c]
    aff2[:-1, -1] = aff2[:-1, -1] - np.matmul(aff2[:-1, :-1], 0.5 * (factor - 1))

    return volume2, aff2


def crop_label(mask, ref_shape=None, threshold=0, margin=None):
    crop_coord = []
    idx = np.where(mask>threshold)

    if ref_shape is not None:
        for it_index, index in enumerate(idx):
            clow = max(0, np.min(idx[it_index]))
            chigh = min(mask.shape[it_index], np.max(idx[it_index]))

            add_margin = (ref_shape[it_index] - (chigh - clow)) // 2
            clow = max(0, clow - add_margin)
            chigh = min(mask.shape[it_index], clow + ref_shape[it_index])
            crop_coord.append([clow, chigh])
    elif margin is not None:
        ndim = len(mask.shape)
        if isinstance(margin, int):
            margin = [margin] * ndim
        for it_index, index in enumerate(idx):
            clow = max(0, np.min(idx[it_index]) - margin[it_index])
            chigh = min(mask.shape[it_index], np.max(idx[it_index]) + margin[it_index])
            crop_coord.append([clow, chigh])
    else:
        raise ValueError("Please, specify either ref_shape or margin in crop_label function.")

    mask_cropped = mask[
                   crop_coord[0][0]: crop_coord[0][1],
                   crop_coord[1][0]: crop_coord[1][1],
                   crop_coord[2][0]: crop_coord[2][1]
                   ]

    return mask_cropped, crop_coord


def binary_distance_map(labelmap, th=0.5, margin=None):
    mask_label = labelmap > th
    if np.sum(mask_label) == 0:
        return -200 * np.ones(labelmap.shape, dtype='float32')
    else:
        if margin is not None:
            distancemap = -np.sqrt(margin ** 2 + margin ** 2) * np.zeros(labelmap.shape, dtype='float32')
            mask_label, crop_coord = crop_label(mask_label, margin=5)

        d_in = (distance_transform_edt(mask_label))
        d_out = -distance_transform_edt(~mask_label)
        d = np.zeros_like(d_in)
        d[mask_label] = d_in[mask_label]
        d[~mask_label] = d_out[~mask_label]

        if margin is not None:
            distancemap[
                crop_coord[0][0]: crop_coord[0][1],
                crop_coord[1][0]: crop_coord[1][1],
                crop_coord[2][0]: crop_coord[2][1]] = d
            return distancemap
        else:
            return d


from monai.transforms import MapTransform
from monai.config.type_definitions import KeysCollection, NdarrayOrTensor
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Union
from monai.utils import TransformBackends, misc
import torch

def movedim1(x, source: int, destination: int):
    dim = x.dim()
    source = dim + source if source < 0 else source
    destination = dim + destination if destination < 0 else destination
    permutation = [d for d in range(dim)]
    permutation = permutation[:source] + permutation[source+1:]
    permutation = permutation[:destination] + [source] + permutation[destination:]
    return x.permute(permutation)

def list_reverse_int(x: List[int]) -> List[int]:
    if len(x) == 0:
        return x
    return [x[i] for i in range(-1, -len(x)-1, -1)]

def l1dt_1d_(f, dim: int = -1, w: float = 1.):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if f.shape[dim] == 1:
        return f

    f = movedim1(f, dim, 0)

    for q in range(1, len(f)):
        f[q] = torch.min(f[q], f[q-1] + w)
    rng: List[int] = [e for e in range(len(f)-1)]
    for q in list_reverse_int(rng):
        f[q] = torch.min(f[q], f[q+1] + w)

    f = movedim1(f, 0, dim)
    return f

if hasattr(torch, 'true_divide'):
    _true_div = torch.true_divide
else:
    _true_div = torch.div

def square(x):
    return x * x

@torch.jit.script
def l1dt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 2 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    return l1dt_1d_(f.clone(), dim, w)


def make_vector(input, n=None, crop=True, *args,
                dtype=None, device=None, **kwargs):
    """Ensure that the input is a (tensor) vector and pad/crop if necessary.

    Parameters
    ----------
    input : scalar or sequence or generator
        Input argument(s).
    n : int, optional
        Target length.
    crop : bool, default=True
        Crop input sequence if longer than `n`.
    default : optional
        Default value to pad with.
        If not provided, replicate the last value.
    dtype : torch.dtype, optional
        Output data type.
    device : torch.device, optional
        Output device

    Returns
    -------
    output : tensor
        Output vector.

    """
    input = torch.as_tensor(input, dtype=dtype, device=device).flatten()
    if n is None:
        return input
    if n is not None and input.numel() >= n:
        return input[:n] if crop else input
    if args:
        default = args[0]
    elif 'default' in kwargs:
        default = kwargs['default']
    else:
        default = input[-1]
    default = input.new_full([n-len(input)], default)
    return torch.cat([input, default])

def edt_1d_fillin(f, v, z, w2: float = 1.):
    # process along the first dimension
    #
    # f: input function
    # v: locations of parabolas in lower envelope
    # z: location of boundaries between parabolas

    k = f.new_zeros(f.shape[1:], dtype=torch.long)
    d = torch.empty_like(f)
    for q in range(len(f)):

        zk = z.gather(0, k[None] + 1)[0]
        mask = zk < q

        while mask.any():
            k = k.add_(mask)
            zk = z.gather(0, k[None] + 1)[0]
            mask = zk < q

        vk = v.gather(0, k[None])[0]
        fvk = f.gather(0, vk[None])[0]
        d[q] = w2 * square(q - vk) + fvk

    return d


@torch.jit.script
def edt_1d_intersection(f, v, z, k, q: int, w2: float = 1.):
    vk = v.gather(0, k[None])[0]
    fvk = f.gather(0, vk[None])[0]
    fq = f[q]
    a, b = w2 * (q - vk), q + vk
    s = _true_div((fq - fvk) + a * b, 2 * a)
    zk = z.gather(0, k[None])[0]
    mask = (k > 0) & (s <= zk)
    return s, mask

def edt_1d(f, dim: int = -1, w: float = 1.):
    """Algorithm 1 in "Distance Transforms of Sampled Functions"
    Pedro F. Felzenszwalb & Daniel P. Huttenlocher
    Theory of Computing (2012)
    https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """

    if f.shape[dim] == 1:
        return f

    w = w * w                                        # unit length (squared)
    f = movedim1(f, dim, 0)                          # input function
    k = f.new_zeros(f.shape[1:], dtype=torch.long)   # index of rightmost parabola in lower envelope
    v = f.new_zeros(f.shape, dtype=torch.long)       # locations of parabolas in lower envelope
    z = f.new_empty([len(f)+1] + list(f.shape[1:]))  # location of boundaries between parabolas

    # compute lower envelope
    z[0] = -float('inf')
    z[1] = float('inf')
    for q in range(1, len(f)):

        s, mask = edt_1d_intersection(f, v, z, k, q, w)
        while mask.any():
            k.add_(mask, alpha=-1)
            s, mask = edt_1d_intersection(f, v, z, k, q, w)

        s.masked_fill_(torch.isnan(s), -float('inf'))  # is this correct?

        k.add_(1)
        v.scatter_(0, k[None], q)
        z.scatter_(0, k[None], s[None])
        z.scatter_(0, k[None] + 1, float('inf'))

    # fill in values of distance transform
    d = edt_1d_fillin(f, v, z, w)
    d = movedim1(d, 0, dim)
    return d

def euclidean_distance_transform(x, ndim=None, vx=1):
    """Compute the Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor. Zeros will stay zero, and the distance will
        be propagated into nonzero voxels.
    ndim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    # if backend.jitfields and jitfields.available:
    #     return jitfields.euclidean_distance_transform(x, ndim, vx)

    dtype = x.dtype if x.dtype.is_floating_point else torch.get_default_dtype()
    x = x.to(dtype, copy=True)
    x.masked_fill_(x > 0, float('inf'))
    ndim = ndim or x.dim()
    vx = make_vector(vx, ndim, dtype=torch.float).tolist()
    x = l1dt_1d_(x, -ndim, vx[0]).square_()
    for d, w in zip(range(1, ndim), vx[1:]):
        x = edt_1d(x, d - ndim, w)
    x.sqrt_()
    return x


def euclidean_signed_transform(x, ndim=None, vx=1, threshold=0):
    """Compute the signed Euclidean distance transform of a binary image

    Parameters
    ----------
    x : (..., *spatial) tensor
        Input tensor.
        A negative distance will propagate into zero voxels and
        a positive distance will propagate into nonzero voxels.
    ndim : int, default=`x.dim()`
        Number of spatial dimensions
    vx : [sequence of] float, default=1
        Voxel size

    Returns
    -------
    d : (..., *spatial) tensor
        Signed distance map

    References
    ----------
    ..[1] "Distance Transforms of Sampled Functions"
          Pedro F. Felzenszwalb & Daniel P. Huttenlocher
          Theory of Computing (2012)
          https://www.theoryofcomputing.org/articles/v008a019/v008a019.pdf
    """
    if x.dtype is not torch.bool:
        x = x > threshold

    d = euclidean_distance_transform(x, ndim, vx)
    d -= euclidean_distance_transform(~x, ndim, vx)
    return torch.clamp(d, min=-3, max=3)

class EDTransform(MapTransform):
    """
    Dictionary-based wrapper of :py:class:`monai.transforms.AsDiscrete`.
    """

    backend = [TransformBackends.TORCH]

    def __init__(
        self,
        keys: KeysCollection,
        to_onehot: Sequence[int | None] | int | None = None,
        allow_missing_keys: bool = False,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            keys: keys of the corresponding items to model output and label.
                See also: :py:class:`monai.transforms.compose.MapTransform`
            argmax: whether to execute argmax function on input data before transform.
                it also can be a sequence of bool, each element corresponds to a key in ``keys``.
            to_onehot: if not None, convert input data into the one-hot format with specified number of classes.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            threshold: if not None, threshold the float values to int number 0 or 1 with specified threshold value.
                defaults to ``None``. it also can be a sequence, each element corresponds to a key in ``keys``.
            rounding: if not None, round the data according to the specified option,
                available options: ["torchrounding"]. it also can be a sequence of str or None,
                each element corresponds to a key in ``keys``.
            allow_missing_keys: don't raise exception if key is missing.
            kwargs: additional parameters to ``AsDiscrete``.
                ``dim``, ``keepdim``, ``dtype`` are supported, unrecognized parameters will be ignored.
                These default to ``0``, ``True``, ``torch.float`` respectively.

        """
        super().__init__(keys, allow_missing_keys)
        self.converter = euclidean_signed_transform
        self.threshold = []
        for flag in misc.ensure_tuple_rep(threshold, len(self.keys)):
            if isinstance(flag, bool):
                raise ValueError("`threshold_values=True/False` is deprecated, please use `threshold=value` instead.")
            self.threshold.append(flag)

        self.to_onehot = []
        for flag in misc.ensure_tuple_rep(to_onehot, len(self.keys)):
            if isinstance(flag, bool):
                raise ValueError("`to_onehot=True/False` is deprecated, please use `to_onehot=num_classes` instead.")
            self.to_onehot.append(flag)

    def __call__(self, data: Mapping[Hashable, NdarrayOrTensor]) -> Dict[Hashable, NdarrayOrTensor]:
        d = dict(data)
        for key, threshold, to_onehot in self.key_iterator(d, self.threshold, self.to_onehot):
            if to_onehot == 2:
                d[key] = self.converter(d[key], ndim=3, threshold=threshold)
                d[key] = torch.stack(-d[key], d[key], dim=0)
            elif to_onehot is not None:
                raise ValueError('Still not implemented.')
            else:
                d[key] = self.converter(d[key], ndim=3, threshold=threshold)

            
        return d