#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Nov 26 2025

Example methods for a test data type.

2D planar maps with convolution using kernel.

This class makes all computations in torch.

Characteristics:
    - in pytorch
    - assume real maps 
    - N0 gives x and y sizes for array shaped (..., Nx, Ny).
    - masks are supported in convolutions
"""
import math
import numpy as np
import torch
import torch.nn.functional as F

def _conv2d_same_symmetric(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    2D convolution with "same" output size and symmetric padding.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [..., C, Nx, Ny].
    w : torch.Tensor
        Kernel tensor of shape [O_c, C, wx, wy].

    Returns
    -------
    torch.Tensor
        Convolved tensor with shape [..., O_c, Nx, Ny].
    """

    *leading_dims, C, Nx, Ny = x.shape
    O_c, Cw, wx, wy = w.shape

    B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
    x4d = x.reshape(B, C, Nx, Ny)

    pad_x = wx // 2
    pad_y = wy // 2

    # Determine grouping strategy: if the input channel count matches the
    # number of output channels and the kernel is single-channel, use
    # depthwise convolution to keep channels independent (orientation-wise
    # filtering). Otherwise fall back to standard grouped convolution with a
    # broadcasted kernel when needed.
    if Cw == 1 and O_c == C:
        groups = C
        w = w.expand(C, 1, wx, wy).contiguous()
    else:
        groups = 1
        if Cw == 1 and C > 1:
            w = w.repeat(1, C, 1, 1)

    x_padded = F.pad(x4d, (pad_y, pad_y, pad_x, pad_x), mode="reflect")
    y = F.conv2d(x_padded, w, groups=groups)

    return y.reshape(*leading_dims, O_c, Nx, Ny)


def _conv2d_circular(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Backend-style 2D convolution mirroring FoCUS/BkTorch strategy.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape [..., C, Nx, Ny].
    w : torch.Tensor
        Kernel tensor of shape [O_c, C, wx, wy].

    Returns
    -------
    torch.Tensor
        Convolved tensor with shape [..., O_c, Nx, Ny].
    """

    *leading_dims, C, Nx, Ny = x.shape
    O_c, Cw, wx, wy = w.shape

    B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
    x4d = x.reshape(B, C, Nx, Ny)

    pad_x = wx // 2
    pad_y = wy // 2

    if Cw == 1 and O_c == C:
        groups = C
        w = w.expand(C, 1, wx, wy).contiguous()
    else:
        groups = 1
        if Cw == 1 and C > 1:
            w = w.repeat(1, C, 1, 1)

    x_padded = F.pad(x4d, (pad_y, pad_y, pad_x, pad_x), mode="circular")
    y = F.conv2d(x_padded, w, groups=groups)

    return y.reshape(*leading_dims, O_c, Nx, Ny)


def _complex_conv2d_same_symmetric(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Complex-aware wrapper around ``_conv2d_same_symmetric``."""

    xr = torch.real(x) if torch.is_complex(x) else x
    xi = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(xr)

    wr = torch.real(w) if torch.is_complex(w) else w
    wi = torch.imag(w) if torch.is_complex(w) else torch.zeros_like(wr)

    real_part = _conv2d_same_symmetric(xr, wr) - _conv2d_same_symmetric(xi, wi)
    imag_part = _conv2d_same_symmetric(xr, wi) + _conv2d_same_symmetric(xi, wr)

    if torch.is_complex(x) or torch.is_complex(w):
        return torch.complex(real_part, imag_part)
    return real_part


def _complex_conv2d_circular(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Complex-aware wrapper around ``_conv2d_circular``."""

    xr = torch.real(x) if torch.is_complex(x) else x
    xi = torch.imag(x) if torch.is_complex(x) else torch.zeros_like(xr)

    wr = torch.real(w) if torch.is_complex(w) else w
    wi = torch.imag(w) if torch.is_complex(w) else torch.zeros_like(wr)

    real_part = _conv2d_circular(xr, wr) - _conv2d_circular(xi, wi)
    imag_part = _conv2d_circular(xr, wi) + _conv2d_circular(xi, wr)

    if torch.is_complex(x) or torch.is_complex(w):
        return torch.complex(real_part, imag_part)
    return real_part

###############################################################################
###############################################################################
class STL_2D_Kernel_Torch:
    '''
    Class which contain the different types of data used in STL.
    Store important parameters, such as DT, N0, and the Fourier type.
    Also allow to convert from numpy to pytorch (or other type).
    Allow to transfer internally these parameters.
    
    Has different standard functions as methods (
    modulus, mean, cov, downsample)
    
    The initial resolution N0 is fixed, but the maps can be downgraded. The 
    downgrading factor is the power of 2 that is used. A map of initial 
    resolution N0=256 and with dg = 3 is thus at resolution 256/2^3 = 32.
    The downgraded resolutions are called N0, N1, N2, ...
    
    Can store array at a given downgradind dg:
        - attribute MR is False
        - attribute N0 gives the initial resolution
        - attribute dg gives the downgrading level
        - attribute list_dg is None
        - array is an array of size (..., N) with N = N0 // 2^dg 
    Or at multi-resolution (MR):
        - attribute MR is True
        - attribute N0 gives the initial resolution
        - attribute dg is None
        - attribute list_dg is the list of downgrading
        - array is a list of array of sizes (..., N1), (..., N2), etc., 
        with the same dimensions excepts N.
     
    Method usages if MR=True.
        - mean, cov give a single vector or last dim len(list_N)
        - downsample gives an output of size (..., len(list_N), Nout). Only 
          possible if all resolution are downsampled this way.
          
    The class initialization is the frontend one, which can work from DT and 
    data only. It enforces MR=False and dg=0. Two backend init functions for 
    MR=False and MR=True also exist.
    
    Attributes
    ----------
    - DT : str
        Type of data (1d, 2d planar, HealPix, 3d)
    - MR: bool
        True if store a list of array in a multi-resolution framework
    - N0 : tuple of int
        Initial size of array (can be multiple dimensions)
    - dg : int
        2^dg is the downgrading level w.r.t. N0. None if MR==False  
    - list_dg : list of int
        list of dowgrading level w.r.t. N0, None if MR==False
    - array : array (..., N) if MR==False
          liste of (..., N1), (..., N2), etc. if MR==True
          array(s) to store
         
    '''
    
    ###########################################################################
    def __init__(self, array, smooth_kernel=None,dg=None,N0=None):
        '''
        Constructor, see details above. Frontend version, which assume the 
        array is at N0 resolution with dg=0, with MR=False.
        
        More sophisticated Back-end constructors (_init_SR and _init_MR) exist.
        
        '''
        
        # Check that MR==False array is given
        if isinstance(array, list):
            raise ValueError("Only single resolution array are accepted.")
        
        # Main 
        self.DT = 'Planar2D_kernel_torch'
        self.MR = False
        if dg is None:
            self.dg = 0
            self.N0 = array.shape[-2:]
        else:
            self.dg=dg
            if N0 is None:
                raise ValueError("dg is given, N0 should not be None")
            self.N0=N0
        
        
        self.array = self.to_array(array)
        
        self.list_dg = None
        
        # Find N0 value
        self.device=self.array.device
        self.dtype=self.array.dtype

        if smooth_kernel is None:
            smooth_kernel=self._smooth_kernel(3)
        self.smooth_kernel=smooth_kernel
        
        
    def _smooth_kernel(self,kernel_size: int):
        """Create a 2D Gaussian kernel."""
        sigma=1
        coords = torch.arange(kernel_size, device=self.device, dtype=self.dtype) - (kernel_size - 1) / 2.0
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, kernel_size, kernel_size)
        
            
    ###########################################################################
    def to_array(self,array):
        """
        Transform input array (NumPy or PyTorch) into a PyTorch tensor.
        Should return None if None.

        Parameters
        ----------
        array : np.ndarray or torch.Tensor
            Input array to be converted.

        Returns
        -------
        torch.Tensor
            Converted PyTorch tensor.
        """
        
        if array is None:
            return None
        elif isinstance(array, list):
            return array

        # Choose device: use GPU if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(array, np.ndarray):
            return torch.from_numpy(array).to(device)
        elif isinstance(array, torch.Tensor):
            return array.to(device)
        else:
            raise TypeError(f"Unsupported array type: {type(array)}")
            
    ###########################################################################
    def copy(self, empty=False):
        """
        Copy a STL_2D_Kernel_Torch instance.
        Array is put to None if empty==True.
        
        Parameters
        ----------
        - empty : bool
            If True, set array to None.
                    
        Output 
        ----------
        - STL_2D_Kernel_Torch
           copy of self
        """
        new = object.__new__(STL_2D_Kernel_Torch)

        # Copy metadata
        new.MR = self.MR
        new.N0 = self.N0
        new.dg = self.dg
        new.list_dg = list(self.list_dg) if self.list_dg is not None else None
        new.device = self.device
        new.dtype = self.dtype

        # Copy kernels
        new.smooth_kernel = (self.smooth_kernel.clone()
                             if isinstance(self.smooth_kernel, torch.Tensor)
                             else None)

        # Copy array
        if empty:
            new.array = None
        else:
            if self.MR:
                new.array = [a.clone() if isinstance(a, torch.Tensor) else None
                             for a in self.array]
            else:
                new.array = (self.array.clone()
                             if isinstance(self.array, torch.Tensor) else None)

        return new

    ###########################################################################
    def __getitem__(self, key):
        """
        To slice directly the array attribute. Produce a view of array, to 
        match with usual practices, allowing to conveniently pass only part
        of an instance.
        """
        new = self.copy(empty=True)

        if self.MR:
            if not isinstance(self.array, list):
                raise ValueError("MR=True but array is not a list.")

            if isinstance(key, (int, slice)):
                new.array = self.array[key]
                new.list_dg = self.list_dg[key] if self.list_dg is not None else None

                # If a single element is selected, keep MR=True with a single resolution
                if isinstance(key, int):
                    new.array = [new.array]
                    new.list_dg = [new.list_dg]
            else:
                raise TypeError("Indexing MR=True data only supports int or slice.")
        else:
            new.array = self.array[key]

        return new
  
    @staticmethod
    def _downsample_tensor(x: torch.Tensor, dg_inc: int) -> torch.Tensor:
        """
        Downsample a tensor by a factor 2**dg_inc along the last two
        dimensions using 2x2 mean pooling (FoCUS ud_grade strategy).

        Requires that both spatial dimensions be divisible by 2**dg_inc.
        """
        if dg_inc < 0:
            raise ValueError("dg_inc must be non-negative")
        if dg_inc == 0:
            return x

        scale = 2 ** dg_inc
        H, W = x.shape[-2:]
        if H % scale != 0 or W % scale != 0:
            raise ValueError(
                f"Cannot downsample from ({H},{W}) by 2^{dg_inc}: "
                "dimensions must be divisible."
            )

    # Create Gaussian kernel for anti-aliasing
        def get_gaussian_kernel(sigma=1.0, dtype=torch.float32):
            """Create a 2D Gaussian kernel with kernel_size based on sigma"""
            kernel_size = int(6 * sigma + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            x_coord = torch.arange(kernel_size, dtype=dtype) - kernel_size // 2
            gauss_1d = torch.exp(-x_coord**2 / (2 * sigma**2))
            gauss_1d = gauss_1d / gauss_1d.sum()
            gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
            return gauss_2d.unsqueeze(0).unsqueeze(0), kernel_size

        leading_dims = x.shape[:-2]
        B = int(torch.prod(torch.tensor(leading_dims))) if leading_dims else 1
        y = x.reshape(B, 1, H, W)

        for _ in range(dg_inc):
            h, w = y.shape[-2:]
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError("Downsampling requires even spatial dimensions at each step.")
            sigma = 1.0
            kernel, kernel_size = get_gaussian_kernel(sigma, dtype=y.dtype)
            kernel = kernel.to(y.device)
            # Add circular padding for periodic boundaries
            pad = kernel_size // 2
            y_padded = F.pad(y, (pad, pad, pad, pad), mode='circular')
            y = F.conv2d(y_padded, kernel)
            # Now downsample with 2x2 mean pooling
            y = F.avg_pool2d(y, kernel_size=2, stride=2)

        H2, W2 = y.shape[-2:]
        return y.reshape(*leading_dims, H2, W2)
  
    ###########################################################################
    def downsample_toMR_Mask(self, dg_max):
        '''
        Take a mask given at a dg=0 resolution, and put it at all resolutions
        from dg=0 to dg=dg_max, in a MR=True object.

        Each resolution is normalized to have unit mean (over spatial dims).
        '''
        if self.MR:
            raise ValueError("downsample_toMR_Mask expects MR == False.")
        if self.dg != 0:
            raise ValueError("Mask should be at dg=0 to build a multi-resolution mask.")
        if self.array is None:
            raise ValueError("No array stored in this object.")

        list_masks = []
        list_dg = list(range(dg_max + 1))

        for dg in list_dg:
            if dg == 0:
                m = self.array
            else:
                m = self._downsample_tensor(self.array, dg)

            if (m < 0).any():
                raise ValueError("Mask contains negative values; expected non-negative weights.")

            mean = m.mean(dim=(-2, -1), keepdim=True)
            m = m / mean.clamp_min(1e-12)
            list_masks.append(m)

        Mask_MR = self.copy(empty=True)
        Mask_MR.MR = True
        Mask_MR.dg = None
        Mask_MR.list_dg = list_dg
        Mask_MR.array = list_masks

        return Mask_MR

    ###########################################################################
    def _get_mask_at_dg(self, mask_MR, dg):
        """Helper to pick the mask at a given dg from a MR mask object."""
        if mask_MR is None:
            return None
        if not mask_MR.MR:
            raise ValueError("mask_MR must have MR=True.")
        if mask_MR.list_dg is None:
            raise ValueError("mask_MR.list_dg is None.")
        try:
            idx = mask_MR.list_dg.index(dg)
        except ValueError:
            raise ValueError(f"Mask does not contain dg={dg}.")
        return mask_MR.array[idx]

    ###########################################################################
    def downsample(self, dg_out, mask_MR=None, inplace=True):
        """
        Downsample the data to the dg_out resolution.
        Only supports MR == False.

        Downsampling is done in real space by average pooling, with factor
        2^(dg_out - dg) on both spatial axes.
        """
        if self.MR:
            raise ValueError("downsample only supports MR == False.")
        if dg_out < 0:
            raise ValueError("dg_out must be non-negative.")
        if dg_out == self.dg and not copy:
            return self
        if dg_out < self.dg:
            raise ValueError("Requested dg_out < current dg; upsampling not supported.")

        data = self.copy(empty=False) if not inplace else self
        dg_inc = dg_out - data.dg
        if dg_inc > 0:
            data.array = self._downsample_tensor(data.array, dg_inc)
            data.dg = dg_out

        # Optionally apply a mask at the target resolution (simple multiplicative mask)
        if mask_MR is not None:
            mask = self._get_mask_at_dg(mask_MR, data.dg)
            if mask.shape[-2:] != data.array.shape[-2:]:
                raise ValueError("Mask and data have incompatible spatial shapes.")
            data.array = data.array * mask
        return data
    
    ###########################################################################
    def downsample_toMR(self, dg_max, mask_MR=None):
        """
        Generate a MR (multi-resolution) object by downsampling the current
        (single-resolution) data to all resolutions between dg=0 and dg_max.

        Only supports MR=False and assumes current dg==0.
        """
        if self.MR:
            raise ValueError("downsample_toMR expects MR == False.")
        if self.dg != 0:
            raise ValueError("downsample_toMR assumes current data is at dg=0.")
        if dg_max < 0:
            raise ValueError("dg_max must be non-negative.")
        if self.array is None:
            raise ValueError("No array stored in this object.")

        list_arrays = []
        list_dg = list(range(dg_max + 1))

        for dg in list_dg:
            if dg == 0:
                arr = self.array
            else:
                arr = self._downsample_tensor(self.array, dg)

            if mask_MR is not None:
                mask = self._get_mask_at_dg(mask_MR, dg)
                if mask.shape[-2:] != arr.shape[-2:]:
                    raise ValueError(f"Mask and data have incompatible shapes at dg={dg}.")
                arr = arr * mask

            list_arrays.append(arr)

        data = self.copy(empty=True)
        data.MR = True
        data.dg = None
        data.list_dg = list_dg
        data.array = list_arrays

        return data
    
    ###########################################################################
    def downsample_fromMR(self, Nout):
        """
        Convert an MR==True object to MR==False at resolution Nout.

        Each resolution in the current MR list is downsampled to Nout and then
        stacked into a single array of shape (..., len(list_dg), *Nout).
        """
        if not self.MR:
            raise ValueError("downsample_fromMR expects MR == True.")
        if self.array is None or len(self.array) == 0:
            raise ValueError("No data stored in this MR object.")
        if not isinstance(Nout, (tuple, list)) or len(Nout) != 2:
            raise ValueError("Nout must be a tuple (Nx_out, Ny_out).")

        Nx_out, Ny_out = Nout
        out_list = []

        for arr in self.array:
            H, W = arr.shape[-2:]
            if (H, W) == (Nx_out, Ny_out):
                y = arr
            else:
                if H % Nx_out != 0 or W % Ny_out != 0:
                    raise ValueError(f"Cannot downsample from ({H},{W}) to ({Nx_out},{Ny_out}).")
                factor_x = H // Nx_out
                factor_y = W // Ny_out
                if factor_x != factor_y:
                    raise ValueError("Anisotropic downsampling is not supported in downsample_fromMR.")
                dg_inc = int(round(math.log2(factor_x)))
                if 2 ** dg_inc != factor_x:
                    raise ValueError("Downsampling factor must be a power of 2.")
                y = self._downsample_tensor(arr, dg_inc)
            out_list.append(y)

        # stack along a new dimension before spatial dims
        stacked = torch.stack(out_list, dim=-3)

        data = self.copy(empty=True)
        data.MR = False
        data.array = stacked

        # infer dg from N0 and Nout if possible
        if self.N0 is not None:
            scale_x = self.N0[0] // Nx_out
            if scale_x > 0 and 2 ** int(round(math.log2(scale_x))) == scale_x:
                data.dg = int(round(math.log2(scale_x)))
            else:
                data.dg = None
        else:
            data.dg = None
        data.list_dg = None

        return data
    
    ###########################################################################
    def smooth(self, inplace=False):
        """Apply isotropic smoothing mirroring FoCUS.smooth 2D pathway."""

        target = self.copy(empty=False) if not inplace else self

        def _apply_smooth(tensor: torch.Tensor) -> torch.Tensor:
            *leading, Nx, Ny = tensor.shape
            ndata = int(torch.prod(torch.tensor(leading))) if leading else 1
            flat = tensor.reshape(ndata, Nx, Ny)
            smoothed = _complex_conv2d_circular(flat, self.smooth_kernel)
            return smoothed.reshape(*leading, Nx, Ny)

        if target.MR:
            target.array = [_apply_smooth(t) for t in target.array]
        else:
            target.array = _apply_smooth(target.array)

        target.dtype = target.array.dtype
        return target
    
    ###########################################################################
    def modulus(self, inplace=False):
        """
        Compute the modulus (absolute value) of the data.
        """
        data = self.copy(empty=False) if not inplace else self

        if data.MR:
            data.array = [torch.abs(a) for a in data.array]
        else:
            data.array = torch.abs(data.array)
            
        data.dtype=data.array.dtype

        return data
        
    ###########################################################################
    def mean(self, square=False, mask_MR=None):
        '''
        Compute the mean on the last two dimensions (Nx, Ny).

        If MR=True, the mean is computed for each resolution and stacked in
        an additional last dimension of size len(list_dg).

        If a multi-resolution mask is given, it is assumed to have unit mean
        at each resolution (as enforced by downsample_toMR_Mask), so the mean
        is computed as mean(x * mask).
        '''
        if self.MR:
            means = []
            for arr, dg in zip(self.array, self.list_dg):
                arr_use = torch.abs(arr) ** 2 if square else arr
                dims = (-2, -1)
                if mask_MR is not None:
                    mask = self._get_mask_at_dg(mask_MR, dg)
                    mean = (arr_use * mask).mean(dim=dims)
                else:
                    mean = arr_use.mean(dim=dims)
                means.append(mean)
            mean = torch.stack(means, dim=-1)
        else:
            if self.array is None:
                raise ValueError("No data stored in this object.")
            arr_use = torch.abs(self.array) ** 2 if square else self.array
            dims = (-2, -1)
            if mask_MR is not None:
                mask = self._get_mask_at_dg(mask_MR, self.dg)
                mean = (arr_use * mask).mean(dim=dims)
            else:
                mean = arr_use.mean(dim=dims)

        return mean
        
    ###########################################################################
    def cov(self, data2=None, mask_MR=None, remove_mean=False):
        """
        Covariance on the spatial dimensions while preserving orientation axes.

        The input arrays are expected to have spatial dimensions as the last
        two axes. If an orientation/channel axis exists, it is assumed to be at
        ``-3``; if not present, a singleton axis is inserted so that the output
        keeps explicit orientation indices. This mirrors the FoCUS behavior
        where covariances are computed per-orientation pair.

        Only works when MR == False.
        """
        if self.MR:
            raise ValueError("cov currently supports only MR == False.")

        x = self.array
        if data2 is None:
            y = x
        else:
            if not isinstance(data2, STL_2D_Kernel_Torch):
                raise TypeError("data2 must be a Planar2D_kernel_torch instance.")
            if data2.MR:
                raise ValueError("data2 must have MR == False.")
            if data2.dg != self.dg:
                raise ValueError("data2 must have the same dg as self.")
            y = data2.array

        # Ensure an explicit orientation axis just before the spatial axes
        def _ensure_orient(t: torch.Tensor) -> torch.Tensor:
            if t.dim() < 2:
                raise ValueError("Inputs to cov must have at least 2 spatial dims.")
            if t.dim() == 2:  # [Nx, Ny]
                return t.unsqueeze(0)
            if t.dim() == 3:  # [..., Nx, Ny] with no orientation
                return t.unsqueeze(-3)
            return t

        x_o = _ensure_orient(x)
        y_o = _ensure_orient(y)

        spatial_dims = (-2, -1)

        if mask_MR is not None:
            mask = self._get_mask_at_dg(mask_MR, self.dg)
            mask = _ensure_orient(mask)
            if remove_mean:
                mx = (x_o * mask).mean(dim=spatial_dims, keepdim=True)
                my = (y_o * mask).mean(dim=spatial_dims, keepdim=True)
                x_c = x_o - mx
                y_c = y_o - my
            else:
                x_c = x_o
                y_c = y_o
            prod = x_c.unsqueeze(-3) * y_c.conj().unsqueeze(-4) * mask.unsqueeze(-3)
        else:
            if remove_mean:
                mx = x_o.mean(dim=spatial_dims, keepdim=True)
                my = y_o.mean(dim=spatial_dims, keepdim=True)
                x_c = x_o - mx
                y_c = y_o - my
            else:
                x_c = x_o
                y_c = y_o
            prod = x_c.unsqueeze(-3) * y_c.conj().unsqueeze(-4)

        cov = prod.mean(dim=spatial_dims)

        return cov
       
    def get_wavelet_op(self, J=None, L=None, kernel_size=None):
        if L is None:
            L = 4
        if kernel_size is None:
            kernel_size = 5
        if J is None:
            J = np.min([int(np.log2(self.N0[0])),int(np.log2(self.N0[1]))])-2
        
        return WavelateOperator2Dkernel_torch(kernel_size,L,J,
                                              device=self.array.device,dtype=self.array.dtype)
       

class WavelateOperator2Dkernel_torch:
    def __init__(self, kernel_size: int, L: int, J: int,
                 device='cuda', dtype=torch.float):
        self.KERNELSZ = kernel_size
        self.L = L
        self.J = J
        self.device = torch.device(device)
        self.dtype = dtype

        self.kernel, real_kernel, imag_kernel, smooth_kernel = self._wavelet_kernel(
            kernel_size, L
        )
        self.WType='simple'
        
    def _wavelet_kernel(self, kernel_size: int, n_orientation: int):
        """FoCUS CNNV1 planar wavelet construction (cos/sin over Gaussian)."""

        KERNELSZ = kernel_size
        NORIENT = n_orientation
        LAMBDA = 1.0

        # Allocate real/imag components using a numpy dtype compatible with the torch dtype
        if self.dtype in (torch.float64, torch.complex128):
            np_dtype = np.float64
        elif self.dtype in (torch.float32, torch.complex64):
            np_dtype = np.float32
        elif self.dtype == torch.float16:
            np_dtype = np.float16
        elif self.dtype == torch.bfloat16:
            # numpy has limited bfloat16 support; fall back to float32 for kernel construction
            np_dtype = np.float32
        else:
            np_dtype = np.float32

        wwc = np.zeros([NORIENT, KERNELSZ * KERNELSZ], dtype=np_dtype)
        wws = np.zeros_like(wwc)

        x = np.repeat(np.arange(KERNELSZ) - KERNELSZ // 2, KERNELSZ).reshape(
            KERNELSZ, KERNELSZ
        )
        y = x.T

        if NORIENT == 1:
            xx = (3.0 / float(KERNELSZ)) * LAMBDA * x
            yy = (3.0 / float(KERNELSZ)) * LAMBDA * y

            if KERNELSZ == 5:
                w_smooth = np.exp(-(xx**2 + yy**2))
                tmp = np.exp(-2 * (xx**2 + yy**2)) - 0.25 * np.exp(
                    -0.5 * (xx**2 + yy**2)
                )
            else:
                w_smooth = np.exp(-0.5 * (xx**2 + yy**2))
                tmp = np.exp(-2 * (xx**2 + yy**2)) - 0.25 * np.exp(
                    -0.5 * (xx**2 + yy**2)
                )

            wwc[0] = tmp.flatten() - tmp.mean()
            wws[0] = np.zeros_like(wwc[0])
            sigma = np.sqrt((wwc[:, 0] ** 2).mean())
            wwc[0] /= sigma
            wws[0] /= sigma

            w_smooth = w_smooth.flatten()
        else:
            for i in range(NORIENT):
                a = (NORIENT - 1 - i) / float(NORIENT) * np.pi
                if KERNELSZ < 5:
                    xx = (3.0 / float(KERNELSZ)) * LAMBDA * (
                        x * np.cos(a) + y * np.sin(a)
                    )
                    yy = (3.0 / float(KERNELSZ)) * LAMBDA * (
                        x * np.sin(a) - y * np.cos(a)
                    )
                else:
                    xx = (3.0 / 5.0) * LAMBDA * (x * np.cos(a) + y * np.sin(a))
                    yy = (3.0 / 5.0) * LAMBDA * (x * np.sin(a) - y * np.cos(a))

                if KERNELSZ == 5:
                    w_smooth = np.exp(-2 * ((3.0 / float(KERNELSZ) * xx) ** 2 + (3.0 / float(KERNELSZ) * yy) ** 2))
                else:
                    w_smooth = np.exp(-0.5 * (xx**2 + yy**2))

                tmp1 = np.cos(yy * np.pi) * w_smooth
                tmp2 = np.sin(yy * np.pi) * w_smooth

                wwc[i] = tmp1.flatten() - tmp1.mean()
                wws[i] = tmp2.flatten() - tmp2.mean()
                sigma = np.mean(w_smooth)
                wwc[i] /= sigma
                wws[i] /= sigma

                w_smooth = w_smooth.flatten()

        w_smooth = w_smooth / w_smooth.sum()

        # Real/imaginary kernels for the primary convolution path (Cin=1)
        real_kernel = torch.tensor(
            wwc.reshape(NORIENT, 1, KERNELSZ, KERNELSZ), device=self.device, dtype=self.dtype
        )
        imag_kernel = torch.tensor(
            wws.reshape(NORIENT, 1, KERNELSZ, KERNELSZ), device=self.device, dtype=self.dtype
        )

        # Low-pass smoothing window (depthwise)
        smooth_kernel = torch.tensor(
            w_smooth.reshape(1, 1, KERNELSZ, KERNELSZ), device=self.device, dtype=self.dtype
        )

        # Orientation-expanded kernels for the second order (Cin=NORIENT, Cout=NORIENT*NORIENT)
        def doorientw(x: np.ndarray) -> np.ndarray:
            y = np.zeros(
                [NORIENT * NORIENT, NORIENT, KERNELSZ, KERNELSZ], dtype=np_dtype
            )
            for k in range(NORIENT):
                start = k * NORIENT
                y[start : start + NORIENT, k, :, :] = x.reshape(NORIENT, KERNELSZ, KERNELSZ)
            return y

        orient_real = torch.tensor(doorientw(wwc), device=self.device, dtype=self.dtype)
        orient_imag = torch.tensor(doorientw(wws), device=self.device, dtype=self.dtype)

        oriented_kernel = torch.complex(orient_real, orient_imag)

        # Complex kernel packed for convenience
        kernel = torch.complex(real_kernel, imag_kernel)

        # Keep both first-order and oriented kernels
        self.ww_RealT = [None, real_kernel, orient_real]
        self.ww_ImagT = [None, imag_kernel, orient_imag]
        self.ww_SmoothT = [None, smooth_kernel]

        self.oriented_kernel = oriented_kernel

        return kernel, real_kernel, imag_kernel, smooth_kernel

    def _bk_resize_image(self, im: torch.Tensor, noutx: int, nouty: int) -> torch.Tensor:
        """Torch bilinear resize mirroring FoCUS.backend.bk_resize_image."""
        *leading, hx, hy = im.shape
        flat = im.reshape(-1, 1, hx, hy)
        resized = F.interpolate(flat, size=(noutx, nouty), mode="bilinear", align_corners=False)
        return resized.reshape(*leading, noutx, nouty)

    def up_grade(self, im: torch.Tensor, nout: int, axis: int = -1, nouty: int = None) -> torch.Tensor:
        if nouty is None:
            nouty = nout
        return self._bk_resize_image(im, nout, nouty)

    def convol(self, in_image: torch.Tensor, use_oriented: bool = False) -> torch.Tensor:
        """FoCUS-like convolution with symmetric padding and complex kernels."""

        image = in_image.to(dtype=self.kernel.dtype, device=self.kernel.device)
        ishape = list(image.shape)
        if len(ishape) < 2:
            raise ValueError("Use of 2D scat with data that has less than 2D")

        # Ensure channel dimension is present
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(1)

        *leading, C, npix, npiy = image.shape
        ndata = int(np.prod(leading)) if leading else 1
        tim = image.reshape(ndata, C, npix, npiy)

        kernel_r = self.ww_RealT[2] if use_oriented else self.ww_RealT[1]
        kernel_i = self.ww_ImagT[2] if use_oriented else self.ww_ImagT[1]

        if torch.is_complex(tim):
            rr1 = _conv2d_same_symmetric(torch.real(tim), kernel_r)
            ii1 = _conv2d_same_symmetric(torch.real(tim), kernel_i)
            rr2 = _conv2d_same_symmetric(torch.imag(tim), kernel_r)
            ii2 = _conv2d_same_symmetric(torch.imag(tim), kernel_i)
            res = torch.complex(rr1 - ii2, ii1 + rr2)
        else:
            rr = _conv2d_same_symmetric(tim, kernel_r)
            ii = _conv2d_same_symmetric(tim, kernel_i)
            res = torch.complex(rr, ii)

        return res.reshape(*leading, kernel_r.shape[0], npix, npiy)

    def smooth(self, in_image: torch.Tensor) -> torch.Tensor:
        image = in_image.to(dtype=self.kernel.dtype, device=self.kernel.device)
        ishape = list(image.shape)
        if len(ishape) < 2:
            raise ValueError("Use of 2D scat with data that has less than 2D")

        npix = ishape[-2]
        npiy = ishape[-1]
        ndata = int(np.prod(ishape[:-2])) if len(ishape) > 2 else 1

        tim = image.reshape(ndata, 1, npix, npiy)

        if torch.is_complex(tim):
            rr = _conv2d_same_symmetric(torch.real(tim), self.ww_SmoothT[1])
            ii = _conv2d_same_symmetric(torch.imag(tim), self.ww_SmoothT[1])
            res = torch.complex(rr, ii)
        else:
            res = _conv2d_same_symmetric(tim, self.ww_SmoothT[1])

        return res.reshape(*ishape[:-2], npix, npiy)

    def ud_grade_2(self, im: torch.Tensor) -> torch.Tensor:
        ishape = list(im.shape)
        if len(ishape) < 2:
            raise ValueError("Use of 2D scat with data that has less than 2D")
        npix = ishape[-2]
        npiy = ishape[-1]
        if npix % 2 != 0 or npiy % 2 != 0:
            raise ValueError("Downsampling requires even spatial dimensions")

        ndata = 1
        for k in range(len(im.shape) - 2):
            ndata *= ishape[k]

        tim = im.reshape(ndata, npix, npiy, 1).permute(0, 3, 1, 2)
        res = F.avg_pool2d(tim, kernel_size=2, stride=2)
        res = res.permute(0, 2, 3, 1)
        return res.reshape(ishape[0:-2] + [npix // 2, npiy // 2])

    def scattering(self, image1: torch.Tensor):
        """
        Compute scattering coefficients (S0, S1, S2, S2L) following FoCUS CNNV1
        planar backend. Masking and normalization are intentionally omitted.
        """

        if image1.dim() == 2:
            I1 = image1.unsqueeze(0)
        else:
            I1 = image1

        # Add explicit channel dimension for convolutions
        if I1.dim() == 3:
            I1 = I1.unsqueeze(1)

        im_shape = I1.shape
        nside = min(im_shape[-2], im_shape[-1])
        jmax = int(math.log(nside - self.KERNELSZ) / math.log(2))

        if self.KERNELSZ > 3:
            if self.KERNELSZ == 5:
                l_image1 = self.up_grade(I1, I1.shape[-2] * 2, nouty=I1.shape[-1] * 2)
            else:
                l_image1 = self.up_grade(I1, I1.shape[-2] * 4, nouty=I1.shape[-1] * 4)
        else:
            l_image1 = I1

        s0 = l_image1.mean(dim=(-2, -1), keepdim=False)
        p00 = None
        s1 = None
        s2 = None
        s2l = None
        l2_image = None
        s2j1 = []
        s2j2 = []

        for j1 in range(jmax):
            c_image1 = self.convol(l_image1)

            conj = c_image1 * torch.conj(c_image1)
            l_p00 = conj.mean(dim=(-2, -1)).unsqueeze(-2)
            conj_mod = torch.abs(conj)
            l_s1 = conj_mod.mean(dim=(-2, -1)).unsqueeze(-2)

            if s1 is None:
                s1 = l_s1
                p00 = l_p00
            else:
                s1 = torch.cat([s1, l_s1], dim=-2)
                p00 = torch.cat([p00, l_p00], dim=-2)

            if l2_image is None:
                l2_image = conj_mod.unsqueeze(1)
            else:
                l2_image = torch.cat([l2_image, conj_mod.unsqueeze(1)], dim=1)

            # Positive path
            l2_pos = F.relu(l2_image)
            pos_flat = l2_pos.reshape(-1, self.L, l2_pos.shape[-2], l2_pos.shape[-1])
            c2_image = self.convol(pos_flat, use_oriented=True).reshape(
                *l2_pos.shape[:-3], self.L * self.L, l2_pos.shape[-2], l2_pos.shape[-1]
            )
            conj2p = c2_image * torch.conj(c2_image)
            conj2pl1 = torch.abs(conj2p)

            # Negative path
            l2_neg = F.relu(-l2_image)
            neg_flat = l2_neg.reshape(-1, self.L, l2_neg.shape[-2], l2_neg.shape[-1])
            c2_image_m = self.convol(neg_flat, use_oriented=True).reshape(
                *l2_neg.shape[:-3], self.L * self.L, l2_neg.shape[-2], l2_neg.shape[-1]
            )
            conj2m = c2_image_m * torch.conj(c2_image_m)
            conj2ml1 = torch.abs(conj2m)

            l_s2 = (conj2p - conj2m).mean(dim=(-2, -1))
            l_s2l1 = (conj2pl1 - conj2ml1).mean(dim=(-2, -1))

            if s2 is None:
                s2l = l_s2
                s2 = l_s2l1
                s2j1 = list(range(l_s2.shape[-3]))
                s2j2 = [j1] * l_s2.shape[-3]
            else:
                s2 = torch.cat([s2, l_s2l1], dim=-3)
                s2l = torch.cat([s2l, l_s2], dim=-3)
                s2j1.extend(list(range(l_s2.shape[-3])))
                s2j2.extend([j1] * l_s2.shape[-3])

            if j1 != jmax - 1:
                l2_image = self.smooth(l2_image)
                l2_image = self.ud_grade_2(l2_image)
                l_image1 = self.smooth(l_image1)
                l_image1 = self.ud_grade_2(l_image1)

        return {
            "s0": s0,
            "p00": p00,
            "s1": s1,
            "s2": s2,
            "s2l": s2l,
            "s2j1": torch.as_tensor(s2j1, device=s0.device),
            "s2j2": torch.as_tensor(s2j2, device=s0.device),
        }
            
    def get_L(self):
        return self.L
        
    def apply(self, data,j):
        """
        Apply the convolution kernel to data.array [..., Nx, Ny]
        and return cdata [..., L, Nx, Ny].

        Parameters
        ----------
        data : object
            Object with an attribute `array` storing the data as a tensor
            or numpy array with shape [..., Nx, Ny].

        Returns
        -------
        torch.Tensor
            Convolved data with shape [..., L, Nx, Ny].
        """
        if j!=data.dg :
            raise 'j is not equal to dg, convolution not possible'
            
        x = data.array  # [..., Nx, Ny]

        # Ensure x is a torch tensor on the same device / dtype as the kernel
        x = torch.as_tensor(x, device=self.kernel.device, dtype=self.kernel.dtype)

        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        # For the operator pathway, always use the base kernel; depthwise
        # grouping inside the convolution keeps per-orientation channels
        # independent when more than one channel is present.
        weight = self.kernel  # [L, 1, K, K]

        convolved = _complex_conv2d_same_symmetric(x, weight)

        return STL_2D_Kernel_Torch(convolved,smooth_kernel=data.smooth_kernel,dg=data.dg,N0=data.N0)
        
    
    def apply_smooth(self, data: STL_2D_Kernel_Torch, inplace: bool = False):
        """
        Smooth the data by convolving with a smooth kernel derived from the
        wavelet orientation 0. The data shape is preserved.

        Parameters
        ----------
        data : STL_Healpix_Kernel_Torch
            Input Healpix data with array of shape [..., K] and cell_ids aligned.
        copy : bool
            If True, return a new STL_Healpix_Kernel_Torch instance.
            If False, modify the input object in-place and return it.

        Returns
        -------
        STL_Healpix_Kernel_Torch
            Smoothed data object with same shape as input (no extra L dimension).
        """
        x = data.array  # [..., K]=
        *leading, K1, K2 = x.shape

        # Flatten leading dims into batch dimension: (B, Ci=1, K)
        if leading:
            B = int(np.prod(leading))
        else:
            B = 1
        x_bc = x.reshape(B, 1, K1, K2)

        # Smooth kernel (Ci=1, Co=1, P)
        w_smooth = self.ww_SmoothT[1].to(device=data.device, dtype=data.dtype)

        y_bc = _conv2d_circular(x_bc, w_smooth)

        if not isinstance(y_bc, torch.Tensor):
            y_bc = torch.as_tensor(y_bc, device=data.device, dtype=data.dtype)

        y = y_bc.reshape(*leading, K1, K2)  # same shape as input x

        # Copy or in-place update
        out = data.copy(empty=True) if not inplace else data
        out.array = y
        # metadata stays identical (nside, N0, dg, cell_ids, ...)
        return out
