# IMPORTS Standard
import math
import numpy as np

# IMPORTS Internal
from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract
from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure
from sfft.utils.PatternRotationCalculator import PatternRotation_Calculator
from sfft.utils.ReadWCS import Read_WCS
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator, KERNEL_CSZ, KERNEL_CSZ_INV
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel

__last_update__ = "2026-05-20"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>, Michael Wood-Vasey <wmwv@pitt.edu>, and VS Code Copilot"


class Numpy_FFTKits:
    @staticmethod
    def KERNEL_CSZ(KERNEL, NX_IMG, NY_IMG, NORMALIZE_KERNEL=False):
        """Circular shift the kernel and extend to the target size."""
        N0, N1 = NX_IMG, NY_IMG
        L0, L1 = KERNEL.shape
        W0, W1 = (L0 - 1) // 2, (L1 - 1) // 2
        assert L0 % 2 == 1 and L1 % 2 == 1

        if NORMALIZE_KERNEL:
            KERNEL_TZP = np.pad(KERNEL / np.sum(KERNEL),
                                pad_width=((0, N0 - L0), (0, N1 - L1)),
                                mode='constant', constant_values=0.)
        else:
            KERNEL_TZP = np.pad(KERNEL,
                                pad_width=((0, N0 - L0), (0, N1 - L1)),
                                mode='constant', constant_values=0.)
        KIMG_CSZ = np.roll(np.roll(KERNEL_TZP, -W0, axis=0), -W1, axis=1)
        return KIMG_CSZ

    @staticmethod
    def KERNEL_CSZ_INV(KIMG, NX_KERN, NY_KERN, VERBOSE_LEVEL=2):
        """Inverse circular shift the kernel and truncate to the target size."""
        L0, L1 = NX_KERN, NY_KERN
        W0, W1 = (L0 - 1) // 2, (L1 - 1) // 2
        assert L0 % 2 == 1 and L1 % 2 == 1

        KIMG_iCSZ = np.roll(np.roll(KIMG, W1, axis=1), W0, axis=0)
        KERNEL = KIMG_iCSZ[:L0, :L1]
        if VERBOSE_LEVEL in [1, 2]:
            LOSE_RATIO = 1. - np.sum(np.abs(KERNEL)) / np.sum(np.abs(KIMG_iCSZ))
            _report_message = "Kernel Truncation Loses APE = [%.4f %s]" % (LOSE_RATIO * 100, '%')
            print("MeLOn CheckPoint: %s " % _report_message)
        return KERNEL

    @staticmethod
    def FFT_CONVOLVE(PixA_Inp, KERNEL, PAD_FILL_VALUE=0., NAN_FILL_VALUE=0.,
                     NORMALIZE_KERNEL=False, FORCE_OUTPUT_C_CONTIGUOUS=False,
                     FFT_BACKEND="Numpy"):
        """FFT convolution with zero padding."""
        N0, N1 = PixA_Inp.shape
        L0, L1 = KERNEL.shape
        assert L0 % 2 == 1 and L1 % 2 == 1

        W0, W1 = (L0 - 1) // 2, (L1 - 1) // 2
        NX_IMG, NY_IMG = N0 + 2 * W0, N1 + 2 * W1

        PixA_EInp = np.pad(PixA_Inp,
                           pad_width=((W0, W0), (W1, W1)),
                           mode='constant', constant_values=PAD_FILL_VALUE)

        if NAN_FILL_VALUE is not None:
            PixA_EInp[np.isnan(PixA_EInp)] = NAN_FILL_VALUE

        KIMG_CSZ = Numpy_FFTKits.KERNEL_CSZ(KERNEL=KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG,
                                           NORMALIZE_KERNEL=NORMALIZE_KERNEL)

        PixA_Out = np.fft.ifft2(np.fft.fft2(PixA_EInp) * np.fft.fft2(KIMG_CSZ)).real[W0:-W0, W1:-W1]

        if FORCE_OUTPUT_C_CONTIGUOUS and not PixA_Out.flags['C_CONTIGUOUS']:
            PixA_Out = np.ascontiguousarray(PixA_Out)
        return PixA_Out


class Numpy_Resampling:
    def __init__(self, RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=2):
        self.RESAMP_METHOD = RESAMP_METHOD
        self.VERBOSE_LEVEL = VERBOSE_LEVEL

    def resamp_projection_astropy(self, hdr_obj, hdr_targ):
        """Project the target pixel centers to the object frame using Astropy WCS."""
        NTX = int(hdr_targ["NAXIS1"])
        NTY = int(hdr_targ["NAXIS2"])

        XX_targ, YY_targ = np.meshgrid(np.arange(1, NTX + 1, dtype=np.float64),
                                       np.arange(1, NTY + 1, dtype=np.float64), indexing='ij')
        XY_targ = np.vstack((XX_targ.ravel(), YY_targ.ravel())).T

        w_targ = Read_WCS.RW(hdr_targ, VERBOSE_LEVEL=self.VERBOSE_LEVEL)
        w_obj = Read_WCS.RW(hdr_obj, VERBOSE_LEVEL=self.VERBOSE_LEVEL)

        XY_world = w_targ.all_pix2world(XY_targ, 1)
        XY_proj = w_obj.all_world2pix(XY_world, 1)

        XX_proj = XY_proj[:, 0].reshape((NTX, NTY))
        YY_proj = XY_proj[:, 1].reshape((NTX, NTY))
        return XX_proj, YY_proj

    def frame_extension(self, XX_proj, YY_proj, PixA_obj,
                        PAD_FILL_VALUE=0., NAN_FILL_VALUE=0.):
        """Extend the object image to cover the projected coordinates."""
        NTX, NTY = XX_proj.shape
        NOX, NOY = PixA_obj.shape

        if self.RESAMP_METHOD == 'BILINEAR':
            KERHW = (1, 1)
        else:
            raise ValueError("Unsupported RESAMP_METHOD '%s'" % self.RESAMP_METHOD)

        RMIN = math.floor(np.min(XX_proj)) - 1 - KERHW[0]
        RMAX = math.floor(np.max(XX_proj)) - 1 + KERHW[0]
        RPAD = (-min(RMIN, 0), max(RMAX - (NOX - 1), 0))

        CMIN = math.floor(np.min(YY_proj)) - 1 - KERHW[1]
        CMAX = math.floor(np.max(YY_proj)) - 1 + KERHW[1]
        CPAD = (-min(CMIN, 0), max(CMAX - (NOY - 1), 0))

        PAD_WIDTH = (RPAD, CPAD)
        PixA_Eobj = np.pad(PixA_obj, PAD_WIDTH, mode='constant', constant_values=PAD_FILL_VALUE)
        if NAN_FILL_VALUE is not None:
            PixA_Eobj[np.isnan(PixA_Eobj)] = NAN_FILL_VALUE

        XX_Eproj = XX_proj + PAD_WIDTH[0][0]
        YY_Eproj = YY_proj + PAD_WIDTH[1][0]

        EProjDict = {
            "NTX": NTX,
            "NTY": NTY,
            "NOX": NOX,
            "NOY": NOY,
            "NEOX": PixA_Eobj.shape[0],
            "NEOY": PixA_Eobj.shape[1],
            "KERHW": KERHW,
            "XX_Eproj": XX_Eproj,
            "YY_Eproj": YY_Eproj
        }
        return PixA_Eobj, EProjDict

    def resampling(self, PixA_Eobj, EProjDict, PIXEL_SCALE_FACTOR=1.,
                   CUDA_COMPILER='nvrtc', THREAD_PER_BLOCK=8,
                   USE_SHARED_MEMORY=False):
        """Bilinear resampling using NumPy."""
        NTX = EProjDict["NTX"]
        NTY = EProjDict["NTY"]
        XX_Eproj = EProjDict["XX_Eproj"]
        YY_Eproj = EProjDict["YY_Eproj"]

        PixA_resamp = np.zeros((NTX, NTY), dtype=np.float64)

        if self.RESAMP_METHOD == "BILINEAR":
            if USE_SHARED_MEMORY and self.VERBOSE_LEVEL in [1, 2]:
                print("MeLOn WARNING: Shared memory is not implemented for NumPy bilinear resampling!")

            R1 = np.floor(XX_Eproj).astype(np.int64) - 1
            C1 = np.floor(YY_Eproj).astype(np.int64) - 1
            R2 = R1 + 1
            C2 = C1 + 1

            DX = XX_Eproj - np.floor(XX_Eproj)
            DY = YY_Eproj - np.floor(YY_Eproj)

            W11 = (1. - DX) * (1. - DY)
            W12 = (1. - DX) * DY
            W21 = DX * (1. - DY)
            W22 = DX * DY

            PixA_resamp = (
                W11 * PixA_Eobj[R1, C1] +
                W12 * PixA_Eobj[R1, C2] +
                W21 * PixA_Eobj[R2, C1] +
                W22 * PixA_Eobj[R2, C2]
            )
            PixA_resamp *= PIXEL_SCALE_FACTOR
        else:
            raise ValueError("Unsupported RESAMP_METHOD '%s'" % self.RESAMP_METHOD)

        return PixA_resamp


class Numpy_ZoomRotate:
    @staticmethod
    def CZR(PixA_obj, ZOOM_SCALE_X=1., ZOOM_SCALE_Y=1., OUTSIZE_PARIRY_X='UNCHANGED', OUTSIZE_PARIRY_Y='UNCHANGED',
            PATTERN_ROTATE_ANGLE=0.0, RESAMP_METHOD='BILINEAR', PAD_FILL_VALUE=0., NAN_FILL_VALUE=0.,
            CUDA_COMPILER='nvrtc', THREAD_PER_BLOCK=8, USE_SHARED_MEMORY=False, VERBOSE_LEVEL=2):
        """Zoom and rotate an image using NumPy."""
        assert ZOOM_SCALE_X > 0.0
        assert ZOOM_SCALE_Y > 0.0
        assert 0.0 <= PATTERN_ROTATE_ANGLE < 360.0

        NAXIS1_ORI, NAXIS2_ORI = PixA_obj.shape

        def NPIX_ORI2ZOOMED(NPIX_ORI, ZOOM_SCALE, OUTSIZE_PARIRY):
            if OUTSIZE_PARIRY == 'UNCHANGED':
                OUTSIZE_UPARIRY = 'EVEN' if NPIX_ORI % 2 == 0 else 'ODD'
            elif OUTSIZE_PARIRY == 'ODD':
                OUTSIZE_UPARIRY = 'ODD'
            elif OUTSIZE_PARIRY == 'EVEN':
                OUTSIZE_UPARIRY = 'EVEN'
            else:
                raise ValueError("Invalid OUTSIZE_PARIRY value: %s" % OUTSIZE_PARIRY)

            HALFWIDTH_ORI = NPIX_ORI / 2.0
            if OUTSIZE_UPARIRY == 'EVEN':
                return 2 * math.ceil(HALFWIDTH_ORI / ZOOM_SCALE)
            return 2 * math.ceil((HALFWIDTH_ORI - ZOOM_SCALE / 2.0) / ZOOM_SCALE) + 1

        NAXIS1_ZOOMED = NPIX_ORI2ZOOMED(NPIX_ORI=NAXIS1_ORI, ZOOM_SCALE=ZOOM_SCALE_X,
                                         OUTSIZE_PARIRY=OUTSIZE_PARIRY_X)
        NAXIS2_ZOOMED = NPIX_ORI2ZOOMED(NPIX_ORI=NAXIS2_ORI, ZOOM_SCALE=ZOOM_SCALE_Y,
                                         OUTSIZE_PARIRY=OUTSIZE_PARIRY_Y)

        def BACKWARD_TRANSFORM(X_ROTATED, Y_ROTATED):
            CRPIX1_ZOOMED = 0.5 + NAXIS1_ZOOMED / 2.0
            CRPIX2_ZOOMED = 0.5 + NAXIS2_ZOOMED / 2.0

            ALPHA_INV = np.deg2rad(-PATTERN_ROTATE_ANGLE)
            ROTMAT_INV = np.array([
                [np.cos(ALPHA_INV), -np.sin(ALPHA_INV)],
                [np.sin(ALPHA_INV), np.cos(ALPHA_INV)]
            ], dtype=np.float64)

            COORD_ROTATED = np.vstack((X_ROTATED - CRPIX1_ZOOMED, Y_ROTATED - CRPIX2_ZOOMED))
            COORD_ZOOMED = ROTMAT_INV.dot(COORD_ROTATED)
            COORD_ZOOMED[0, :] += CRPIX1_ZOOMED
            COORD_ZOOMED[1, :] += CRPIX2_ZOOMED

            OFFSET_X = ZOOM_SCALE_X * (0.5 + NAXIS1_ZOOMED / 2.0) - (0.5 + NAXIS1_ORI / 2.0)
            OFFSET_Y = ZOOM_SCALE_Y * (0.5 + NAXIS2_ZOOMED / 2.0) - (0.5 + NAXIS2_ORI / 2.0)

            X_ORI = COORD_ZOOMED[0, :] * ZOOM_SCALE_X - OFFSET_X
            Y_ORI = COORD_ZOOMED[1, :] * ZOOM_SCALE_Y - OFFSET_Y
            return X_ORI, Y_ORI

        XX_targ, YY_targ = np.meshgrid(np.arange(1, NAXIS1_ZOOMED + 1, dtype=np.float64),
                                       np.arange(1, NAXIS2_ZOOMED + 1, dtype=np.float64), indexing='ij')

        X_ORI, Y_ORI = BACKWARD_TRANSFORM(X_ROTATED=XX_targ.ravel(), Y_ROTATED=YY_targ.ravel())
        XX_proj = X_ORI.reshape((NAXIS1_ZOOMED, NAXIS2_ZOOMED))
        YY_proj = Y_ORI.reshape((NAXIS1_ZOOMED, NAXIS2_ZOOMED))

        GWR = Numpy_Resampling(RESAMP_METHOD=RESAMP_METHOD, VERBOSE_LEVEL=VERBOSE_LEVEL)
        PixA_Eobj, EProjDict = GWR.frame_extension(XX_proj=XX_proj, YY_proj=YY_proj,
                                                   PixA_obj=PixA_obj,
                                                   PAD_FILL_VALUE=PAD_FILL_VALUE,
                                                   NAN_FILL_VALUE=NAN_FILL_VALUE)

        PIXEL_SCALE_FACTOR = ZOOM_SCALE_X * ZOOM_SCALE_Y
        PixA_resamp = GWR.resampling(PixA_Eobj=PixA_Eobj, EProjDict=EProjDict,
                                    PIXEL_SCALE_FACTOR=PIXEL_SCALE_FACTOR,
                                    USE_SHARED_MEMORY=USE_SHARED_MEMORY,
                                    THREAD_PER_BLOCK=THREAD_PER_BLOCK)
        return PixA_resamp


class SpaceSFFT_NumpyFlow:
    """Run a NumPy-based CPU workflow for SFFT subtraction."""

    def __init__(self, hdr_target, hdr_object,
                 target_skyrms, object_skyrms,
                 PixA_target, PixA_object,
                 PixA_targetVar, PixA_objectVar,
                 PixA_target_DMASK, PixA_object_DMASK,
                 PSF_target, PSF_object,
                 sci_is_target=True,
                 GKerHW=9, KerPolyOrder=2, BGPolyOrder=0, ConstPhotRatio=True,
                 Consider_Matching_Kernel=False,
                 NUM_CPU_THREADS_4SUBTRACT=8, NUMBA_CACHE=True,
                 GAIN=1.0, RANDOM_SEED=10086):

        assert PixA_target.flags['C_CONTIGUOUS']
        assert PixA_object.flags['C_CONTIGUOUS']
        assert PixA_target_DMASK.flags['C_CONTIGUOUS']
        assert PixA_object_DMASK.flags['C_CONTIGUOUS']
        assert PSF_target.flags['C_CONTIGUOUS']
        assert PSF_object.flags['C_CONTIGUOUS']

        self.hdr_target = hdr_target
        self.hdr_object = hdr_object

        self.target_skyrms = target_skyrms
        self.object_skyrms = object_skyrms

        self.PixA_target = PixA_target.astype(np.float64)
        self.PixA_object = PixA_object.astype(np.float64)

        self.PixA_targetVar = PixA_targetVar.astype(np.float64)
        self.PixA_objectVar = PixA_objectVar.astype(np.float64)

        self.PixA_target_DMASK = PixA_target_DMASK.astype(np.float64)
        self.PixA_object_DMASK = PixA_object_DMASK.astype(np.float64)

        self.PSF_target = PSF_target.astype(np.float64)
        self.PSF_object = PSF_object.astype(np.float64)

        self.sci_is_target = sci_is_target
        self.GKerHW = GKerHW
        self.KerPolyOrder = KerPolyOrder
        self.BGPolyOrder = BGPolyOrder
        self.ConstPhotRatio = ConstPhotRatio
        self.Consider_Matching_Kernel = Consider_Matching_Kernel
        self.NUM_CPU_THREADS_4SUBTRACT = NUM_CPU_THREADS_4SUBTRACT
        self.NUMBA_CACHE = NUMBA_CACHE
        self.GAIN = GAIN
        self.RANDOM_SEED = RANDOM_SEED

    def resampling_image_mask_psf(self):
        CR = Numpy_Resampling(RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=1)
        XX_proj, YY_proj = CR.resamp_projection_astropy(hdr_obj=self.hdr_object,
                                                       hdr_targ=self.hdr_target)

        NTX = int(self.hdr_target["NAXIS1"])
        NTY = int(self.hdr_target["NAXIS2"])
        NPIX_INNER = np.sum(np.logical_and(
            np.logical_and(XX_proj >= 0.5, XX_proj < NTX + 0.5),
            np.logical_and(YY_proj >= 0.5, YY_proj < NTY + 0.5)
        ))
        assert NPIX_INNER > 0, "SFFT Error: Projection of object image is completely outside of target image!"

        PixA_Eobj, EProjDict = CR.frame_extension(XX_proj=XX_proj,
                                                  YY_proj=YY_proj,
                                                  PixA_obj=self.PixA_object,
                                                  PAD_FILL_VALUE=0.,
                                                  NAN_FILL_VALUE=0.)

        self.PixA_resamp_object = CR.resampling(PixA_Eobj=PixA_Eobj,
                                               EProjDict=EProjDict,
                                               CUDA_COMPILER='nvrtc',
                                               USE_SHARED_MEMORY=False)

        PixA_EobjVar, _ = CR.frame_extension(XX_proj=XX_proj,
                                             YY_proj=YY_proj,
                                             PixA_obj=self.PixA_objectVar,
                                             PAD_FILL_VALUE=0.,
                                             NAN_FILL_VALUE=0.)

        self.PixA_resamp_objectVar = CR.resampling(PixA_Eobj=PixA_EobjVar,
                                                  EProjDict=EProjDict,
                                                  CUDA_COMPILER='nvrtc',
                                                  USE_SHARED_MEMORY=False)

        PixA_Eobj_mask, _ = CR.frame_extension(XX_proj=XX_proj,
                                               YY_proj=YY_proj,
                                               PixA_obj=self.PixA_object_DMASK,
                                               PAD_FILL_VALUE=0.,
                                               NAN_FILL_VALUE=0.)

        self.PixA_resamp_object_DMASK = CR.resampling(PixA_Eobj=PixA_Eobj_mask,
                                                      EProjDict=EProjDict,
                                                      CUDA_COMPILER='nvrtc',
                                                      USE_SHARED_MEMORY=False)

        self.BlankMask = self.PixA_resamp_object == 0.

        PATTERN_ROTATE_ANGLE = PatternRotation_Calculator.PRC(hdr_obj=self.hdr_object,
                                                             hdr_targ=self.hdr_target)

        self.PSF_resamp_object = Numpy_ZoomRotate.CZR(PixA_obj=self.PSF_object,
                                                      ZOOM_SCALE_X=1.,
                                                      ZOOM_SCALE_Y=1.,
                                                      OUTSIZE_PARIRY_X='UNCHANGED',
                                                      OUTSIZE_PARIRY_Y='UNCHANGED',
                                                      PATTERN_ROTATE_ANGLE=PATTERN_ROTATE_ANGLE,
                                                      RESAMP_METHOD='BILINEAR',
                                                      PAD_FILL_VALUE=0.,
                                                      NAN_FILL_VALUE=0.,
                                                      THREAD_PER_BLOCK=8,
                                                      USE_SHARED_MEMORY=False,
                                                      VERBOSE_LEVEL=2)

    def cross_convolution(self):
        self.PixA_Ctarget = Numpy_FFTKits.FFT_CONVOLVE(PixA_Inp=self.PixA_target,
                                                      KERNEL=self.PSF_resamp_object,
                                                      PAD_FILL_VALUE=0.,
                                                      NAN_FILL_VALUE=None,
                                                      NORMALIZE_KERNEL=True,
                                                      FORCE_OUTPUT_C_CONTIGUOUS=True,
                                                      FFT_BACKEND="Numpy")

        self.PSF_Ctarget = Numpy_FFTKits.FFT_CONVOLVE(PixA_Inp=self.PSF_target,
                                                      KERNEL=self.PSF_resamp_object,
                                                      PAD_FILL_VALUE=0.,
                                                      NAN_FILL_VALUE=None,
                                                      NORMALIZE_KERNEL=True,
                                                      FORCE_OUTPUT_C_CONTIGUOUS=True,
                                                      FFT_BACKEND="Numpy")

        self.PixA_Cresamp_object = Numpy_FFTKits.FFT_CONVOLVE(PixA_Inp=self.PixA_resamp_object,
                                                              KERNEL=self.PSF_target,
                                                              PAD_FILL_VALUE=0.,
                                                              NAN_FILL_VALUE=None,
                                                              NORMALIZE_KERNEL=True,
                                                              FORCE_OUTPUT_C_CONTIGUOUS=True,
                                                              FFT_BACKEND="Numpy")

    def sfft_subtraction(self):
        LYMASK_BKG = np.logical_or(self.PixA_target_DMASK == 0,
                                   self.PixA_resamp_object_DMASK < 0.1)

        NaNmask_Ctarget = np.isnan(self.PixA_Ctarget)
        NaNmask_Cresamp_object = np.isnan(self.PixA_Cresamp_object)
        if NaNmask_Ctarget.any() or NaNmask_Cresamp_object.any():
            NaNmask = np.logical_or(NaNmask_Ctarget, NaNmask_Cresamp_object)
            ZeroMask = np.logical_or(NaNmask, LYMASK_BKG)
        else:
            ZeroMask = LYMASK_BKG

        PixA_mCtarget = self.PixA_Ctarget.copy()
        PixA_mCtarget[ZeroMask] = 0.

        PixA_mCresamp_object = self.PixA_Cresamp_object.copy()
        PixA_mCresamp_object[ZeroMask] = 0.

        if self.sci_is_target:
            PixA_REF = self.PixA_Cresamp_object
            PixA_SCI = self.PixA_Ctarget
            PixA_mREF = PixA_mCresamp_object
            PixA_mSCI = PixA_mCtarget
            ForceConv = 'REF'
        else:
            PixA_REF = self.PixA_Ctarget
            PixA_SCI = self.PixA_Cresamp_object
            PixA_mREF = PixA_mCtarget
            PixA_mSCI = PixA_mCresamp_object
            ForceConv = 'SCI'

        assert PixA_REF.ndim == 2 and PixA_SCI.ndim == 2
        assert PixA_mREF.ndim == 2 and PixA_mSCI.ndim == 2

        NX, NY = PixA_REF.shape
        SFFTConfig = SingleSFFTConfigure.SSC(NX=NX, NY=NY, KerHW=self.GKerHW,
                                             KerPolyOrder=self.KerPolyOrder,
                                             BGPolyOrder=self.BGPolyOrder,
                                             ConstPhotRatio=self.ConstPhotRatio,
                                             BACKEND_4SUBTRACT='Numpy',
                                             NUM_CPU_THREADS_4SUBTRACT=self.NUM_CPU_THREADS_4SUBTRACT,
                                             NUMBA_CACHE=self.NUMBA_CACHE,
                                             VERBOSE_LEVEL=2)

        self.Solution, self.PixA_DIFF, _ = GeneralSFFTSubtract.GSS(
            PixA_I=PixA_REF,
            PixA_J=PixA_SCI,
            PixA_mI=PixA_mREF,
            PixA_mJ=PixA_mSCI,
            SFFTConfig=SFFTConfig,
            ContamMask_I=None,
            BACKEND_4SUBTRACT='Numpy',
            NUM_CPU_THREADS_4SUBTRACT=self.NUM_CPU_THREADS_4SUBTRACT,
            VERBOSE_LEVEL=2
        )
        self.PixA_DIFF[self.BlankMask] = 0.

    def find_decorrelation(self):
        N0, N1 = self.PixA_DIFF.shape
        L0, L1 = 2 * self.GKerHW + 1, 2 * self.GKerHW + 1
        DK = self.KerPolyOrder
        Fpq = int((self.BGPolyOrder + 1) * (self.BGPolyOrder + 2) / 2)
        XY_q = np.array([[N0 / 2. + 0.5, N1 / 2. + 0.5]])

        self.MATCH_KERNEL = Realize_MatchingKernel(XY_q=XY_q).FromArray(
            Solution=self.Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq
        )[0]

        if self.Consider_Matching_Kernel:
            MK = self.MATCH_KERNEL
        else:
            MK = None

        self.FKDECO = DeCorrelation_Calculator(NX_IMG=N0,
                                               NY_IMG=N1,
                                               KERNEL_JQueue=[self.PSF_resamp_object],
                                               BKGSIG_JQueue=[self.target_skyrms],
                                               KERNEL_IQueue=[self.PSF_target],
                                               BKGSIG_IQueue=[self.object_skyrms],
                                               MATCH_KERNEL=MK,
                                               REAL_OUTPUT=False,
                                               REAL_OUTPUT_SIZE=None,
                                               NORMALIZE_OUTPUT=True,
                                               VERBOSE_LEVEL=2)
        print("Decorrelation kernel calculated.")

    def apply_decorrelation(self, img):
        _img = np.asarray(img)
        if _img.shape == self.FKDECO.shape:
            FPixA = np.fft.fft2(_img)
            PixA_decorr = np.fft.ifft2(FPixA * self.FKDECO).real
            decorimg = np.array(PixA_decorr, dtype=np.float64)
        else:
            NK0, NK1 = _img.shape
            N0, N1 = self.FKDECO.shape
            KERN_CSZ = KERNEL_CSZ(KERNEL=_img, NX_IMG=N0, NY_IMG=N1)
            FKERN_decorr = np.fft.fft2(KERN_CSZ) * self.FKDECO
            PixA_KERN_decorr = KERNEL_CSZ_INV(np.fft.ifft2(FKERN_decorr).real,
                                              NX_KERN=NK0, NY_KERN=NK1)
            decorimg = np.array(PixA_KERN_decorr, dtype=np.float64)
        return decorimg

    def create_score_image(self):
        NX, NY = self.PixA_target.shape
        PSF_object_CSZ = Numpy_FFTKits.KERNEL_CSZ(KERNEL=self.PSF_object, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ = Numpy_FFTKits.KERNEL_CSZ(KERNEL=self.PSF_target, NX_IMG=NX, NY_IMG=NY)

        FPSF_dDIFF = np.fft.fft2(PSF_object_CSZ) * np.fft.fft2(PSF_target_CSZ) * self.FKDECO

        FPixA_DIFF = np.fft.fft2(self.PixA_DIFF)
        FPixA_dDIFF = FPixA_DIFF * self.FKDECO

        FPixA_SCORE = FPixA_dDIFF * np.conj(FPSF_dDIFF)
        PixA_SCORE = np.fft.ifft2(FPixA_SCORE).real

        skysig_SCORE = SkyLevel_Estimator.SLE(PixA_obj=PixA_SCORE)[1]
        PixA_SCORE /= skysig_SCORE
        return PixA_SCORE

    def create_variance_image(self):
        assert self.PixA_targetVar.flags['C_CONTIGUOUS']
        assert self.PixA_resamp_objectVar.flags['C_CONTIGUOUS']

        NX, NY = self.PixA_target.shape
        PSF_resamp_object_CSZ = Numpy_FFTKits.KERNEL_CSZ(KERNEL=self.PSF_resamp_object, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ = Numpy_FFTKits.KERNEL_CSZ(KERNEL=self.PSF_target, NX_IMG=NX, NY_IMG=NY)

        PixA_dDIFFVar = np.fft.ifft2(
            np.fft.fft2(self.PixA_resamp_objectVar) *
            np.fft.fft2((np.fft.ifft2(np.fft.fft2(PSF_target_CSZ) * self.FKDECO)).real ** 2)
        ).real
        PixA_dDIFFVar += np.fft.ifft2(
            np.fft.fft2(self.PixA_targetVar) *
            np.fft.fft2((np.fft.ifft2(np.fft.fft2(PSF_resamp_object_CSZ) * self.FKDECO)).real ** 2)
        ).real

        return PixA_dDIFFVar

    def cleanup(self):
        pass
