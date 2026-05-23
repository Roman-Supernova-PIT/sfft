# IMPORTS Standard
import numpy as np

# IMPORTS Internal
from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract
from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure
from sfft.utils.PatternRotationCalculator import PatternRotation_Calculator
from sfft.utils.NumpyFFTKits import Numpy_FFTKits
from sfft.utils.NumpyResampKits import Numpy_Resampling, Numpy_ZoomRotate
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator, KERNEL_CSZ, KERNEL_CSZ_INV
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel

__last_update__ = "2026-05-20"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>, Michael Wood-Vasey <wmwv@pitt.edu>, and VS Code Copilot"


class SpaceSFFT_NumpyFlow:
    """Run a NumPy-based CPU workflow for SFFT subtraction."""

    def __init__(
        self,
        hdr_target,
        hdr_object,
        target_skyrms,
        object_skyrms,
        PixA_target,
        PixA_object,
        PixA_targetVar,
        PixA_objectVar,
        PixA_target_DMASK,
        PixA_object_DMASK,
        PSF_target,
        PSF_object,
        sci_is_target=True,
        GKerHW=9,
        KerPolyOrder=2,
        BGPolyOrder=0,
        ConstPhotRatio=True,
        Consider_Matching_Kernel=False,
        NUM_CPU_THREADS_4SUBTRACT=8,
        NUMBA_CACHE=True,
        GAIN=1.0,
        RANDOM_SEED=10086,
        **kwargs
    ):

        assert PixA_target.flags["C_CONTIGUOUS"]
        assert PixA_object.flags["C_CONTIGUOUS"]
        assert PixA_target_DMASK.flags["C_CONTIGUOUS"]
        assert PixA_object_DMASK.flags["C_CONTIGUOUS"]
        assert PSF_target.flags["C_CONTIGUOUS"]
        assert PSF_object.flags["C_CONTIGUOUS"]

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
        XX_proj, YY_proj = CR.resamp_projection_astropy(hdr_obj=self.hdr_object, hdr_targ=self.hdr_target)

        NTX = int(self.hdr_target["NAXIS1"])
        NTY = int(self.hdr_target["NAXIS2"])
        NPIX_INNER = np.sum(
            np.logical_and(
                np.logical_and(XX_proj >= 0.5, XX_proj < NTX + 0.5), np.logical_and(YY_proj >= 0.5, YY_proj < NTY + 0.5)
            )
        )
        assert NPIX_INNER > 0, "SFFT Error: Projection of object image is completely outside of target image!"

        PixA_Eobj, EProjDict = CR.frame_extension(
            XX_proj=XX_proj, YY_proj=YY_proj, PixA_obj=self.PixA_object, PAD_FILL_VALUE=0.0, NAN_FILL_VALUE=0.0
        )

        self.PixA_resamp_object = CR.resampling(
            PixA_Eobj=PixA_Eobj, EProjDict=EProjDict, CUDA_COMPILER="nvrtc", USE_SHARED_MEMORY=False
        )

        PixA_EobjVar, _ = CR.frame_extension(
            XX_proj=XX_proj, YY_proj=YY_proj, PixA_obj=self.PixA_objectVar, PAD_FILL_VALUE=0.0, NAN_FILL_VALUE=0.0
        )

        self.PixA_resamp_objectVar = CR.resampling(
            PixA_Eobj=PixA_EobjVar, EProjDict=EProjDict, CUDA_COMPILER="nvrtc", USE_SHARED_MEMORY=False
        )

        PixA_Eobj_mask, _ = CR.frame_extension(
            XX_proj=XX_proj, YY_proj=YY_proj, PixA_obj=self.PixA_object_DMASK, PAD_FILL_VALUE=0.0, NAN_FILL_VALUE=0.0
        )

        self.PixA_resamp_object_DMASK = CR.resampling(
            PixA_Eobj=PixA_Eobj_mask, EProjDict=EProjDict, CUDA_COMPILER="nvrtc", USE_SHARED_MEMORY=False
        )

        self.BlankMask = self.PixA_resamp_object == 0.0

        PATTERN_ROTATE_ANGLE = PatternRotation_Calculator.PRC(hdr_obj=self.hdr_object, hdr_targ=self.hdr_target)

        self.PSF_resamp_object = Numpy_ZoomRotate.CZR(
            PixA_obj=self.PSF_object,
            ZOOM_SCALE_X=1.0,
            ZOOM_SCALE_Y=1.0,
            OUTSIZE_PARITY_X="UNCHANGED",
            OUTSIZE_PARITY_Y="UNCHANGED",
            PATTERN_ROTATE_ANGLE=PATTERN_ROTATE_ANGLE,
            RESAMP_METHOD="BILINEAR",
            PAD_FILL_VALUE=0.0,
            NAN_FILL_VALUE=0.0,
            THREAD_PER_BLOCK=8,
            USE_SHARED_MEMORY=False,
            VERBOSE_LEVEL=2,
        )

    def cross_convolution(self):
        self.PixA_Ctarget = Numpy_FFTKits.FFT_CONVOLVE(
            PixA_Inp=self.PixA_target,
            KERNEL=self.PSF_resamp_object,
            PAD_FILL_VALUE=0.0,
            NAN_FILL_VALUE=None,
            NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True,
            FFT_BACKEND="Numpy",
        )

        self.PSF_Ctarget = Numpy_FFTKits.FFT_CONVOLVE(
            PixA_Inp=self.PSF_target,
            KERNEL=self.PSF_resamp_object,
            PAD_FILL_VALUE=0.0,
            NAN_FILL_VALUE=None,
            NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True,
            FFT_BACKEND="Numpy",
        )

        self.PixA_Cresamp_object = Numpy_FFTKits.FFT_CONVOLVE(
            PixA_Inp=self.PixA_resamp_object,
            KERNEL=self.PSF_target,
            PAD_FILL_VALUE=0.0,
            NAN_FILL_VALUE=None,
            NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True,
            FFT_BACKEND="Numpy",
        )

    def sfft_subtraction(self):
        LYMASK_BKG = np.logical_or(self.PixA_target_DMASK == 0, self.PixA_resamp_object_DMASK < 0.1)

        NaNmask_Ctarget = np.isnan(self.PixA_Ctarget)
        NaNmask_Cresamp_object = np.isnan(self.PixA_Cresamp_object)
        if NaNmask_Ctarget.any() or NaNmask_Cresamp_object.any():
            NaNmask = np.logical_or(NaNmask_Ctarget, NaNmask_Cresamp_object)
            ZeroMask = np.logical_or(NaNmask, LYMASK_BKG)
        else:
            ZeroMask = LYMASK_BKG

        PixA_mCtarget = self.PixA_Ctarget.copy()
        PixA_mCtarget[ZeroMask] = 0.0

        PixA_mCresamp_object = self.PixA_Cresamp_object.copy()
        PixA_mCresamp_object[ZeroMask] = 0.0

        if self.sci_is_target:
            PixA_REF = self.PixA_Cresamp_object
            PixA_SCI = self.PixA_Ctarget
            PixA_mREF = PixA_mCresamp_object
            PixA_mSCI = PixA_mCtarget
            ForceConv = "REF"
        else:
            PixA_REF = self.PixA_Ctarget
            PixA_SCI = self.PixA_Cresamp_object
            PixA_mREF = PixA_mCtarget
            PixA_mSCI = PixA_mCresamp_object
            ForceConv = "SCI"

        assert PixA_REF.ndim == 2 and PixA_SCI.ndim == 2
        assert PixA_mREF.ndim == 2 and PixA_mSCI.ndim == 2

        NX, NY = PixA_REF.shape
        SFFTConfig = SingleSFFTConfigure.SSC(
            NX=NX,
            NY=NY,
            KerHW=self.GKerHW,
            KerPolyOrder=self.KerPolyOrder,
            BGPolyOrder=self.BGPolyOrder,
            ConstPhotRatio=self.ConstPhotRatio,
            BACKEND_4SUBTRACT="Numpy",
            NUM_CPU_THREADS_4SUBTRACT=self.NUM_CPU_THREADS_4SUBTRACT,
            NUMBA_CACHE=self.NUMBA_CACHE,
            VERBOSE_LEVEL=2,
        )

        self.Solution, self.PixA_DIFF, _ = GeneralSFFTSubtract.GSS(
            PixA_I=PixA_REF,
            PixA_J=PixA_SCI,
            PixA_mI=PixA_mREF,
            PixA_mJ=PixA_mSCI,
            SFFTConfig=SFFTConfig,
            ContamMask_I=None,
            BACKEND_4SUBTRACT="Numpy",
            NUM_CPU_THREADS_4SUBTRACT=self.NUM_CPU_THREADS_4SUBTRACT,
            VERBOSE_LEVEL=2,
        )
        self.PixA_DIFF[self.BlankMask] = 0.0

    def find_decorrelation(self):
        N0, N1 = self.PixA_DIFF.shape
        L0, L1 = 2 * self.GKerHW + 1, 2 * self.GKerHW + 1
        DK = self.KerPolyOrder
        Fpq = int((self.BGPolyOrder + 1) * (self.BGPolyOrder + 2) / 2)
        XY_q = np.array([[N0 / 2.0 + 0.5, N1 / 2.0 + 0.5]])

        self.MATCH_KERNEL = Realize_MatchingKernel(XY_q=XY_q).FromArray(
            Solution=self.Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq
        )[0]

        if self.Consider_Matching_Kernel:
            MK = self.MATCH_KERNEL
        else:
            MK = None

        self.FKDECO = DeCorrelation_Calculator(
            NX_IMG=N0,
            NY_IMG=N1,
            KERNEL_JQueue=[self.PSF_resamp_object],
            BKGSIG_JQueue=[self.target_skyrms],
            KERNEL_IQueue=[self.PSF_target],
            BKGSIG_IQueue=[self.object_skyrms],
            MATCH_KERNEL=MK,
            REAL_OUTPUT=False,
            REAL_OUTPUT_SIZE=None,
            NORMALIZE_OUTPUT=True,
            VERBOSE_LEVEL=2,
        )
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
            PixA_KERN_decorr = KERNEL_CSZ_INV(np.fft.ifft2(FKERN_decorr).real, NX_KERN=NK0, NY_KERN=NK1)
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
        assert self.PixA_targetVar.flags["C_CONTIGUOUS"]
        assert self.PixA_resamp_objectVar.flags["C_CONTIGUOUS"]

        NX, NY = self.PixA_target.shape
        PSF_resamp_object_CSZ = Numpy_FFTKits.KERNEL_CSZ(KERNEL=self.PSF_resamp_object, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ = Numpy_FFTKits.KERNEL_CSZ(KERNEL=self.PSF_target, NX_IMG=NX, NY_IMG=NY)

        PixA_dDIFFVar = np.fft.ifft2(
            np.fft.fft2(self.PixA_resamp_objectVar)
            * np.fft.fft2((np.fft.ifft2(np.fft.fft2(PSF_target_CSZ) * self.FKDECO)).real ** 2)
        ).real
        PixA_dDIFFVar += np.fft.ifft2(
            np.fft.fft2(self.PixA_targetVar)
            * np.fft.fft2((np.fft.ifft2(np.fft.fft2(PSF_resamp_object_CSZ) * self.FKDECO)).real ** 2)
        ).real

        return PixA_dDIFFVar

    def cleanup(self):
        pass
