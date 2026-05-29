import numpy as np

from sfft.sfftcore.SFFTSubtract import GeneralSFFTSubtract
from sfft.sfftcore.SFFTConfigure import SingleSFFTConfigure
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator, KERNEL_CSZ, KERNEL_CSZ_INV
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel

__last_update__ = "2025-05-28"
__author__ = "Lei Hu <leihu@andrew.cmu.edu> and Michael Wood-Vasey <wmwv@pitt.edu>"


class SpaceSFFT_Flow:
    """Run A Cupy or Numpy WorkFlow for SFFT subtraction"""

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
        BACKEND_4SUBTRACT="Cupy",
        CUDA_COMPILER="nvrtc",
        CUDA_DEVICE_4SUBTRACT="0",
        NUM_CPU_THREADS_4SUBTRACT=8,
        NUMBA_CACHE=True,
        RANDOM_SEED=10086,
    ):
        """Prepare inputs arrays and send to Cupy or Numpy based SFFT code for subtraction.

        Parameters
        ----------
        hdr_target : astropy header
            The target image has the coordinate system to which we are
            resampling.  This is that image's header.  SFFT will use the following keywords from the header:
                * All WCS keywords (including LONPOLE)
                * NAXIS1
                * NAXIS2

        hdr_object : astropy header
            Original (unresampled) header of the image to be resampled to match target.

        target_skyrms : float

        object_skyrms : float

        PixA_target : numpy or cupy array (float64)
            2d image data of target, indexed by x, y.  (Note that raw
            data read from fits files are indexed y, x; transpose to
            get this.)

        PixA_object : numpy or cupy array (float64)
            2d image data of original image, indexed by x, y.

        PixA_targetVar : numpy or cupy array (float64)
            2d image variance of original image, indexed by x, y.

        PixA_objectVar : numpy or cupy array (float64)
            2d image variance of original image, indexed by x, y.

        PixA_target_DMASK : numpy or cupy array (bool)
            2d detection mask for target image

        PixA_object_DMASK : numpy or cupy array (bool)
            2d detection mask for unresampled object image

        PSF_target : numpy or cupy array (float64)
            2d PSF model; axis lengths must be odd.  center of PSF is
            center of center pixel.

        PSF_object : numpy or cupy array (float64)
            object PSF

        sci_is_target : bool
            If True, will subtract object - target.  If false, will subtract target - object.

        GKerHW : int
            Matching kernel half-width (full width is 2*GkerHW + 1 )

        KerPolyOrder : int
            Order of spatial variation in kernel

        BGPolyOrder : int
            Order of differential background 2d polynomial.  (Usually
            just leave this 0, we assume the image is sky subtracted.)

        ConstPhotRatio : bool
            Assume relative zeropoints of target and (resampled)
            object have no spatial variation.

        Consider_Matching_Kernel : bool, default False
            Whether to consider the matching kernel in the decorrelation.
            The matching kernel is close to a delta function doing a shift,
            by default, we may ignore it.

        BACKEND_4SUBTRACT : str, default 'Cupy'
            Which backend to use for subtraction.  Options are 'Cupy' and 'Numpy'.
            If 'Cupy', will use GPU acceleration for the subtraction step.
            If 'Numpy', will use CPU for subtraction (and Cupy does not have to be installed).

        CUDA_COMPILER : str, default 'nvrtc'
            Which CUDA compiler to use for compiling the custom CUDA kernels.  Options are 'nvrtc' and 'nvcc'.
            Ignore if Numpy backend is used.

        CUDA_DEVICE_4SUBTRACT : str, default '0'
            Which CUDA device to use.  The most likely reason to use this option is if you're running multiple SFFT processes
            on a multi-GPUe machine.  In that case you'll have to specify which GPU each SFFT process uses.
            Ignored if Numpy backend is used.

        RANDOM_SEED : int, default 10086
            Random seed to use for CR.resamp_projection_sip when inverting an SIP transformation.
        """

        # Dependent loads if we're Numpy or Cupy
        # We do this in the object initialization
        # so that in prinpciple we can have objects that are cupy and ones that are numpy
        # in the same running processing.  I don't think we'll want to run this way
        # but I don't want to debug accidentally doing this.
        if self.BACKEND_4SUBTRACT == "Cupy":

            class CupyOperations:
                def __init__(self):
                    import cupy as cp

                    self.fft = cp.fft
                    self.sum = cp.sum
                    self.logical_and = cp.logical_and
                    self.logical_or = cp.logical_or
                    self.isnan = cp.isnan
                    self.asnumpy = cp.asnumpy
                    self.array = cp.array
                    self.conj = cp.conj
                    self.require = cp.require

                    from sfft.PureCupyCustomizedPacket import PureCupy_Customized_Packet
                    from sfft.utils.PureCupyFFTKits import PureCupy_FFTKits
                    from sfft.utils.PatternRotationCalculator import PatternRotation_Calculator
                    from sfft.utils.ResampKits import Cupy_ZoomRotate
                    from sfft.utils.ResampKits import Cupy_Resampling

                    self.PCCP = PureCupy_Customized_Packet.PCCP
                    self.FFT_CONVOLVE = PureCupy_FFTKits.FFT_CONVOLVE
                    self.KERNEL_CSZ = PureCupy_FFTKits.KERNEL_CSZ
                    self.PRC = PatternRotation_Calculator.PRC
                    self.CZR = Cupy_ZoomRotate.CZR
                    self.Resampling = Cupy_Resampling

            self.op = CupyOperations()

        elif self.BACKEND_4SUBTRACT == "Numpy":

            class NumpyOperations:
                def __init__(self):
                    self.fft = np.fft
                    self.sum = np.sum
                    self.logical_and = np.logical_and
                    self.logical_or = np.logical_or
                    self.isnan = np.isnan
                    self.asnumpy = np.asarray
                    self.array = np.array
                    self.conj = np.conj
                    self.require = np.require

                    from sfft.utils.NumpyFFTKits import Numpy_FFTKits
                    from sfft.utils.PatternRotationCalculator import PatternRotation_Calculator
                    from sfft.utils.NumpyResampKits import Numpy_ZoomRotate
                    from sfft.utils.NumpyResampKits import Numpy_Resampling

                    self.FFT_CONVOLVE = Numpy_FFTKits.FFT_CONVOLVE
                    self.KERNEL_CSZ = Numpy_FFTKits.KERNEL_CSZ
                    self.PRC = PatternRotation_Calculator.PRC
                    self.CZR = Numpy_ZoomRotate.CZR
                    self.Resampling = Numpy_Resampling

            self.op = NumpyOperations()

        else:
            raise ValueError("Unsupported BACKEND_4SUBTRACT '%s'" % self.BACKEND_4SUBTRACT)

        self.hdr_target = hdr_target
        self.hdr_object = hdr_object

        self.target_skyrms = target_skyrms
        self.object_skyrms = object_skyrms

        # Ensure that our arrays are contiguous in memory in C-format (x, y) and have dtype float64.
        # If the arrays already satisfy these conditions, this is a no-op and uses no additional memory.
        self.PixA_target = self.op.require(PixA_target, dtype=np.float64, requirements=["C_CONTIGUOUS"])
        self.PixA_object = self.op.require(PixA_object, dtype=np.float64, requirements=["C_CONTIGUOUS"])

        self.PixA_targetVar = self.op.require(PixA_targetVar, dtype=np.float64, requirements=["C_CONTIGUOUS"])
        self.PixA_objectVar = self.op.require(PixA_objectVar, dtype=np.float64, requirements=["C_CONTIGUOUS"])

        self.PixA_target_DMASK = self.op.require(PixA_target_DMASK, dtype=np.float64, requirements=["C_CONTIGUOUS"])
        self.PixA_object_DMASK = self.op.require(PixA_object_DMASK, dtype=np.float64, requirements=["C_CONTIGUOUS"])

        self.PSF_target = self.op.require(PSF_target, dtype=np.float64, requirements=["C_CONTIGUOUS"])
        self.PSF_object = self.op.require(PSF_object, dtype=np.float64, requirements=["C_CONTIGUOUS"])

        self.sci_is_target = sci_is_target

        self.GKerHW = GKerHW
        self.KerPolyOrder = KerPolyOrder
        self.BGPolyOrder = BGPolyOrder
        self.ConstPhotRatio = ConstPhotRatio
        self.Consider_Matching_Kernel = Consider_Matching_Kernel

        self.BACKEND_4SUBTRACT = BACKEND_4SUBTRACT
        self.CUDA_COMPILER = CUDA_COMPILER
        self.CUDA_DEVICE_4SUBTRACT = CUDA_DEVICE_4SUBTRACT
        self.NUM_CPU_THREADS_4SUBTRACT = NUM_CPU_THREADS_4SUBTRACT
        self.NUMBA_CACHE = NUMBA_CACHE
        self.RANDOM_SEED = RANDOM_SEED

    def resampling_image_mask_psf(self):
        """Resample the object image, variance, mask, and PSF to target coordinates.

        Project the target pixel grid into the object frame and then resample
        the object image, its variance, and the detection mask onto the
        target grid.

        It also resamples the object PSF using the same geometric transform.

        The implementation dispatches to either the Cupy or Numpy backend based on
        `BACKEND_4SUBTRACT`.

        This is Step 0 in standard processing.

        Raises
        ------
        ValueError
            If (1) the projected object image falls completely outside the target
            image or (2) if the backend is unsupported.
        """
        if self.BACKEND_4SUBTRACT == "Cupy":
            CR = self.op.Resampling(RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=1)

            if self.hdr_target["CTYPE1"] == "RA---TAN" and self.hdr_target["CTYPE2"] == "DEC--TAN":
                XX_proj, YY_proj = CR.resamp_projection_cd(
                    hdr_obj=self.hdr_object, hdr_targ=self.hdr_target, CDKEY="CD"
                )
            elif self.hdr_target["CTYPE1"] == "RA---TAN-SIP" and self.hdr_target["CTYPE2"] == "DEC--TAN-SIP":
                XX_proj, YY_proj = CR.resamp_projection_sip(
                    hdr_obj=self.hdr_object, hdr_targ=self.hdr_target, NSAMP=1024, RANDOM_SEED=self.RANDOM_SEED
                )
            else:
                raise ValueError(
                    "Unsupported WCS type in target header: CTYPE1 = '%s', CTYPE2 = '%s'"
                    % (self.hdr_target["CTYPE1"], self.hdr_target["CTYPE2"])
                )

        elif self.BACKEND_4SUBTRACT == "Numpy":
            CR = self.op.Resampling(RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=1)
            XX_proj, YY_proj = CR.resamp_projection_astropy(hdr_obj=self.hdr_object, hdr_targ=self.hdr_target)

        else:
            raise ValueError("Unsupported BACKEND_4SUBTRACT '%s'" % self.BACKEND_4SUBTRACT)

        # Check if projection completely outside of target image
        # This computation counts the number of pixels that are within the target image bounds
        NTX = int(self.hdr_target["NAXIS1"])
        NTY = int(self.hdr_target["NAXIS2"])
        NPIX_INNER = self.op.sum(
            self.op.logical_and(
                self.op.logical_and(XX_proj >= 0.5, XX_proj < NTX + 0.5),
                self.op.logical_and(YY_proj >= 0.5, YY_proj < NTY + 0.5),
            )
        )
        if NPIX_INNER == 0:
            raise ValueError("SFFT Error: Projection of object image is completely outside of target image!")

        # Object image:
        PixA_Eobj, EProjDict = CR.frame_extension(
            XX_proj=XX_proj, YY_proj=YY_proj, PixA_obj=self.PixA_object, PAD_FILL_VALUE=0.0, NAN_FILL_VALUE=0.0
        )

        self.PixA_resamp_object = CR.resampling(
            PixA_Eobj=PixA_Eobj, EProjDict=EProjDict, CUDA_COMPILER=self.CUDA_COMPILER, USE_SHARED_MEMORY=False
        )

        # Variance image:
        PixA_EobjVar, EProjDict_Var = CR.frame_extension(
            XX_proj=XX_proj, YY_proj=YY_proj, PixA_obj=self.PixA_objectVar, PAD_FILL_VALUE=0.0, NAN_FILL_VALUE=0.0
        )

        self.PixA_resamp_objectVar = CR.resampling(
            PixA_Eobj=PixA_EobjVar, EProjDict=EProjDict, CUDA_COMPILER=self.CUDA_COMPILER, USE_SHARED_MEMORY=False
        )

        # Mask:
        PixA_Eobj, EProjDict = CR.frame_extension(
            XX_proj=XX_proj, YY_proj=YY_proj, PixA_obj=self.PixA_object_DMASK, PAD_FILL_VALUE=0.0, NAN_FILL_VALUE=0.0
        )

        del XX_proj
        del YY_proj

        self.PixA_resamp_object_DMASK = CR.resampling(
            PixA_Eobj=PixA_Eobj, EProjDict=EProjDict, CUDA_COMPILER=self.CUDA_COMPILER, USE_SHARED_MEMORY=False
        )
        self.BlankMask = self.PixA_resamp_object == 0.0

        # PSF:
        PATTERN_ROTATE_ANGLE = self.op.PRC(hdr_obj=self.hdr_object, hdr_targ=self.hdr_target)

        self.PSF_resamp_object = self.op.CZR(
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
        """Compute the cross-convolution products needed for SFFT subtraction.

        This step convolves the target and object PSFs with each other's
        matched PSF representations, as well as the resampled object image with
        the target PSF. These convolutions prepare the images for the SFFT
        subtraction step.

        This is Step 1 in standard processing.
        """
        self.PixA_Ctarget = self.op.FFT_CONVOLVE(
            PixA_Inp=self.PixA_target,
            KERNEL=self.PSF_resamp_object,
            PAD_FILL_VALUE=0.0,
            NAN_FILL_VALUE=None,
            NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True,
            FFT_BACKEND="Cupy",
        )

        self.PSF_Ctarget = self.op.FFT_CONVOLVE(
            PixA_Inp=self.PSF_target,
            KERNEL=self.PSF_resamp_object,
            PAD_FILL_VALUE=0.0,
            NAN_FILL_VALUE=None,
            NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True,
            FFT_BACKEND="Cupy",
        )

        self.PixA_Cresamp_object = self.op.FFT_CONVOLVE(
            PixA_Inp=self.PixA_resamp_object,
            KERNEL=self.PSF_target,
            PAD_FILL_VALUE=0.0,
            NAN_FILL_VALUE=None,
            NORMALIZE_KERNEL=True,
            FORCE_OUTPUT_C_CONTIGUOUS=True,
            FFT_BACKEND="Cupy",
        )

    def sfft_subtraction(self):
        """Perform the SFFT subtraction and apply background masking.

        This method masks invalid or background pixels, selects the correct
        reference and science images depending on the subtraction direction,
        and runs the chosen backend subtraction implementation.

        This is Step 2 in standard processing.
        """
        LYMASK_BKG = self.op.logical_or(
            self.PixA_target_DMASK == 0, self.PixA_resamp_object_DMASK < 0.1
        )  # background-mask

        NaNmask_Ctarget = self.op.isnan(self.PixA_Ctarget)
        NaNmask_Cresamp_object = self.op.isnan(self.PixA_Cresamp_object)
        if NaNmask_Ctarget.any() or NaNmask_Cresamp_object.any():
            NaNmask = self.op.logical_or(NaNmask_Ctarget, NaNmask_Cresamp_object)
            ZeroMask = self.op.logical_or(NaNmask, LYMASK_BKG)
        else:
            ZeroMask = LYMASK_BKG

        del LYMASK_BKG

        PixA_mCtarget = self.PixA_Ctarget.copy()
        PixA_mCtarget[ZeroMask] = 0.0

        PixA_mCresamp_object = self.PixA_Cresamp_object.copy()
        PixA_mCresamp_object[ZeroMask] = 0.0

        del ZeroMask

        # trigger sfft subtraction
        if self.sci_is_target:
            PixA_REF = self.PixA_Cresamp_object
            PixA_SCI = self.PixA_Ctarget
            PixA_mREF = PixA_mCresamp_object
            PixA_mSCI = PixA_mCtarget
        else:
            PixA_REF = self.PixA_Ctarget
            PixA_SCI = self.PixA_Cresamp_object
            PixA_mREF = PixA_mCtarget
            PixA_mSCI = PixA_mCresamp_object

        if self.BACKEND_4SUBTRACT == "Cupy":
            self.Solution, self.PixA_DIFF = self.op.PCCP(
                PixA_REF=PixA_REF,
                PixA_SCI=PixA_SCI,
                PixA_mREF=PixA_mREF,
                PixA_mSCI=PixA_mSCI,
                ForceConv="REF" if self.sci_is_target else "NEW",
                GKerHW=self.GKerHW,
                KerPolyOrder=self.KerPolyOrder,
                BGPolyOrder=self.BGPolyOrder,
                ConstPhotRatio=self.ConstPhotRatio,
                CUDA_DEVICE_4SUBTRACT=self.CUDA_DEVICE_4SUBTRACT,
            )
        elif self.BACKEND_4SUBTRACT == "Numpy":
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
        else:
            raise ValueError("Unsupported BACKEND_4SUBTRACT '%s'" % self.BACKEND_4SUBTRACT)

        self.PixA_DIFF[self.BlankMask] = 0.0

    def find_decorrelation(self):
        """Compute the decorrelation kernel for the difference image.

        Extracts the matching kernel from the SFFT solution and
        computes the full decorrelation filter in Fourier space.
        The resulting kernel is stored for later application
        on the difference image and score image generation.

        This is Step 3 in standard processing.
        """
        N0, N1 = self.PixA_DIFF.shape
        L0, L1 = 2 * self.GKerHW + 1, 2 * self.GKerHW + 1
        DK = self.KerPolyOrder
        Fpq = int((self.BGPolyOrder + 1) * (self.BGPolyOrder + 2) / 2)
        XY_q = np.array([[N0 / 2.0 + 0.5, N1 / 2.0 + 0.5]])

        self.Solution = self.op.asnumpy(self.Solution)
        MATCH_KERNEL = self.op.array(
            Realize_MatchingKernel(XY_q=XY_q).FromArray(
                Solution=self.Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq
            )[0],
            dtype=np.float64,
        )
        self.MATCH_KERNEL = self.op.asnumpy(MATCH_KERNEL)

        if self.Consider_Matching_Kernel:
            MK = self.op.asnumpy(MATCH_KERNEL)
        else:
            MK = None
        self.FKDECO = DeCorrelation_Calculator(
            NX_IMG=N0,
            NY_IMG=N1,
            KERNEL_JQueue=[self.op.asnumpy(self.PSF_resamp_object)],
            BKGSIG_JQueue=[self.target_skyrms],
            KERNEL_IQueue=[self.op.asnumpy(self.PSF_target)],
            BKGSIG_IQueue=[self.object_skyrms],
            MATCH_KERNEL=MK,
            REAL_OUTPUT=False,
            REAL_OUTPUT_SIZE=None,
            NORMALIZE_OUTPUT=True,
            VERBOSE_LEVEL=2,
        )
        self.FKDECO = self.op.array(self.FKDECO, dtype=np.complex128)
        print("Decorrelaton kernel calculated.")

    def apply_decorrelation(self, img, requirements=None):
        """Apply the precomputed decorrelation filter to an image.

        If the input image has the same shape as the decorrelation kernel, the
        filter is applied directly in Fourier space. Otherwise, the input kernel
        is centered and truncated to match the decoration kernel shape.

        This is Step 4 in standard processing.

        Parameters
        ----------
        img : ndarray
            Input image or kernel to decorrelate.

        Returns
        -------
        ndarray
            Decorrelated image.
        """
        _img = self.op.asnumpy(img)
        if _img.shape == self.FKDECO.shape:
            FPixA = np.fft.fft2(_img)
            PixA_decorr = np.fft.ifft2(FPixA * self.FKDECO).real
            decorimg = self.op.array(PixA_decorr, dtype=np.float64)
        else:
            NK0, NK1 = _img.shape
            N0, N1 = self.FKDECO.shape
            KERN_CSZ = KERNEL_CSZ(KERNEL=_img, NX_IMG=N0, NY_IMG=N1)
            FKERN_decorr = np.fft.fft2(KERN_CSZ) * self.FKDECO
            PixA_KERN_decorr = KERNEL_CSZ_INV(np.fft.ifft2(FKERN_decorr).real, NX_KERN=NK0, NY_KERN=NK1)
            decorimg = self.op.array(PixA_KERN_decorr, dtype=np.float64)

        return self.op.require(decorimg, requirements=requirements)

    def create_score_image(self, requirements=None):
        """Create a score image from the decorrelated difference and PSFs.

        Filter the difference image with the decorrelation kernel and
        compute the matched-filter score using the target and object PSFs, and
        normalize the result by the background noise level.

        This is Step 5 in standard processing.

        Returns
        -------
        ndarray
            Score image used for transient detection.
        """
        # retrieve the decorrelated PSF
        # Note: here we assume the same pixel size for PSF and imgaes.
        NX, NY = self.PixA_target.shape
        PSF_object_CSZ = self.op.KERNEL_CSZ(KERNEL=self.PSF_object, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ = self.op.KERNEL_CSZ(KERNEL=self.PSF_target, NX_IMG=NX, NY_IMG=NY)
        FPSF_dDIFF = self.op.fft.fft2(PSF_object_CSZ) * self.op.fft.fft2(PSF_target_CSZ) * self.FKDECO

        # apply the decorrelation on difference image again
        # This is redundant, but we want to compute the decorrelation kernel in fourier space
        FPixA_DIFF = self.op.fft.fft2(self.PixA_DIFF)
        FPixA_dDIFF = FPixA_DIFF * self.FKDECO

        FPixA_SCORE = FPixA_dDIFF * self.op.conj(FPSF_dDIFF)
        PixA_SCORE = self.op.fft.ifft2(FPixA_SCORE).real

        # an ad-hoc correction to make score image has standard Gaussian distribution at background
        skysig_SCORE = SkyLevel_Estimator.SLE(PixA_obj=self.op.asnumpy(PixA_SCORE))[1]
        PixA_SCORE /= skysig_SCORE

        return self.op.require(PixA_SCORE, requirements=requirements)

    def create_variance_image(self, requirements=None):
        """Estimate the variance image of the un-decorrelated difference.

        The method propagates the variance image through the PSF-convolved
        resampled object and target images to produce an estimate of the difference-image noise
        variance.

        This is not necessarily in the standard processing because it's the variance of the
        un-decorrelated difference image, which is not a product we directly use.

        Returns
        -------
        ndarray
            Estimated variance image for the difference image.
        """
        self.op.require(self.PixA_targetVar, requirements="C_CONTIGUOUS")
        self.op.require(self.PixA_resamp_objectVar, requirements="C_CONTIGUOUS")

        # calculate variance image for (un-decorrelated) difference image
        NX, NY = self.PixA_target.shape
        PSF_resamp_object_CSZ = self.op.KERNEL_CSZ(KERNEL=self.PSF_resamp_object, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ = self.op.KERNEL_CSZ(KERNEL=self.PSF_target, NX_IMG=NX, NY_IMG=NY)

        # Note: let's skip the matching kernel here, as it is expected to be a minor compensation.
        PixA_dDIFFVar = self.op.fft.ifft2(
            self.op.fft.fft2(self.PixA_resamp_objectVar)
            * self.op.fft.fft2((self.op.fft.ifft2(self.op.fft.fft2(PSF_target_CSZ) * self.FKDECO)).real ** 2)
        ).real
        PixA_dDIFFVar += self.op.fft.ifft2(
            self.op.fft.fft2(self.PixA_targetVar)
            * self.op.fft.fft2((self.op.fft.ifft2(self.op.fft.fft2(PSF_resamp_object_CSZ) * self.FKDECO)).real ** 2)
        ).real

        return self.op.require(PixA_dDIFFVar, requirements=requirements)
