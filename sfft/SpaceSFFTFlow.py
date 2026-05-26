# IMPORTS Standard
import numpy as np

# IMPORTS Internal
from sfft.utils.DeCorrelationCalculator import DeCorrelation_Calculator, KERNEL_CSZ, KERNEL_CSZ_INV
from sfft.utils.SkyLevelEstimator import SkyLevel_Estimator
from sfft.utils.SFFTSolutionReader import Realize_MatchingKernel

__last_update__ = "2025-05-26"
__author__ = "Lei Hu <leihu@andrew.cmu.edu> and Michael Wood-Vasey <wmwv@pitt.edu>"

class SpaceSFFT_CupyFlow:
    """Run A Cupy WorkFlow for SFFT subtraction"""

    def __init__(self, hdr_target, hdr_object, 
                 target_skyrms, object_skyrms, 
                 PixA_target, PixA_object,
                 PixA_targetVar, PixA_objectVar,
                 PixA_target_DMASK, PixA_object_DMASK, 
                 PSF_target, PSF_object,
                 sci_is_target=True,
                 GKerHW=9, KerPolyOrder=2, BGPolyOrder=0, ConstPhotRatio=True, 
                 Consider_Matching_Kernel=False,
                 BACKEND_4SUBTRACT="Cupy",
                 CUDA_COMPILER="nvrtc", CUDA_DEVICE_4SUBTRACT='0',
                 NUM_CPU_THREADS_4SUBTRACT=8, NUMBA_CACHE=True,
                 GAIN=1.0, RANDOM_SEED=10086):
        """Do things.

        Parameters
        ----------
           hdr_target: astropy header
              The target image has the coordinate system to which we are
              resampling.  This is that image's header.  SFFT will use the following keywords from the header:
                * All WCS keywords (including LONPOLE)
                * NAXIS1
                * NAXIS2

           hdr_object: astropy header
              Original (unresampled) header of the image to be resampled to match target.

           target_skyrms: float

           object_skyrms: float

           PixA_target: cupy array (float64)
              2d image data of target, indexed by x, y.  (Note that raw
              data read from fits files are indexed y, x; transpose to
              get this.)

           PixA_object: cupy array (float64)
              2d image data of original image, indexed by x, y.

           PixA_targetVar: cupy array (float64)
              2d image variance of original image, indexed by x, y.

           PixA_objectVar: cupy array (float64)
              2d image variance of original image, indexed by x, y. 

           PixA_target_DMASK: cupy array (bool)
              2d detection mask for target image
        
           PixA_object_DMASK: cupy array (bool)
              2d detection mask for unresampled object image

           PSF_target: cupy array (float64)
              2d PSF model; axis lengths must be odd.  center of PSF is
              center of center pixel.

           PSF_object: cupy array (float64)
              object PSF

           sci_is_target : bool
              If True, will subtract object - target.  If false, will subtract target - object.

           GKerHW: int
              Matching kernel half-width (full width is 2*GkerHW + 1 )
           
           KerPolyOrder: int
              Order of spatial variation in kernel

           BGPolyOrder: int
              Order of differential background 2d polynomial.  (Usually
              just leave this 0, we assume the image is sky subtracted.)

           ConstPhotRatio: bool
              Assume relative zeropoints of target and (resampled)
              object have no spatial variation.

           Consider_Matching_Kernel: bool, default False
              Whether to consider the matching kernel in the decorrelation.
              The mathching kernel is close to a delta function doing a shift, 
              by default, we may ignore it.

           CUDA_DEVICE_4SUBTRACT: str, default '0'
              Which CUDA device to use.

           GAIN: float
              e-/ADU gain for both images.  (So, poisson noise, σ_adu = √(gain) * adu.)

           RANDOM_SEED: int default 10086
              Random seed to use to CR.resamp_projection_sip.  TODO :
              make it so that when this is None, a "real" random seed is
              generated.

        """

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

        self.PixA_target = PixA_target
        self.PixA_object = PixA_object

        if PixA_targetVar.dtype != np.float64:
            PixA_targetVar = PixA_targetVar.astype(np.float64)
        self.PixA_targetVar = PixA_targetVar
        if PixA_objectVar.dtype != np.float64:
            PixA_objectVar = PixA_objectVar.astype(np.float64)
        self.PixA_objectVar = PixA_objectVar

        if PixA_target_DMASK.dtype != np.float64:
            PixA_target_DMASK = PixA_target_DMASK.astype(np.float64)
        self.PixA_target_DMASK = PixA_target_DMASK
        if PixA_object_DMASK.dtype != np.float64:
            PixA_object_DMASK = PixA_object_DMASK.astype(np.float64)
        self.PixA_object_DMASK = PixA_object_DMASK

        self.PSF_target = PSF_target
        self.PSF_object = PSF_object

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
        self.GAIN = GAIN
        self.RANDOM_SEED = RANDOM_SEED

        # Dependent loads if we're Numpy or Cupy
        # We do this in the object initialization
        # so that in prinpciple we can have objects that are cupy and ones that are numpy
        # in the same running processing.  I don't think we'll want to run this way
        # but I don't want to debug accidentally doing this.
        if self.BACKEND_4SUBTRACT == "Cupy":
            import cupy as cp

            from sfft.PureCupyCustomizedPacket import PureCupy_Customized_Packet
            self.PCCP = PureCupy_Customized_Packet.PCCP

            from sfft.utils.PureCupyFFTKits import PureCupy_FFTKits
            self.op_fft_CONVOLVE = PureCupy_FFTKits.FFT_CONVOLVE
            self.op_KERNEL_CSZ = PureCupy_FFTKits.KERNEL_CSZ

            from sfft.utils.PatternRotationCalculator import PatternRotation_Calculator
            self.op_PRC = PatternRotation_Calculator.PRC

            from sfft.utils.ResampKits import Cupy_ZoomRotate
            self.CZR = Cupy_ZoomRotate.CZR

            from sfft.utils.ResampKits import Cupy_Resampling
            self.op_Resampling = Cupy_Resampling

            self.op_fft = cp.fft
            self.op_sum = cp.sum
            self.op_logical_and = cp.logical_and
            self.op_logical_or = cp.logical_or
            self.op_isnan = cp.isnan
            self.op_asnumpy = cp.asnumpy
            self.op_array = cp.array
            self.op_conj = cp.conj

        elif self.BACKEND_4SUBTRACT == "Numpy":
            from sfft.utils.NumpyFFTKits import Numpy_FFTKits
            self.op_fft_CONVOLVE = Numpy_FFTKits.FFT_CONVOLVE
            self.op_KERNEL_CSZ = Numpy_FFTKits.KERNEL_CSZ
            from sfft.utils.ResampKits import Numpy_ZoomRotate
            self.op_CZR = Numpy_ZoomRotate.CZR
            from sfft.utils.ResampKits import Numpy_Resampling
            self.op_Resampling = Numpy_Resampling

            self.op_fft = np.fft
            self.op_sum = np.sum
            self.op_logical_and = np.logical_and
            self.op_logical_or = np.logical_or
            self.op_isnan = np.isnan
            self.op_asnumpy = lambda x: x  # identity function, since we're already in numpy
            self.op_array = np.array
            self.op_conj = np.conj

        else:
            raise ValueError("Unsupported BACKEND_4SUBTRACT '%s'" % self.BACKEND_4SUBTRACT)

    def resampling_image_mask_psf( self ):
        """Step 0. run resampling for input object image, variance image, mask, and PSF"""
        CR = self.op_Resampling(RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=1)

        if self.hdr_target["CTYPE1"] == "RA---TAN":
            assert self.hdr_target["CTYPE2"] == "DEC--TAN"
            XX_proj, YY_proj = CR.resamp_projection_cd(hdr_obj=self.hdr_object,
                                                               hdr_targ=self.hdr_target, CDKEY="CD")

        if self.hdr_target["CTYPE1"] == "RA---TAN-SIP":
            assert self.hdr_target["CTYPE2"] == "DEC--TAN-SIP"
            XX_proj, YY_proj = CR.resamp_projection_sip(hdr_obj=self.hdr_object,
                                                                hdr_targ=self.hdr_target,
                                                                NSAMP=1024,
                                                                RANDOM_SEED=self.RANDOM_SEED)

        # check if projection completely outside of target image
        # TODO: this check is currently not smart...
        NTX = int(self.hdr_target["NAXIS1"])
        NTY = int(self.hdr_target["NAXIS2"])
        NPIX_INNER = self.op_sum(self.op_logical_and( self.op_logical_and(XX_proj >= 0.5, XX_proj < NTX+0.5),
                                            self.op_logical_and(YY_proj >= 0.5, YY_proj < NTY+0.5) ))
        assert NPIX_INNER > 0, "SFFT Error: Projection of object image is completely outside of target image!"

        # Object image:
        PixA_Eobj, EProjDict = CR.frame_extension(XX_proj=XX_proj,
                                                      YY_proj=YY_proj, 
                                                      PixA_obj=self.PixA_object,
                                                      PAD_FILL_VALUE=0.,
                                                      NAN_FILL_VALUE=0.)

        self.PixA_resamp_object = CR.resampling(PixA_Eobj=PixA_Eobj,
                                                    EProjDict=EProjDict,
                                                    CUDA_COMPILER=self.CUDA_COMPILER,
                                                    USE_SHARED_MEMORY=False)

        # Variance image:
        PixA_EobjVar, EProjDict_Var = CR.frame_extension(XX_proj=XX_proj,
                                                             YY_proj=YY_proj, 
                                                             PixA_obj=self.PixA_objectVar,
                                                             PAD_FILL_VALUE=0.,
                                                             NAN_FILL_VALUE=0.)

        self.PixA_resamp_objectVar = CR.resampling(PixA_Eobj=PixA_EobjVar,
                                                       EProjDict=EProjDict,
                                                       CUDA_COMPILER=self.CUDA_COMPILER,
                                                       USE_SHARED_MEMORY=False)

        # Mask:
        PixA_Eobj, EProjDict = CR.frame_extension(XX_proj=XX_proj,
                                                      YY_proj=YY_proj, 
                                                      PixA_obj=self.PixA_object_DMASK,
                                                      PAD_FILL_VALUE=0.,
                                                      NAN_FILL_VALUE=0.)

        del XX_proj
        del YY_proj
        
        self.PixA_resamp_object_DMASK = CR.resampling(PixA_Eobj=PixA_Eobj,
                                                          EProjDict=EProjDict,
                                                          CUDA_COMPILER=self.CUDA_COMPILER,
                                                          USE_SHARED_MEMORY=False)
        self.BlankMask = self.PixA_resamp_object == 0.


        # PSF:
        PATTERN_ROTATE_ANGLE = self.op_PRC(hdr_obj=self.hdr_object, hdr_targ=self.hdr_target)

        self.PSF_resamp_object = self.op_CZR(PixA_obj=self.PSF_object,
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

    def cross_convolution( self ):
        # * step 1. cross convolution
        self.PixA_Ctarget = self.op_FFT_CONVOLVE(PixA_Inp=self.PixA_target,
                                                              KERNEL=self.PSF_resamp_object, 
                                                              PAD_FILL_VALUE=0.,
                                                              NAN_FILL_VALUE=None,
                                                              NORMALIZE_KERNEL=True,
                                                              FORCE_OUTPUT_C_CONTIGUOUS=True,
                                                              FFT_BACKEND="Cupy")
                                                            
        self.PSF_Ctarget = self.op_FFT_CONVOLVE(PixA_Inp=self.PSF_target,
                                                             KERNEL=self.PSF_resamp_object,
                                                             PAD_FILL_VALUE=0.,
                                                             NAN_FILL_VALUE=None,
                                                             NORMALIZE_KERNEL=True,
                                                             FORCE_OUTPUT_C_CONTIGUOUS=True,
                                                             FFT_BACKEND="Cupy")

        self.PixA_Cresamp_object = self.op_FFT_CONVOLVE(PixA_Inp=self.PixA_resamp_object,
                                                                     KERNEL=self.PSF_target,
                                                                     PAD_FILL_VALUE=0.,
                                                                     NAN_FILL_VALUE=None,
                                                                     NORMALIZE_KERNEL=True,
                                                                     FORCE_OUTPUT_C_CONTIGUOUS=True,
                                                                     FFT_BACKEND="Cupy")

    def sfft_subtraction( self ):
        """Step 2. sfft subtraction"""
        LYMASK_BKG = self.op_logical_or(self.PixA_target_DMASK == 0, self.PixA_resamp_object_DMASK < 0.1)   # background-mask

        NaNmask_Ctarget = self.op_isnan(self.PixA_Ctarget)
        NaNmask_Cresamp_object = self.op_isnan(self.PixA_Cresamp_object)
        if NaNmask_Ctarget.any() or NaNmask_Cresamp_object.any():
            NaNmask = self.op_logical_or(NaNmask_Ctarget, NaNmask_Cresamp_object)
            ZeroMask = self.op_logical_or(NaNmask, LYMASK_BKG)
        else:
            ZeroMask = LYMASK_BKG

        del LYMASK_BKG
            
        PixA_mCtarget = self.PixA_Ctarget.copy()
        PixA_mCtarget[ZeroMask] = 0.

        PixA_mCresamp_object = self.PixA_Cresamp_object.copy()
        PixA_mCresamp_object[ZeroMask] = 0.

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
            self.Solution, self.PixA_DIFF = self.PCCP(
                PixA_REF=PixA_REF,
                PixA_SCI=PixA_SCI,
                PixA_mREF=PixA_mREF,
                PixA_mSCI=PixA_mSCI,
                ForceConv='REF' if self.sci_is_target else 'NEW',
                GKerHW=self.GKerHW,
                KerPolyOrder=self.KerPolyOrder,
                BGPolyOrder=self.BGPolyOrder,
                ConstPhotRatio=self.ConstPhotRatio, 
                CUDA_DEVICE_4SUBTRACT=self.CUDA_DEVICE_4SUBTRACT
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
    
        self.PixA_DIFF[self.BlankMask] = 0.

    def find_decorrelation( self ):
        """Step 3. perform decorrelation in Fourier domain
        extract matching kernel at the center
        """
        N0, N1 = self.PixA_DIFF.shape
        L0, L1 = 2*self.GKerHW + 1, 2*self.GKerHW + 1
        DK = self.KerPolyOrder
        Fpq = int((self.BGPolyOrder+1)*(self.BGPolyOrder+2)/2)
        XY_q = np.array([[N0/2.+0.5, N1/2.+0.5]])

        self.Solution = self.op_asnumpy(self.Solution)
        MATCH_KERNEL = self.op_array(Realize_MatchingKernel(XY_q=XY_q).FromArray(
            Solution=self.Solution, N0=N0, N1=N1, L0=L0, L1=L1, DK=DK, Fpq=Fpq
        )[0], dtype=np.float64)
        self.MATCH_KERNEL = self.op_asnumpy(MATCH_KERNEL)

        if self.Consider_Matching_Kernel:
            MK = self.op_asnumpy(MATCH_KERNEL)
        else:
            MK = None
        self.FKDECO = DeCorrelation_Calculator(NX_IMG=N0, 
                                               NY_IMG=N1, 
                                               KERNEL_JQueue=[self.op_asnumpy(self.PSF_resamp_object)], 
                                               BKGSIG_JQueue=[self.target_skyrms], 
                                               KERNEL_IQueue=[self.op_asnumpy(self.PSF_target)], 
                                               BKGSIG_IQueue=[self.object_skyrms], 
                                               MATCH_KERNEL=MK, 
                                               REAL_OUTPUT=False, 
                                               REAL_OUTPUT_SIZE=None, 
                                               NORMALIZE_OUTPUT=True, 
                                               VERBOSE_LEVEL=2)
        self.FKDECO = self.op_array(self.FKDECO, dtype=np.complex128)
        print("Decorrelaton kernel calculated.")
    
    def apply_decorrelation( self, img ):
        # do decorrelation

        # decorrelate difference image
        _img = self.op_asnumpy(img)
        if _img.shape == self.FKDECO.shape:
            FPixA = np.fft.fft2(_img)
            PixA_decorr = np.fft.ifft2(FPixA * self.FKDECO).real
            decorimg = self.op_array(PixA_decorr, dtype=np.float64)
        else:
            NK0, NK1 = _img.shape
            N0, N1 = self.FKDECO.shape
            KERN_CSZ = KERNEL_CSZ(KERNEL=_img, NX_IMG=N0, NY_IMG=N1)
            FKERN_decorr = np.fft.fft2(KERN_CSZ) * self.FKDECO
            PixA_KERN_decorr = KERNEL_CSZ_INV(np.fft.ifft2(FKERN_decorr).real, NX_KERN=NK0, NY_KERN=NK1)
            decorimg = self.op_array(PixA_KERN_decorr, dtype=np.float64)
        return decorimg
    
    def create_score_image( self ):        
        # retrieve the decorrelated PSF
        # Note: here we assume the same pixel size for PSF and imgaes.
        NX, NY = self.PixA_target.shape
        PSF_object_CSZ = self.op_KERNEL_CSZ(KERNEL=self.PSF_object, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ = self.op_KERNEL_CSZ(KERNEL=self.PSF_target, NX_IMG=NX, NY_IMG=NY)
        FPSF_dDIFF = self.op_fft.fft2(PSF_object_CSZ) * self.op_fft.fft2(PSF_target_CSZ) * self.FKDECO

        # apply the decorrelation on difference image again (redundant, a workaround) 
        FPixA_DIFF = self.op_fft.fft2( self.PixA_DIFF )
        FPixA_dDIFF = FPixA_DIFF * self.FKDECO

        FPixA_SCORE = FPixA_dDIFF * self.op_conj(FPSF_dDIFF)
        PixA_SCORE = self.op_fft.ifft2(FPixA_SCORE).real

        # an ad-hoc correction to make score image has standrd Gaussian distribution at background
        skysig_SCORE = SkyLevel_Estimator.SLE(PixA_obj=self.op_asnumpy(PixA_SCORE))[1]
        PixA_SCORE /= skysig_SCORE

        return PixA_SCORE

    def create_variance_image( self ):

        assert self.PixA_targetVar.flags['C_CONTIGUOUS']
        assert self.PixA_resamp_objectVar.flags['C_CONTIGUOUS']

        # calculate variance image for (un-decorrelated) difference image
        NX, NY = self.PixA_target.shape
        PSF_resamp_object_CSZ = self.op_KERNEL_CSZ(KERNEL=self.PSF_resamp_object, NX_IMG=NX, NY_IMG=NY)
        PSF_target_CSZ = self.op_KERNEL_CSZ(KERNEL=self.PSF_target, NX_IMG=NX, NY_IMG=NY)

        # Note: let's skip the matching kernel here, as it is expected to be a minor compensation.
        PixA_dDIFFVar = self.op_fft.ifft2(
            self.op_fft.fft2(self.PixA_resamp_objectVar) * \
            self.op_fft.fft2((self.op_fft.ifft2(self.op_fft.fft2(PSF_target_CSZ) * self.FKDECO)).real**2)
        ).real
        PixA_dDIFFVar += self.op_fft.ifft2(
            self.op_fft.fft2(self.PixA_targetVar) * \
            self.op_fft.fft2((self.op_fft.ifft2(self.op_fft.fft2(PSF_resamp_object_CSZ) * self.FKDECO)).real**2)
        ).real

        return PixA_dDIFFVar
    
    # Do we need this?  We should just unreference the object
    def cleanup( self ):
        pass
