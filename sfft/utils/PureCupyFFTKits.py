import importlib.util as iu
if iu.find_spec('cupy') is not None:
    import cupy as cp

__last_update__ = "2024-09-22"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>"
__version__ = "v1.6.1"

#########################################################################
#||                                                                   ||#
#||                           Acknowledgment                          ||#
#||                                                                   ||#
#########################################################################
#||                                                                   ||#
#||   This code was developed during the NASA GPU Hackathon 2024      ||#
#||   (https://www.nas.nasa.gov/hackathon/#home) as part of the       ||#
#||   pipeline optimization efforts for the Roman Supernova PIT       ||#
#||   team (https://github.com/Roman-Supernova-PIT). I extend my      ||#
#||   sincere appreciation to my team members and our Hackathon       ||#
#||   mentors for their valuable support and insightful               ||#
#||   suggestions.                                                    ||#
#||                                                                   ||#
#||   I would like to specifically acknowledge the contributions      ||#
#||   of the following team members: Lauren Aldoroty (Duke),          ||#
#||   Robert Knop (LBNL), Shu Liu (Pitt), and Michael Wood-Vasey      ||#
#||   (Pitt). Additionally, I am grateful for the guidance provided   ||#
#||   by our mentors, Marcus Manos and Lucas Erlandson from NVIDIA.   ||#
#||                                                                   ||#
#||   The collaborative environment fostered by the NASA GPU          ||#
#||   Hackathon and the Roman Supernova PIT team was essential for    ||#
#||   the development and optimization of this code.                  ||#
#||                                                                   ||#
#########################################################################

class PureCupy_FFTKits:
    @staticmethod
    def KERNEL_CSZ(KERNEL, NX_IMG, NY_IMG, NORMALIZE_KERNEL=False):
        """ Circular Shift the kernel and extend to the target size """
        N0, N1 = NX_IMG, NY_IMG
        L0, L1 = KERNEL.shape
        W0, W1 = (L0 - 1)//2, (L1 - 1)//2
        # Note: currently only support odd-sized kernel
        assert L0 % 2 == 1 and L1 % 2 == 1   
        
        if NORMALIZE_KERNEL:
            KERNEL_TZP_GPU = cp.pad(KERNEL / cp.sum(KERNEL), \
                pad_width=((0, N0 - L0), (0, N1 - L1)), mode='constant', constant_values=0.)
        else:   
            KERNEL_TZP_GPU = cp.pad(KERNEL, pad_width=((0, N0 - L0), (0, N1 - L1)), \
                mode='constant', constant_values=0.)
        KIMG_CSZ_GPU = cp.roll(cp.roll(KERNEL_TZP_GPU, -W0, axis=0), -W1, axis=1)
        return KIMG_CSZ_GPU

    @staticmethod
    def KERNEL_CSZ_INV(KIMG_GPU, NX_KERN, NY_KERN, VERBOSE_LEVEL=2):
        """ Inverse Circular Shift the kernel and truncate to the target size """
        L0, L1 = NX_KERN, NY_KERN
        W0, W1 = (L0 - 1)//2, (L1 - 1)//2
        # Note: currently only support odd-sized kernel
        assert L0 % 2 == 1 and L1 % 2 == 1
        
        KIMG_iCSZ_GPU = cp.roll(cp.roll(KIMG_GPU, W1, axis=1), W0, axis=0)
        KERNEL_GPU = KIMG_iCSZ_GPU[:L0, :L1]
        if VERBOSE_LEVEL in [1, 2]:
            LOSE_RATIO = 1. - cp.sum(cp.abs(KERNEL_GPU)) / cp.sum(cp.abs(KIMG_iCSZ_GPU))
            _report_message = "Kernel Truncation Loses APE = [%.4f %s]" %(LOSE_RATIO*100, '%')
            print("MeLOn CheckPoint: %s " % _report_message)
        return KERNEL_GPU

    @staticmethod
    def FFT_CONVOLVE(PixA_Inp_GPU, KERNEL, PAD_FILL_VALUE=0., NAN_FILL_VALUE=0., 
        NORMALIZE_KERNEL=False, FORCE_OUTPUT_C_CONTIGUOUS=False, FFT_BACKEND="Cupy"):
        """ FFT Convolition """
        N0, N1 = PixA_Inp_GPU.shape
        L0, L1 = KERNEL.shape
        assert L0 % 2 == 1 and L1 % 2 == 1

        W0, W1 = (L0 - 1)//2, (L1 - 1)//2 
        NX_IMG, NY_IMG = N0 + 2*W0, N1 + 2*W1

        # zero padding on input image
        PixA_EInp_GPU = cp.pad(
            array=PixA_Inp_GPU, 
            pad_width=((W0, W0), (W1, W1)), 
            mode='constant', 
            constant_values=PAD_FILL_VALUE
        )

        if NAN_FILL_VALUE is not None:
            PixA_EInp_GPU[cp.isnan(PixA_EInp_GPU)] = NAN_FILL_VALUE
        
        # circular shift the kernel to the center and extend to the target size
        KIMG_CSZ_GPU = PureCupy_FFTKits.KERNEL_CSZ(KERNEL=KERNEL, 
            NX_IMG=NX_IMG, NY_IMG=NY_IMG, NORMALIZE_KERNEL=NORMALIZE_KERNEL)

        # perform convolution in Fourier domain
        if FFT_BACKEND == "Cupy":
            PixA_Out_GPU = cp.fft.ifft2(
                cp.fft.fft2(PixA_EInp_GPU) * cp.fft.fft2(KIMG_CSZ_GPU)
            ).real[W0: -W0, W1: -W1]

        if FORCE_OUTPUT_C_CONTIGUOUS:
            if not PixA_Out_GPU.flags['C_CONTIGUOUS']:
                PixA_Out_GPU = cp.ascontiguousarray(PixA_Out_GPU)
        return PixA_Out_GPU

class Cupy_DeCorrelation:
    @staticmethod
    def DeCorrelation_Calculator(NX_IMG, NY_IMG, KERNEL_JQueue, BKGSIG_JQueue, KERNEL_IQueue=[], BKGSIG_IQueue=[], 
        MATCH_KERNEL=None, REAL_OUTPUT=False, REAL_OUTPUT_SIZE=None, NORMALIZE_OUTPUT=True, VERBOSE_LEVEL=2):

        NUM_I, NUM_J = len(KERNEL_IQueue), len(KERNEL_JQueue)
        assert NUM_J > 0

        DELTA_KERNEL = cp.array([
            [0, 0, 0], 
            [0, 1, 0], 
            [0, 0, 0]], dtype=cp.float64
        )

        FDENO = None
        for KERNEL, BKGSIG in zip(KERNEL_JQueue, BKGSIG_JQueue):
            if KERNEL is not None:
                K_CSZ = PureCupy_FFTKits.KERNEL_CSZ(KERNEL=KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            else:
                K_CSZ = PureCupy_FFTKits.KERNEL_CSZ(KERNEL=DELTA_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            FK_CSZ = cp.fft.fft2(K_CSZ)
            FK2_CSZ = (cp.conj(FK_CSZ) * FK_CSZ).real
            if FDENO is None:
                FDENO = (BKGSIG**2 * FK2_CSZ) / NUM_J**2
            else: 
                FDENO += (BKGSIG**2 * FK2_CSZ) / NUM_J**2

        if MATCH_KERNEL is not None:
            MK_CSZ = PureCupy_FFTKits.KERNEL_CSZ(KERNEL=MATCH_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
        else:
            MK_CSZ = PureCupy_FFTKits.KERNEL_CSZ(KERNEL=DELTA_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)

        FMK_CSZ = cp.fft.fft2(MK_CSZ)
        FMK2_CSZ = (cp.conj(FMK_CSZ) * FMK_CSZ).real

        for KERNEL, BKGSIG in zip(KERNEL_IQueue, BKGSIG_IQueue):
            if KERNEL is not None:
                K_CSZ = PureCupy_FFTKits.KERNEL_CSZ(KERNEL=KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            else:
                K_CSZ = PureCupy_FFTKits.KERNEL_CSZ(KERNEL=DELTA_KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG)
            FK_CSZ = cp.fft.fft2(K_CSZ)
            FK2_CSZ = (cp.conj(FK_CSZ) * FK_CSZ).real
            FDENO += (BKGSIG**2 * FK2_CSZ * FMK2_CSZ) / NUM_I**2

        FDENO = cp.sqrt(FDENO)
        FKDECO = 1. / FDENO

        if not REAL_OUTPUT:
            if NORMALIZE_OUTPUT:
                NORMALIZE_FACTOR = 1./FKDECO[0, 0]
                FKDECO *= NORMALIZE_FACTOR
            return FKDECO

        if REAL_OUTPUT:
            KDECO = cp.fft.ifft2(FKDECO).real
            KDECO = PureCupy_FFTKits.KERNEL_CSZ_INV(
                KIMG=KDECO, NX_KERN=REAL_OUTPUT_SIZE[0], NY_KERN=REAL_OUTPUT_SIZE[1],
                VERBOSE_LEVEL=VERBOSE_LEVEL
            )
            
            if NORMALIZE_OUTPUT:
                assert REAL_OUTPUT_SIZE is not None
                NORMALIZE_FACTOR = 1./np.sum(KDECO)
                KDECO *= NORMALIZE_FACTOR
            return KDECO
