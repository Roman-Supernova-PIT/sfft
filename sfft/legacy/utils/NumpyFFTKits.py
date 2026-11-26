# IMPORTS Standard
import numpy as np

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
            KERNEL_TZP = np.pad(
                KERNEL / np.sum(KERNEL), pad_width=((0, N0 - L0), (0, N1 - L1)), mode="constant", constant_values=0.0
            )
        else:
            KERNEL_TZP = np.pad(KERNEL, pad_width=((0, N0 - L0), (0, N1 - L1)), mode="constant", constant_values=0.0)
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
            LOSE_RATIO = 1.0 - np.sum(np.abs(KERNEL)) / np.sum(np.abs(KIMG_iCSZ))
            _report_message = "Kernel Truncation Loses APE = [%.4f %s]" % (LOSE_RATIO * 100, "%")
            print("MeLOn CheckPoint: %s " % _report_message)
        return KERNEL

    @staticmethod
    def FFT_CONVOLVE(
        PixA_Inp,
        KERNEL,
        PAD_FILL_VALUE=0.0,
        NAN_FILL_VALUE=0.0,
        NORMALIZE_KERNEL=False,
        FORCE_OUTPUT_C_CONTIGUOUS=False,
        FFT_BACKEND="Numpy",
    ):
        """FFT convolution with zero padding."""
        N0, N1 = PixA_Inp.shape
        L0, L1 = KERNEL.shape
        assert L0 % 2 == 1 and L1 % 2 == 1

        W0, W1 = (L0 - 1) // 2, (L1 - 1) // 2
        NX_IMG, NY_IMG = N0 + 2 * W0, N1 + 2 * W1

        PixA_EInp = np.pad(PixA_Inp, pad_width=((W0, W0), (W1, W1)), mode="constant", constant_values=PAD_FILL_VALUE)

        if NAN_FILL_VALUE is not None:
            PixA_EInp[np.isnan(PixA_EInp)] = NAN_FILL_VALUE

        KIMG_CSZ = Numpy_FFTKits.KERNEL_CSZ(
            KERNEL=KERNEL, NX_IMG=NX_IMG, NY_IMG=NY_IMG, NORMALIZE_KERNEL=NORMALIZE_KERNEL
        )

        PixA_Out = np.fft.ifft2(np.fft.fft2(PixA_EInp) * np.fft.fft2(KIMG_CSZ)).real[W0:-W0, W1:-W1]

        if FORCE_OUTPUT_C_CONTIGUOUS and not PixA_Out.flags["C_CONTIGUOUS"]:
            PixA_Out = np.ascontiguousarray(PixA_Out)
        return PixA_Out
