import math  # Using math module functions is faster for scalars
import numpy as np

from sfft.utils.NumpyFFTKits import Numpy_FFTKits
from sfft.utils.ReadWCS import Read_WCS

__last_update__ = "2026-05-29"
__author__ = "Lei Hu <leihu@andrew.cmu.edu>, Michael Wood-Vasey <wmwv@pitt.edu>, and VS Code Copilot"


class Numpy_Resampling:
    def __init__(self, RESAMP_METHOD="BILINEAR", VERBOSE_LEVEL=2):
        self.RESAMP_METHOD = RESAMP_METHOD
        self.VERBOSE_LEVEL = VERBOSE_LEVEL

    def resamp_projection_astropy(self, hdr_obj, hdr_targ):
        """Map target pixel positions into the object image frame.

        This method uses Astropy WCS to project each target pixel center from
        the target image coordinate system into the object image coordinate
        system. The resulting coordinate arrays are used for resampling the
        object image onto the target grid.

        Parameters
        ----------
        hdr_obj : dict-like
            FITS header or WCS header for the object image.
        hdr_targ : dict-like
            FITS header or WCS header for the target image.

        Returns
        -------
        tuple of ndarray
            Two arrays (XX_proj, YY_proj) of projected x/y coordinates in the
            object frame, shaped (NAXIS1, NAXIS2).
        """
        NTX = int(hdr_targ["NAXIS1"])
        NTY = int(hdr_targ["NAXIS2"])

        XX_targ, YY_targ = np.meshgrid(
            np.arange(1, NTX + 1, dtype=np.float64), np.arange(1, NTY + 1, dtype=np.float64), indexing="ij"
        )
        XY_targ = np.vstack((XX_targ.ravel(), YY_targ.ravel())).T

        w_targ = Read_WCS.RW(hdr_targ, VERBOSE_LEVEL=self.VERBOSE_LEVEL)
        w_obj = Read_WCS.RW(hdr_obj, VERBOSE_LEVEL=self.VERBOSE_LEVEL)

        XY_world = w_targ.all_pix2world(XY_targ, 1)
        XY_proj = w_obj.all_world2pix(XY_world, 1)

        XX_proj = XY_proj[:, 0].reshape((NTX, NTY))
        YY_proj = XY_proj[:, 1].reshape((NTX, NTY))

        return XX_proj, YY_proj

    def frame_extension(self, XX_proj, YY_proj, PixA_obj, PAD_FILL_VALUE=0.0, NAN_FILL_VALUE=0.0):
        """Pad the object image so it fully covers the projected target footprint.

        The object image is extended with constant padding so that all coordinates
        that project onto the target image are defined in the padded object array.
        This avoids out-of-bounds sampling during the resampling while
        preserves the original object image values within the valid region.

        Parameters
        ----------
        XX_proj : ndarray
            Projected x coordinates of target pixels in the object frame.
        YY_proj : ndarray
            Projected y coordinates of target pixels in the object frame.
        PixA_obj : ndarray
            Original object image to be padded.
        PAD_FILL_VALUE : float, optional
            Value used to pad regions outside the original object image.
        NAN_FILL_VALUE : float or None, optional
            Value used to replace NaNs in the padded object image.

        Returns
        -------
        tuple
            `PixA_Eobj`, the padded object image, and `EProjDict`, the
            metadata dictionary needed for resampling.
        """
        NTX, NTY = XX_proj.shape
        NOX, NOY = PixA_obj.shape

        if self.RESAMP_METHOD == "BILINEAR":
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
        PixA_Eobj = np.pad(PixA_obj, PAD_WIDTH, mode="constant", constant_values=PAD_FILL_VALUE)
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
            "YY_Eproj": YY_Eproj,
        }

        return PixA_Eobj, EProjDict

    def resampling(
        self,
        PixA_Eobj,
        EProjDict,
        PIXEL_SCALE_FACTOR=1.0,
        **kwargs,
   ):
        """Perform bilinear resampling of an extended object image.

        The method uses precomputed projected coordinates from `frame_extension`
        and interpolates the extended image onto the target grid. The output is
        scaled by `PIXEL_SCALE_FACTOR` to account for pixel-area changes from
        resampling.

        Parameters
        ----------
        PixA_Eobj : ndarray
            Extended object image padded to cover the projected target footprint.
        EProjDict : dict
            Projection metadata produced by `frame_extension`, including target
            dimensions and warped coordinates.
        PIXEL_SCALE_FACTOR : float, optional
            Multiplicative scaling factor applied after interpolation.
        **kwargs
            Allow additional keyword arguments for API compatibility with
            the GPU Cupy resampling code.

        Returns
        -------
        ndarray
            Resampled image on the target grid.
        """
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

            W11 = (1.0 - DX) * (1.0 - DY)
            W12 = (1.0 - DX) * DY
            W21 = DX * (1.0 - DY)
            W22 = DX * DY

            PixA_resamp = (
                W11 * PixA_Eobj[R1, C1] + W12 * PixA_Eobj[R1, C2] + W21 * PixA_Eobj[R2, C1] + W22 * PixA_Eobj[R2, C2]
            )
            PixA_resamp *= PIXEL_SCALE_FACTOR
        else:
            raise ValueError("Unsupported RESAMP_METHOD '%s'" % self.RESAMP_METHOD)

        return PixA_resamp


class Numpy_ZoomRotate:
    @staticmethod
    def CZR(
        PixA_obj,
        ZOOM_SCALE_X=1.0,
        ZOOM_SCALE_Y=1.0,
        OUTSIZE_PARITY_X="UNCHANGED",
        OUTSIZE_PARITY_Y="UNCHANGED",
        PATTERN_ROTATE_ANGLE=0.0,
        RESAMP_METHOD="BILINEAR",
        PAD_FILL_VALUE=0.0,
        NAN_FILL_VALUE=0.0,
        VERBOSE_LEVEL=2,
        **kwargs,
    ):
        """Zoom and rotate an object image using NumPy resampling.

        The image is first resized according to the requested zoom factors and
        then rotated around its center by the specified pattern rotation angle.
        This is done with a backward mapping from output pixel coordinates to
        input coordinates, followed by bilinear interpolation via
        `Numpy_Resampling`.

        Parameters
        ----------
        PixA_obj : ndarray
            Input object image to be zoomed and rotated.
        ZOOM_SCALE_X : float, optional
            Scale factor along the x-axis (horizontal direction).
        ZOOM_SCALE_Y : float, optional
            Scale factor along the y-axis (vertical direction).
        OUTSIZE_PARITY_X : {'UNCHANGED', 'ODD', 'EVEN'}, optional
            Desired parity of the output width.
            'UNCHANGED' preserves the input parity, while 'ODD' and 'EVEN'
            enforce the corresponding output dimension parity.
        OUTSIZE_PARITY_Y : {'UNCHANGED', 'ODD', 'EVEN'}, optional
            Desired parity of the output height.
        PATTERN_ROTATE_ANGLE : float, optional
            Rotation angle in degrees, applied after zooming.
        RESAMP_METHOD : str, optional
            Resampling method to use for the final interpolation.
            Currently only 'BILINEAR' is supported.
        PAD_FILL_VALUE : float, optional
            Constant value used to pad the object image before interpolation.
        NAN_FILL_VALUE : float or None, optional
            Value used to replace NaN values in the extended object image.
        VERBOSE_LEVEL : int, optional
            Verbosity level for warnings and status messages.
        **kwargs
            Additional keyword arguments accepted for compatibility with
            other backend implementations.

        Returns
        -------
        ndarray
            Zoomed and rotated image on the resampled target grid.
        """
        assert ZOOM_SCALE_X > 0.0
        assert ZOOM_SCALE_Y > 0.0
        assert 0.0 <= PATTERN_ROTATE_ANGLE < 360.0

        NAXIS1_ORI, NAXIS2_ORI = PixA_obj.shape

        def NPIX_ORI2ZOOMED(NPIX_ORI, ZOOM_SCALE, OUTSIZE_PARITY):
            if OUTSIZE_PARITY == "UNCHANGED":
                OUTSIZE_UPARITY = "EVEN" if NPIX_ORI % 2 == 0 else "ODD"
            elif OUTSIZE_PARITY == "ODD":
                OUTSIZE_UPARITY = "ODD"
            elif OUTSIZE_PARITY == "EVEN":
                OUTSIZE_UPARITY = "EVEN"
            else:
                raise ValueError("Invalid OUTSIZE_PARITY value: %s" % OUTSIZE_PARITY)

            HALFWIDTH_ORI = NPIX_ORI / 2.0
            if OUTSIZE_UPARITY == "EVEN":
                return 2 * math.ceil(HALFWIDTH_ORI / ZOOM_SCALE)
            return 2 * math.ceil((HALFWIDTH_ORI - ZOOM_SCALE / 2.0) / ZOOM_SCALE) + 1

        NAXIS1_ZOOMED = NPIX_ORI2ZOOMED(
            NPIX_ORI=NAXIS1_ORI, ZOOM_SCALE=ZOOM_SCALE_X, OUTSIZE_PARITY=OUTSIZE_PARITY_X
        )
        NAXIS2_ZOOMED = NPIX_ORI2ZOOMED(
            NPIX_ORI=NAXIS2_ORI, ZOOM_SCALE=ZOOM_SCALE_Y, OUTSIZE_PARITY=OUTSIZE_PARITY_Y
        )

        def BACKWARD_TRANSFORM(X_ROTATED, Y_ROTATED):
            CRPIX1_ZOOMED = 0.5 + NAXIS1_ZOOMED / 2.0
            CRPIX2_ZOOMED = 0.5 + NAXIS2_ZOOMED / 2.0

            ALPHA_INV = np.deg2rad(-PATTERN_ROTATE_ANGLE)
            ROTMAT_INV = np.array(
                [[np.cos(ALPHA_INV), -np.sin(ALPHA_INV)], [np.sin(ALPHA_INV), np.cos(ALPHA_INV)]], dtype=np.float64
            )

            COORD_ROTATED = np.vstack((X_ROTATED - CRPIX1_ZOOMED, Y_ROTATED - CRPIX2_ZOOMED))
            COORD_ZOOMED = ROTMAT_INV.dot(COORD_ROTATED)
            COORD_ZOOMED[0, :] += CRPIX1_ZOOMED
            COORD_ZOOMED[1, :] += CRPIX2_ZOOMED

            OFFSET_X = ZOOM_SCALE_X * (0.5 + NAXIS1_ZOOMED / 2.0) - (0.5 + NAXIS1_ORI / 2.0)
            OFFSET_Y = ZOOM_SCALE_Y * (0.5 + NAXIS2_ZOOMED / 2.0) - (0.5 + NAXIS2_ORI / 2.0)

            X_ORI = COORD_ZOOMED[0, :] * ZOOM_SCALE_X - OFFSET_X
            Y_ORI = COORD_ZOOMED[1, :] * ZOOM_SCALE_Y - OFFSET_Y
            return X_ORI, Y_ORI

        XX_targ, YY_targ = np.meshgrid(
            np.arange(1, NAXIS1_ZOOMED + 1, dtype=np.float64),
            np.arange(1, NAXIS2_ZOOMED + 1, dtype=np.float64),
            indexing="ij",
        )

        X_ORI, Y_ORI = BACKWARD_TRANSFORM(X_ROTATED=XX_targ.ravel(), Y_ROTATED=YY_targ.ravel())
        XX_proj = X_ORI.reshape((NAXIS1_ZOOMED, NAXIS2_ZOOMED))
        YY_proj = Y_ORI.reshape((NAXIS1_ZOOMED, NAXIS2_ZOOMED))

        GWR = Numpy_Resampling(RESAMP_METHOD=RESAMP_METHOD, VERBOSE_LEVEL=VERBOSE_LEVEL)
        PixA_Eobj, EProjDict = GWR.frame_extension(
            XX_proj=XX_proj,
            YY_proj=YY_proj,
            PixA_obj=PixA_obj,
            PAD_FILL_VALUE=PAD_FILL_VALUE,
            NAN_FILL_VALUE=NAN_FILL_VALUE,
        )

        PIXEL_SCALE_FACTOR = ZOOM_SCALE_X * ZOOM_SCALE_Y
        PixA_resamp = GWR.resampling(
            PixA_Eobj=PixA_Eobj,
            EProjDict=EProjDict,
            PIXEL_SCALE_FACTOR=PIXEL_SCALE_FACTOR,
            **kwargs
        )

        return PixA_resamp
