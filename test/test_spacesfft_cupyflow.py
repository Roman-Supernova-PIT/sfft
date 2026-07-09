import importlib.machinery
import importlib.util
import pathlib
import sys
import types
import unittest

import numpy as np


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEST_MODULE_NAMES = [
    "cupy",
    "sfft",
    "sfft.sfftcore",
    "sfft.sfftcore.SFFTSubtract",
    "sfft.sfftcore.SFFTConfigure",
    "sfft.utils",
    "sfft.utils.PureCupyFFTKits",
    "sfft.utils.PatternRotationCalculator",
    "sfft.utils.DeCorrelationCalculator",
    "sfft.utils.ResampKits",
    "sfft.utils.SkyLevelEstimator",
    "sfft.utils.SFFTSolutionReader",
    "sfft.PureCupyCustomizedPacket",
    "sfft.SpaceSFFTCupyFlow",
]


def _install_module(name, module):
    sys.modules[name] = module
    return module


def _make_package(name, path):
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    return module


def _load_repo_module(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_test_stubs():
    cupy = types.ModuleType("cupy")
    cupy.__spec__ = importlib.machinery.ModuleSpec("cupy", loader=None)
    cupy.float64 = np.float64
    cupy.complex128 = np.complex128
    cupy.array = np.array
    cupy.asnumpy = np.asarray
    cupy.isnan = np.isnan
    cupy.logical_or = np.logical_or
    _install_module("cupy", cupy)

    _install_module("sfft", _make_package("sfft", REPO_ROOT / "sfft"))
    _install_module("sfft.sfftcore", _make_package("sfft.sfftcore", REPO_ROOT / "sfft" / "sfftcore"))
    _install_module("sfft.utils", _make_package("sfft.utils", REPO_ROOT / "sfft" / "utils"))

    subtract = types.ModuleType("sfft.sfftcore.SFFTSubtract")
    subtract.GeneralSFFTSubtract_PureCupy = object
    _install_module("sfft.sfftcore.SFFTSubtract", subtract)

    configure = types.ModuleType("sfft.sfftcore.SFFTConfigure")
    configure.SingleSFFTConfigure = object
    _install_module("sfft.sfftcore.SFFTConfigure", configure)

    fftkits = types.ModuleType("sfft.utils.PureCupyFFTKits")
    fftkits.PureCupy_FFTKits = object
    _install_module("sfft.utils.PureCupyFFTKits", fftkits)

    pattern = types.ModuleType("sfft.utils.PatternRotationCalculator")
    pattern.PatternRotation_Calculator = object
    _install_module("sfft.utils.PatternRotationCalculator", pattern)

    decorr = types.ModuleType("sfft.utils.DeCorrelationCalculator")
    decorr.DeCorrelation_Calculator = object
    decorr.KERNEL_CSZ = object
    decorr.KERNEL_CSZ_INV = object
    _install_module("sfft.utils.DeCorrelationCalculator", decorr)

    resamp = types.ModuleType("sfft.utils.ResampKits")
    resamp.Cupy_ZoomRotate = object
    resamp.Cupy_Resampling = object
    _install_module("sfft.utils.ResampKits", resamp)

    sky = types.ModuleType("sfft.utils.SkyLevelEstimator")
    sky.SkyLevel_Estimator = object
    _install_module("sfft.utils.SkyLevelEstimator", sky)

    reader = types.ModuleType("sfft.utils.SFFTSolutionReader")
    reader.Realize_MatchingKernel = object
    _install_module("sfft.utils.SFFTSolutionReader", reader)


class SpaceSFFTCupyFlowForceConvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.saved_modules = {name: sys.modules.get(name) for name in TEST_MODULE_NAMES}
        _install_test_stubs()
        purecupy = _load_repo_module("sfft.PureCupyCustomizedPacket", "sfft/PureCupyCustomizedPacket.py")
        flow = _load_repo_module("sfft.SpaceSFFTCupyFlow", "sfft/SpaceSFFTCupyFlow.py")
        cls.packet = purecupy.PureCupy_Customized_Packet
        cls.flow_cls = flow.SpaceSFFT_CupyFlow

    @classmethod
    def tearDownClass(cls):
        for name in reversed(TEST_MODULE_NAMES):
            module = cls.saved_modules[name]
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def setUp(self):
        self.pccp_calls = []

        def fake_pccp(**kwargs):
            self.pccp_calls.append(kwargs)
            return np.array([1.0], dtype=np.float64), np.full(kwargs["PixA_REF_GPU"].shape, 5.0)

        self.packet.PCCP = staticmethod(fake_pccp)

    def _make_flow(self, sci_is_target):
        flow = self.flow_cls.__new__(self.flow_cls)
        flow.PixA_target_DMASK_GPU = np.ones((2, 2), dtype=np.float64)
        flow.PixA_resamp_object_DMASK_GPU = np.ones((2, 2), dtype=np.float64)
        flow.PixA_Ctarget_GPU = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        flow.PixA_Cresamp_object_GPU = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float64)
        flow.BlankMask_GPU = np.array([[False, True], [False, False]])
        flow.sci_is_target = sci_is_target
        flow.GKerHW = 3
        flow.KerPolyOrder = 2
        flow.BGPolyOrder = 0
        flow.ConstPhotRatio = True
        flow.CUDA_DEVICE_4SUBTRACT = "0"
        return flow

    def test_sci_is_target_false_uses_documented_science_convolution_token(self):
        flow = self._make_flow(sci_is_target=False)

        flow.sfft_subtraction()

        call = self.pccp_calls[0]
        self.assertEqual(call["ForceConv"], "SCI")
        self.assertIs(call["PixA_REF_GPU"], flow.PixA_Ctarget_GPU)
        self.assertIs(call["PixA_SCI_GPU"], flow.PixA_Cresamp_object_GPU)
        self.assertEqual(flow.PixA_DIFF_GPU[0, 1], 0.0)

    def test_sci_is_target_true_keeps_existing_ref_convolution_token(self):
        flow = self._make_flow(sci_is_target=True)

        flow.sfft_subtraction()

        call = self.pccp_calls[0]
        self.assertEqual(call["ForceConv"], "REF")
        self.assertIs(call["PixA_REF_GPU"], flow.PixA_Cresamp_object_GPU)
        self.assertIs(call["PixA_SCI_GPU"], flow.PixA_Ctarget_GPU)

    def test_purecupy_packet_accepts_new_as_legacy_science_alias(self):
        self.assertEqual(self.packet._normalize_forceconv("REF"), "REF")
        self.assertEqual(self.packet._normalize_forceconv("SCI"), "SCI")
        self.assertEqual(self.packet._normalize_forceconv("NEW"), "SCI")

        with self.assertRaises(AssertionError):
            self.packet._normalize_forceconv("AUTO")


if __name__ == "__main__":
    unittest.main()
