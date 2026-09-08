import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "setup" / "12_verify_xlsr_sls.py"
SPEC = importlib.util.spec_from_file_location("xlsr_sls_compatibility", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeInitializer:
    def __init__(self, name, dims):
        self.name = name
        self.dims = dims


class FakeNode:
    def __init__(self, inputs, output, name="classifier", op_type="Gemm"):
        self.input = inputs
        self.output = [output]
        self.name = name
        self.op_type = op_type


class FakeGraph:
    def __init__(self):
        self.initializer = [FakeInitializer("fc3.weight", (2, MODULE.EMBEDDING_DIM)), FakeInitializer("fc3.bias", (2,))]
        self.node = [FakeNode(["sls_embedding", "fc3.weight", "fc3.bias"], "logits")]


class FakeModel:
    graph = FakeGraph()


class XLSRSLSCompatibilityTests(unittest.TestCase):
    def test_audio_runtime_is_pinned_to_cuda_12_compatible_release(self):
        requirements = (SCRIPT.parents[2] / "requirements-audio.txt").read_text(encoding="utf-8")
        self.assertIn("onnxruntime-gpu==1.23.2", requirements)

    def test_checkpoint_identity_is_pinned(self):
        self.assertEqual(len(MODULE.MODEL_SHA256), 64)
        self.assertEqual(MODULE.WINDOW_SAMPLES, 64_600)
        self.assertEqual(MODULE.EMBEDDING_DIM, 1_024)

    def test_streaming_sha256(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                MODULE.sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )

    def test_gpu_memory_parser_counts_only_current_process(self):
        output = "123, 500\n456, 100\n123, 25\n"
        self.assertEqual(MODULE.parse_process_gpu_memory(output, 123), 525 * 1024**2)

    def test_internal_embedding_tensor_is_located_from_final_classifier(self):
        tensor, node = MODULE.locate_sls_embedding_tensor(FakeModel())
        self.assertEqual(tensor, "sls_embedding")
        self.assertEqual(node, "classifier")

    def test_supported_input_ranks(self):
        self.assertEqual(MODULE.concrete_input_shape(["batch", MODULE.WINDOW_SAMPLES]), (1, MODULE.WINDOW_SAMPLES))
        self.assertEqual(MODULE.concrete_input_shape(["batch", MODULE.WINDOW_SAMPLES, 1]), (1, MODULE.WINDOW_SAMPLES, 1))
        with self.assertRaises(RuntimeError):
            MODULE.concrete_input_shape([MODULE.WINDOW_SAMPLES])


if __name__ == "__main__":
    unittest.main()
