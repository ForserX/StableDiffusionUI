# RUN: %PYTHON %s

import absl.testing
import imagenet_test_data
import numpy
import test_util

model_path = "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v2_224_1.0_uint8.tflite"


class MobilenetV2Uint8Test(test_util.TFLiteModelTest):
    def __init__(self, *args, **kwargs):
        super(MobilenetV2Uint8Test, self).__init__(model_path, *args, **kwargs)

    def compare_results(self, iree_results, tflite_results, details):
        super(MobilenetV2Uint8Test, self).compare_results(
            iree_results, tflite_results, details
        )
        # Dequantize outputs.
        zero_point = details[0]["quantization_parameters"]["zero_points"][0]
        scale = details[0]["quantization_parameters"]["scales"][0]
        dequantized_iree_results = (iree_results - zero_point) * scale
        dequantized_tflite_results = (tflite_results - zero_point) * scale
        self.assertTrue(
            numpy.isclose(
                dequantized_iree_results, dequantized_tflite_results, atol=5e-3
            ).all()
        )

    def generate_inputs(self, input_details):
        return [imagenet_test_data.generate_input(self.workdir, input_details)]

    def test_compile_tflite(self):
        self.compile_and_execute()


if __name__ == "__main__":
    absl.testing.absltest.main()
