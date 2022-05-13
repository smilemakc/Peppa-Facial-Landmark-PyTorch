import torch
import torch.onnx
from models.slim import Slim, SlimScore

# x = torch.randn(1, 3, 160, 160)
# model = Slim()
# model.load_state_dict(torch.load("../pretrained_weights/slim_160_latest.pth", map_location="cpu"))
# model.eval()
# torch.onnx.export(model, x, "../pretrained_weights/slim_160_latest.onnx", input_names=["input1"], output_names=['output1'])

x = torch.randn(1, 3, 160, 160)
model = SlimScore()
model = torch.nn.DataParallel(model)
model.load_state_dict(
    torch.load(
        "/Users/balashov/Documents/landmarks/simpleface/slim_score_1207_batch64_160x160_epoch_99_0.0992.pth",
        map_location="cpu",
    )
)
model.eval()
torch.onnx.export(
    model.module,
    x,
    "../pretrained_weights/slim-score_160_0.0992_out_as_dict2.onnx",
    input_names=["input1"],
    output_names={
        "landmarks": [1, 136],
        "pose": [1, 3],
        "expressions": [1, 4],
        "score": [1, 1],
    },
)

# import tensorflow as tf
# import tensorflowjs as tfjs
# import onnx
# from onnx_tf.backend import prepare
#
# onnx_model = onnx.load("/Users/balashov/PycharmProjects/Peppa-Facial-Landmark-PyTorch/pretrained_weights/slim-score_160_0.0992_out_as_dict.onnx")
# tf_rep = prepare(onnx_model)  # returns: A TensorflowRep class object representing the ONNX model
# tf_rep.export_graph("/Users/balashov/PycharmProjects/Peppa-Facial-Landmark-PyTorch/pretrained_weights/slim-score_160_0.0992_out_as_dict-tf")
#
# mdl2 = tf.saved_model.load("/Users/balashov/PycharmProjects/Peppa-Facial-Landmark-PyTorch/pretrained_weights/slim-score_160_0.0992_out_as_dict-tf")
# tfjs.converters.convert_tf_saved_model("/Users/balashov/PycharmProjects/Peppa-Facial-Landmark-PyTorch/pretrained_weights/slim-score_160_0.0992_out_as_dict-tf",
#                                        "/Users/balashov/PycharmProjects/Peppa-Facial-Landmark-PyTorch/pretrained_weights/slim-score_160_0.0992_out_as_dict-tfjs")
