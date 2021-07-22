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
model.load_state_dict(torch.load("/Users/balashov/Downloads/slim_score_1207_batch64_160x160_epoch_99_0.0992.pth", map_location="cpu"))
model.eval()
torch.onnx.export(model.module, x, "../pretrained_weights/slim-score_160_0.0992.onnx", input_names=["input1"], output_names=['output1'])


