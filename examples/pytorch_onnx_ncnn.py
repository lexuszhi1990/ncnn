
import io
import numpy as np
import torch.onnx

import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
}


class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x))
        ], 1)


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000):
        super(SqueezeNet, self).__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        if version == 1.0:
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(512, 64, 256, 256),
            )
        else:
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=False),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        # Final convolution is initialized differently form the rest
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            final_conv,
            nn.ReLU(inplace=True),
            nn.AvgPool2d(13)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight.data, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.view(x.size(0), self.num_classes)


def squeezenet1_1(pretrained=False, **kwargs):
    r"""SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SqueezeNet(version=1.1, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load('squeezenet1_1-f364aa15.pth', map_location=lambda storage, loc: storage))
        # model.load_state_dict(model_zoo.load_url(model_urls['squeezenet1_1']))
    return model


torch_model = squeezenet1_1(True)
torch_model.eval()
from torch.autograd import Variable
batch_size = 1    # just a random number

# Input to the model
inputs = torch.randn(batch_size, 3, 224, 224)
x = Variable(inputs, requires_grad=True)
torch_out = torch.onnx.export(torch_model, x, 'squeezenet.onnx')

# Export the model
# torch_out = torch.onnx._export(torch_model,             # model being run
#                                x,                       # model input (or a tuple for multiple inputs)
#                                "squeezenet.onnx",       # where to save the model (can be a file or file-like object)
#                                export_params=True)      # store the trained parameter weights inside the model file

# input_names = [ "actual_input_1" ]
# output_names = [ "output1" ]
# torch_out = torch.onnx.export(torch_model, x, 'squeezenet.onnx', verbose=True, input_names=input_names, output_names=output_names)


import cv2
npimg = cv2.imread('example.png')
npimg_resize = cv2.resize(npimg, (224, 224))
npimg_inputs = np.expand_dims(npimg_resize.transpose(2, 0, 1), axis=0).astype(np.float32)

torch_results = torch_model(torch.from_numpy(npimg_inputs))

import onnx
# Load the ONNX model
model = onnx.load("squeezenet.onnx")
# Check that the IR is well formed
onnx.checker.check_model(model)
# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

import onnxruntime
session = onnxruntime.InferenceSession("squeezenet.onnx")

print("The model expects input shape: ", session.get_inputs()[0].shape)
print("The shape of the Image is: ", npimg_inputs.shape)

input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
onnx_result = session.run([label_name], {input_name: npimg_inputs})
prob = onnx_result[0]

print(torch_results[0][:5])
print(prob.ravel()[:5])


session = onnxruntime.InferenceSession("squeezenet-sim.onnx")

print("The model expects input shape: ", session.get_inputs()[0].shape)
print("The shape of the Image is: ", npimg_inputs.shape)

input_name = session.get_inputs()[0].name
label_name = session.get_outputs()[0].name
onnx_result = session.run([label_name], {input_name: npimg_inputs})
prob = onnx_result[0]
print(prob.ravel()[:5])

import pdb; pdb.set_trace()




import torch.onnx
help(torch.onnx.export)
import torch.onnx
import torchvision

# Standard ImageNet input - 3 channels, 224x224,
# values don't matter as we care about network structure.
# But they can also be real inputs.
dummy_input = torch.randn(1, 3, 224, 224)
# Obtain your model, it can be also constructed in your script explicitly
model = torchvision.models.alexnet(pretrained=True)
# Invoke export
torch.onnx.export(model, dummy_input, "alexnet.onnx")

import onnx

# Load the ONNX model
model = onnx.load("alexnet.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))


import onnxruntime
session = onnxruntime.InferenceSession("path to model")

sess = nxrun.InferenceSession("./squeezenet1.1.onnx")

print("The model expects input shape: ", sess.get_inputs()[0].shape)
print("The shape of the Image is: ", ximg.shape)

input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
result = sess.run(None, {input_name: ximg})
prob = result[0]
print(prob.ravel()[:10])




import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.onnx as torch_onnx

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), stride=1, padding=0, bias=False)

    def forward(self, inputs):
        x = self.conv(inputs)
        return torch.mean(x, dim=2)

input_shape = (3, 100, 100)

curr_dir = '.'
model_onnx_path = curr_dir + "/pytorch_model.onnx"

model = Model()
model.train(False)

dummy_input = Variable(torch.randn(1, *input_shape))
torch_onnx.export(model,
                  dummy_input,
                  model_onnx_path,
                  verbose=True)


import os

import onnx

curr_dir = '.'
model_proto = onnx.load(curr_dir + "/pytorch_model.onnx")
onnx.checker.check_model(model_proto)

graph = model_proto.graph
inputs = []
for i in graph.input:
    inputs.append(i.name)
assert inputs == ['0', '1']

params = []
for tensor_vals in graph.initializer:
    params.append(tensor_vals.name)
assert params == ['1']

nodes = []
for node in graph.node:
    nodes.append(node.op_type)
assert nodes == ['Conv', 'ReduceMean']

import pdb; pdb.set_trace()



