import onnx
model = onnx.load("policy.onnx")
print(onnx.helper.printable_graph(model.graph))