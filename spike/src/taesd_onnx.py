"""Builds TAESD as ONNX graphs straight from the official .pth weights (no PyTorch),
then runs them with onnxruntime. Same layers as taesd.py, fast on CPU."""
import numpy as np, onnx, onnxruntime as ort
from onnx import helper, TensorProto, numpy_helper
from taesd_numpy import load_pth

class G:
    def __init__(self, params): self.p, self.nodes, self.inits, self.n = params, [], [], 0
    def name(self, s): self.n += 1; return f"{s}_{self.n}"
    def init(self, key):
        nm = key.replace(".", "_")
        if nm not in {i.name for i in self.inits}:
            self.inits.append(numpy_helper.from_array(self.p[key].astype(np.float32), nm))
        return nm
    def conv(self, x, key, bias=True, stride=1, k=3):
        ins = [x, self.init(f"{key}.weight")] + ([self.init(f"{key}.bias")] if bias else [])
        out = self.name("conv")
        pads = [1, 1, 1, 1] if k == 3 else [0, 0, 0, 0]
        self.nodes.append(helper.make_node("Conv", ins, [out], kernel_shape=[k, k], pads=pads, strides=[stride, stride]))
        return out
    def op(self, kind, ins, **kw):
        out = self.name(kind.lower()); self.nodes.append(helper.make_node(kind, ins, [out], **kw)); return out
    def relu(self, x): return self.op("Relu", [x])
    def const(self, val, nm):
        if nm not in {i.name for i in self.inits}: self.inits.append(numpy_helper.from_array(np.array(val, np.float32), nm))
        return nm
    def block(self, x, i):
        h = self.relu(self.conv(x, f"{i}.conv.0")); h = self.relu(self.conv(h, f"{i}.conv.2")); h = self.conv(h, f"{i}.conv.4")
        return self.relu(self.op("Add", [h, x]))      # TAESD 64->64 blocks use identity skip
    def model(self, inp, out_name, out):
        self.nodes.append(helper.make_node("Identity", [out], [out_name]))
        g = helper.make_graph(self.nodes, "taesd",
                              [helper.make_tensor_value_info(inp, TensorProto.FLOAT, [1, None, None, None])],
                              [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, None, None, None])], self.inits)
        m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]); m.ir_version = 8; onnx.checker.check_model(m); return m

def build_encoder(p):
    g = G(p); x = g.conv("image", "0"); x = g.block(x, 1)
    for down, blocks in ((2, (3, 4, 5)), (6, (7, 8, 9)), (10, (11, 12, 13))):
        x = g.conv(x, str(down), bias=False, stride=2)
        for b in blocks: x = g.block(x, b)
    return g.model("image", "latents", g.conv(x, "14"))

def build_decoder(p):
    g = G(p)
    x = g.op("Mul", [g.op("Tanh", [g.op("Mul", ["latents", g.const(1 / 3, "third")])]), g.const(3.0, "three")])
    x = g.relu(g.conv(x, "1"))
    scales = g.const([1, 1, 2, 2], "up2")
    for blocks, up in (((3, 4, 5), 7), ((8, 9, 10), 12), ((13, 14, 15), 17)):
        for b in blocks: x = g.block(x, b)
        x = g.op("Resize", [x, "", scales], mode="nearest")
        x = g.conv(x, str(up), bias=False)
    x = g.block(x, 18)
    x = g.op("Clip", [g.conv(x, "19"), g.const(0.0, "zero"), g.const(1.0, "one")])
    return g.model("latents", "image_out", x)

class TAESD:
    def __init__(self, enc_path, dec_path):
        o = ort.SessionOptions(); o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.enc = ort.InferenceSession(build_encoder(load_pth(enc_path)).SerializeToString(), o, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(build_decoder(load_pth(dec_path)).SerializeToString(), o, providers=["CPUExecutionProvider"])
    def roundtrip(self, rgb_uint8):
        x = (rgb_uint8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        lat = self.enc.run(None, {"image": x})[0]
        y = self.dec.run(None, {"latents": lat})[0][0]
        return (y.transpose(1, 2, 0) * 255).round().astype(np.uint8), lat
