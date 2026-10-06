"""TAESD (Tiny AutoEncoder for Stable Diffusion) in plain numpy.
Loads the official .pth weights from github.com/madebyollin/taesd without PyTorch,
so the Stable Diffusion latent round-trip can be measured on any machine."""
import pickle, zipfile, numpy as np

DT = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}

def load_pth(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(storage, offset, size, stride, *a, **k):
                    item = storage.itemsize
                    return np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                                           strides=[s * item for s in stride]).copy()
                return rebuild
            if name in DT: return name
            if mod == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            if name == "_rebuild_parameter":
                return lambda data, *a: data
            raise pickle.UnpicklingError(f"unexpected {mod}.{name}")
        def persistent_load(self, pid):
            _, stype, key, _loc, _n = pid
            return np.frombuffer(z.read(f"{root}/data/{key}"), dtype=DT[stype])
    return dict(U(z.open(f"{root}/data.pkl")).load())

def conv(x, w, b=None, stride=1):
    """3x3 (padding 1) or 1x1 convolution on (C,H,W) as nine small matmuls."""
    k = w.shape[-1]; C, H, W = x.shape
    if k == 1:
        out = w[:, :, 0, 0] @ x.reshape(C, -1); Ho, Wo = H, W
    else:
        xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
        Ho, Wo = (H - 1) // stride + 1, (W - 1) // stride + 1
        out = np.zeros((w.shape[0], Ho * Wo), np.float32)
        for i in range(3):
            for j in range(3):
                out += w[:, :, i, j] @ xp[:, i:i + stride * Ho:stride, j:j + stride * Wo:stride].reshape(C, -1)
    if b is not None: out += b[:, None]
    return out.reshape(-1, Ho, Wo)

relu = lambda x: np.maximum(x, 0)

def block(x, p, i):
    h = relu(conv(x, p[f"{i}.conv.0.weight"], p[f"{i}.conv.0.bias"]))
    h = relu(conv(h, p[f"{i}.conv.2.weight"], p[f"{i}.conv.2.bias"]))
    h = conv(h, p[f"{i}.conv.4.weight"], p[f"{i}.conv.4.bias"])
    skip = conv(x, p[f"{i}.skip.weight"]) if f"{i}.skip.weight" in p else x
    return relu(h + skip)

def encode(img01, p):
    """(3,H,W) in [0,1] -> raw SD latents (4,H/8,W/8)."""
    x = conv(img01, p["0.weight"], p["0.bias"]); x = block(x, p, 1)
    for down, blocks in ((2, (3, 4, 5)), (6, (7, 8, 9)), (10, (11, 12, 13))):
        x = conv(x, p[f"{down}.weight"], None, stride=2)
        for b in blocks: x = block(x, p, b)
    return conv(x, p["14.weight"], p["14.bias"])

def decode(lat, p):
    """raw SD latents -> (3,H,W) in [0,1]."""
    x = np.tanh(lat / 3) * 3
    x = relu(conv(x, p["1.weight"], p["1.bias"]))
    for blocks, up in (((3, 4, 5), 7), ((8, 9, 10), 12), ((13, 14, 15), 17)):
        for b in blocks: x = block(x, p, b)
        x = x.repeat(2, 1).repeat(2, 2)                 # nn.Upsample(scale_factor=2), nearest
        x = conv(x, p[f"{up}.weight"], None)
    x = block(x, p, 18)
    return np.clip(conv(x, p["19.weight"], p["19.bias"]), 0, 1)

def roundtrip(rgb_uint8, enc, dec):
    x = rgb_uint8.astype(np.float32).transpose(2, 0, 1) / 255.0
    lat = encode(x, enc)
    y = decode(lat, dec)
    return (y.transpose(1, 2, 0) * 255).round().astype(np.uint8), lat
