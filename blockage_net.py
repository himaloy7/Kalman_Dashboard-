# blockage_net.py
#
# Loads the ONNX blockage model at Flask startup and exposes a
# single function: infer_blockage(snr_window) -> float.
#
# The ONNX file is the reference artifact for a future Vitis AI /
# HLS port to the RFSoC PL. It is NOT directly flashable — it must
# be quantized and compiled to an .xmodel or hand-translated to
# ap_fixed<32,16> C++ for HLS synthesis. See the project notes.

import json
import os
import threading
import numpy as np

# Lazy import: if onnxruntime isn't installed, we degrade gracefully.
try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


_HERE       = os.path.dirname(os.path.abspath(__file__))
_ONNX_PATH  = os.path.join(_HERE, "blockage_gru.onnx")
_META_PATH  = os.path.join(_HERE, "blockage_gru_meta.json")

_session = None
_meta    = None
_lock    = threading.Lock()


def _load():
    """Load ONNX session and metadata once, on first use."""
    global _session, _meta

    if not _ORT_AVAILABLE:
        print("[blockage_net] onnxruntime not installed — GRU inference disabled")
        return

    if not os.path.isfile(_ONNX_PATH):
        print(f"[blockage_net] {_ONNX_PATH} not found — GRU inference disabled")
        return

    try:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1
        _session = ort.InferenceSession(_ONNX_PATH, sess_options=opts,
                                        providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"[blockage_net] failed to load ONNX model: {e}")
        _session = None
        return

    if os.path.isfile(_META_PATH):
        try:
            with open(_META_PATH) as f:
                _meta = json.load(f)
        except Exception as e:
            print(f"[blockage_net] failed to load metadata: {e}")
            _meta = None

    print(f"[blockage_net] GRU loaded from {_ONNX_PATH}")
    if _meta:
        print(f"[blockage_net]   seq_len={_meta.get('seq_len')}  "
              f"hidden={_meta.get('hidden_units')}  "
              f"mean={_meta.get('input_mean'):.2f}  std={_meta.get('input_std'):.2f}")


def is_loaded():
    """Returns True if the ONNX session is ready for inference."""
    with _lock:
        if _session is None and _ORT_AVAILABLE:
            _load()
    return _session is not None


def infer_blockage(snr_window):
    """
    snr_window : list or 1-D array of SNR values in dB. Length must
                 equal the training sequence length from the metadata.
    returns    : float in [0, 1] — probability of active/near-term
                 blockage. Returns None if the model isn't loaded.
    """
    if _session is None and _ORT_AVAILABLE:
        _load()
    if _session is None or _meta is None:
        return None

    seq_len = int(_meta["seq_len"])
    mu      = float(_meta["input_mean"])
    sig     = float(_meta["input_std"])

    arr = np.asarray(snr_window, dtype=np.float32).flatten()
    if arr.size < seq_len:
        # Zero-pad on the left with the mean so the network sees a
        # neutral history before the real samples arrive.
        pad = np.full(seq_len - arr.size, mu, dtype=np.float32)
        arr = np.concatenate([pad, arr])
    elif arr.size > seq_len:
        arr = arr[-seq_len:]

    norm = (arr - mu) / sig
    x = norm.reshape(1, seq_len, 1).astype(np.float32)

    try:
        out = _session.run(["blockage_prob"], {"snr_window": x})[0]
        return float(np.clip(out[0, 0], 0.0, 1.0))
    except Exception as e:
        print(f"[blockage_net] inference error: {e}")
        return None
