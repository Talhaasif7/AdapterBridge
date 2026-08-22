"""Memory-mapped safetensors inspection and zero-copy byte-slice binary remapping utilities."""

import json
import math
import os
import struct
from typing import Dict
from safetensors import safe_open
from safetensors.numpy import save_file as numpy_save_file

from adapterbridge.models.manifest import TensorMetadata


def extract_tensor_headers(file_path: str) -> Dict[str, TensorMetadata]:
    """Read tensor headers only (shape/dtype/n_elements) via memory-mapped I/O.
    
    Uses framework="numpy" and math.prod so PyTorch/CUDA are not required.
    """
    if not os.path.exists(file_path):
        return {}

    manifest: Dict[str, TensorMetadata] = {}
    with safe_open(file_path, framework="numpy") as f:
        for key in f.keys():
            slice_obj = f.get_slice(key)
            shape = list(slice_obj.get_shape())
            dtype_str = str(slice_obj.get_dtype())
            manifest[key] = TensorMetadata(
                name=key,
                shape=shape,
                dtype=dtype_str,
                n_elements=math.prod(shape) if shape else 1,
            )
    return manifest


def remap_safetensors_file(src_path: str, dst_path: str, key_map: Dict[str, str]) -> None:
    """Remap tensor names in a safetensors file and write to dst_path using zero-copy binary streaming."""
    os.makedirs(os.path.dirname(os.path.abspath(dst_path)), exist_ok=True)
    
    try:
        remap_safetensors_headers_zero_copy(src_path, dst_path, key_map)
    except Exception:
        # Fallback to NumPy memory-mapped load & save if zero-copy streaming encounters custom layout
        tensors_dict = {}
        with safe_open(src_path, framework="numpy") as f:
            for key in f.keys():
                new_key = key_map.get(key, key)
                tensors_dict[new_key] = f.get_tensor(key)
        numpy_save_file(tensors_dict, dst_path)


def remap_safetensors_headers_zero_copy(src_path: str, dst_path: str, key_map: Dict[str, str]) -> None:
    """True zero-copy header remapping without loading binary tensor payloads into RAM.
    
    Reads 8-byte header length, updates JSON dictionary keys, pads header to 8-byte alignment,
    and streams raw binary byte buffers directly from source to destination file.
    """
    with open(src_path, "rb") as f_in:
        header_len_bytes = f_in.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Invalid safetensors file: smaller than 8 bytes header length.")
        
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        header_bytes = f_in.read(header_len)
        header_json = json.loads(header_bytes.decode("utf-8"))

        new_header_json = {}
        for key, val in header_json.items():
            new_key = key_map.get(key, key)
            new_header_json[new_key] = val

        new_header_bytes = json.dumps(new_header_json, separators=(",", ":")).encode("utf-8")
        
        # Pad with space characters so header length matches 8-byte alignment
        padding_needed = (8 - (len(new_header_bytes) % 8)) % 8
        if padding_needed > 0:
            new_header_bytes += b" " * padding_needed

        new_header_len = len(new_header_bytes)

        # Write to temporary file first then atomic move
        temp_dst = dst_path + ".tmp_zc"
        with open(temp_dst, "wb") as f_out:
            f_out.write(struct.pack("<Q", new_header_len))
            f_out.write(new_header_bytes)

            # Stream binary tensor buffers directly
            buf_size = 1024 * 1024  # 1MB buffer chunks
            while True:
                chunk = f_in.read(buf_size)
                if not chunk:
                    break
                f_out.write(chunk)

        os.replace(temp_dst, dst_path)
