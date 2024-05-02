import hashlib
import struct
from typing import Any


def string_to_float32(data: str) -> Any:
    # Generate a hash for the input string
    hash_object = hashlib.sha256(data.encode())
    hash_hex = hash_object.hexdigest()

    # Convert the first 4 bytes of the hash to a float32 value
    hash_bytes = bytes.fromhex(hash_hex)[:4]
    float_value = struct.unpack('f', hash_bytes)[0]

    return float_value
