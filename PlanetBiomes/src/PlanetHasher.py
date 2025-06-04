import os
import zlib


def get_directory_crc(path: str, use_backslash: bool = False) -> int:
    """
    Computes the CRC32 of the normalized folder path.
    """
    dummy_filename = "dummy_file.txt"
    full_path = os.path.join(path, dummy_filename)
    if use_backslash:
        full_path = full_path.replace("/", "\\")
    else:
        full_path = full_path.replace("\\", "/")
    folder_path = os.path.dirname(full_path).lower()  # Try lowercase
    return zlib.crc32(folder_path.encode("utf-8")) & 0xFFFFFFFF


def get_file_crc(file_name: str) -> int:
    """
    Computes the CRC32 of a file name.
    """
    return zlib.crc32(file_name.encode("utf-8")) & 0xFFFFFFFF


def get_plugin_crc(plugin_name: str) -> int:
    """
    Computes the CRC32 of the plugin name.
    """
    return zlib.crc32(plugin_name.lower().encode("utf-8")) & 0xFFFFFFFF  # Try lowercase


def get_final_crc():
    print("logic stuff")


def create_resource_id(folder_path: str, file_name: str, plugin_name: str) -> str:
    """
    Generates a full Starfield-style resource ID:
    res:<folder_crc>:<file_crc>:<plugin_crc>
    """
    folder_crc = get_directory_crc(folder_path)
    file_crc = get_file_crc(file_name)
    plugin_crc = get_plugin_crc(plugin_name)
    return f"res:{folder_crc:08X}:{file_crc:08X}:{plugin_crc:08X}"


def create_simple_crc_id(
    name: str, plugin_name: str, suffix: str = "", index: int = 0
) -> str:
    """
    Generates a simple hashed ID for internal component references.
    """
    seed = f"{name}_{suffix}_{index}"
    crc = zlib.crc32(seed.encode("utf-8")) & 0xFFFFFFFF
    plugin_crc = get_plugin_crc(plugin_name)
    return f"res:{crc:08X}:{crc:08X}:{plugin_crc:08X}"
