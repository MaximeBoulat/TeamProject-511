import ctypes
import glob
import os
import site


def load_cuda_libraries():
    for sp in site.getsitepackages():
        nvidia_path = os.path.join(sp, "nvidia")
        if os.path.exists(nvidia_path):
            lib_pattern = os.path.join(nvidia_path, "*", "lib", "*.so*")
            for lib_file in glob.glob(lib_pattern):
                try:
                    ctypes.CDLL(lib_file, mode=ctypes.RTLD_GLOBAL)
                except Exception:
                    pass  # Skip files that can't be loaded
            print(f"Loaded CUDA libraries from {nvidia_path}")
            return True
    print("No CUDA libraries found")
    return False


def configure_tensorflow():
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def test_gpu():
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"GPU setup complete - {len(gpus)} GPU(s) available")
    return len(gpus)
