from contextlib import nullcontext
import logging

import torch


logger = logging.getLogger(__name__)


def as_torch_device(device):
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def device_type(device):
    return as_torch_device(device).type


def is_cuda(device):
    return device_type(device) == "cuda"


def is_xpu(device):
    return device_type(device) == "xpu"


def backend_module(device):
    dtype = device_type(device)
    if dtype == "cuda" and hasattr(torch, "cuda"):
        return torch.cuda
    if dtype == "xpu" and hasattr(torch, "xpu"):
        return torch.xpu
    if dtype == "mps" and hasattr(torch, "mps"):
        return torch.mps
    return None


def _can_use_dtype(device, dtype):
    device = as_torch_device(device)
    if dtype == torch.float32:
        return True
    try:
        x = torch.ones((1,), device=device, dtype=dtype)
        _ = x + x
        return True
    except Exception:
        return False


def select_dtype(device, preferred=None):
    device = as_torch_device(device)
    if preferred is not None and _can_use_dtype(device, preferred):
        return preferred
    if device.type == "cpu":
        return torch.float32
    if device.type == "cuda":
        return torch.float16
    if device.type == "xpu":
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            if _can_use_dtype(device, dtype):
                return dtype
        return torch.float32
    if device.type == "mps":
        return torch.float16 if _can_use_dtype(device, torch.float16) else torch.float32
    return torch.float32


def make_generator(device, seed):
    device = as_torch_device(device)
    try:
        return torch.Generator(device=device).manual_seed(int(seed))
    except Exception:
        return torch.Generator(device="cpu").manual_seed(int(seed))


def autocast_if_available(device, dtype=None, enabled=True):
    device = as_torch_device(device)
    if not enabled or dtype is torch.float32:
        return nullcontext()
    try:
        available = torch.amp.autocast_mode.is_autocast_available(device.type)
    except Exception:
        available = False
    if not available:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def safe_empty_cache(device=None):
    backend = backend_module(device or "cpu")
    if backend is None:
        return
    try:
        if hasattr(backend, "empty_cache"):
            backend.empty_cache()
        elif hasattr(backend, "memory") and hasattr(backend.memory, "empty_cache"):
            backend.memory.empty_cache()
    except Exception:
        logger.debug("Failed to empty accelerator cache", exc_info=True)


def safe_reset_peak_memory_stats(device=None):
    backend = backend_module(device or "cpu")
    if backend is None:
        return
    try:
        if hasattr(backend, "reset_peak_memory_stats"):
            backend.reset_peak_memory_stats(device)
        elif hasattr(backend, "memory") and hasattr(backend.memory, "reset_peak_memory_stats"):
            backend.memory.reset_peak_memory_stats(device)
    except Exception:
        logger.debug("Failed to reset accelerator peak memory stats", exc_info=True)


def memory_stats(device=None):
    backend = backend_module(device or "cpu")
    if backend is None:
        return None
    stats = {}
    for key, fn_name in (
        ("allocated", "memory_allocated"),
        ("max_allocated", "max_memory_allocated"),
        ("max_reserved", "max_memory_reserved"),
    ):
        try:
            fn = getattr(backend, fn_name, None)
            if fn is None and hasattr(backend, "memory"):
                fn = getattr(backend.memory, fn_name)
            stats[key] = fn(device) / 1024**3
        except Exception:
            stats[key] = None
    return stats


def sanitize_compile_args(compile_args, device):
    if compile_args is None:
        return None
    args = dict(compile_args)
    if not is_cuda(device) and args.get("backend") == "cudagraphs":
        logger.info("Disabling cudagraphs torch.compile backend on non-CUDA device")
        args["backend"] = "inductor"
    return args


def should_use_diffusers_cpu_offload(device):
    return is_cuda(device)
