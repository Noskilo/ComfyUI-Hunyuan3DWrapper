import torch

from hy3dgen.device_utils import make_generator, sanitize_compile_args, select_dtype


def test_select_dtype_uses_float32_on_cpu():
    assert select_dtype(torch.device("cpu")) == torch.float32


def test_make_generator_falls_back_to_usable_generator():
    generator = make_generator(torch.device("cpu"), 123)
    first = torch.randn((2,), generator=generator)

    generator = make_generator(torch.device("cpu"), 123)
    second = torch.randn((2,), generator=generator)

    assert torch.equal(first, second)


def test_sanitize_compile_args_replaces_cudagraphs_off_cuda():
    args = sanitize_compile_args({"backend": "cudagraphs", "mode": "default"}, torch.device("cpu"))

    assert args["backend"] == "inductor"
    assert args["mode"] == "default"
