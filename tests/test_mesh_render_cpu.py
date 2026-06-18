import torch
import pytest

from hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender


def test_torch_rasterizer_covers_triangle_and_interpolates_attributes():
    renderer = MeshRender(default_resolution=16, texture_size=16, device="cpu")
    pos = torch.tensor(
        [
            [-0.5, -0.5, 0.0, 1.0],
            [0.5, -0.5, 0.0, 1.0],
            [0.0, 0.5, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    rast_out, _ = renderer.raster_rasterize(pos, tri, [16, 16])
    covered = rast_out[0, ..., -1] > 0

    assert covered.any()
    assert torch.allclose(rast_out[0, covered, :3].sum(dim=-1), torch.ones(covered.sum()), atol=1e-5)

    attrs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    interpolated, _ = renderer.raster_interpolate(attrs[None, ...], rast_out, tri)

    assert interpolated.shape == (1, 16, 16, 3)
    assert torch.all(interpolated[0, covered] >= 0)
    assert torch.allclose(interpolated[0, covered].sum(dim=-1), torch.ones(covered.sum()), atol=1e-5)


def test_torch_rasterizer_uses_custom_rasterizer_y_axis_convention():
    renderer = MeshRender(default_resolution=16, texture_size=16, device="cpu")
    pos = torch.tensor(
        [
            [-0.8, -0.8, 0.0, 1.0],
            [0.8, -0.8, 0.0, 1.0],
            [0.0, 0.8, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    rast_out, _ = renderer.raster_rasterize(pos, tri, [16, 16])
    covered = rast_out[0, ..., -1] > 0
    apex_barycentric = rast_out[0, ..., 2].masked_fill(~covered, -1)
    apex_y = torch.argmax(apex_barycentric).item() // 16

    assert apex_y > 8


def test_mesh_render_keeps_xpu_device_when_available(monkeypatch):
    if not hasattr(torch, "xpu"):
        pytest.skip("torch was built without XPU support")

    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    renderer = MeshRender(default_resolution=16, texture_size=16, device="xpu")

    assert renderer.device.type == "xpu"
    assert renderer.raster_mode == "torch"


def test_torch_rasterizer_matches_cpu_on_xpu():
    if not hasattr(torch, "xpu") or not torch.xpu.is_available():
        pytest.skip("XPU device is not available")

    pos = torch.tensor(
        [
            [-0.8, -0.8, 0.0, 1.0],
            [0.8, -0.8, 0.0, 1.0],
            [0.0, 0.8, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    tri = torch.tensor([[0, 1, 2]], dtype=torch.int32)

    cpu_renderer = MeshRender(default_resolution=32, texture_size=32, device="cpu")
    xpu_renderer = MeshRender(default_resolution=32, texture_size=32, device="xpu")

    cpu_rast, _ = cpu_renderer.raster_rasterize(pos, tri, [32, 32])
    xpu_rast, _ = xpu_renderer.raster_rasterize(pos.to("xpu"), tri.to("xpu"), [32, 32])
    xpu_rast = xpu_rast.cpu()

    cpu_covered = cpu_rast[0, ..., -1] > 0
    xpu_covered = xpu_rast[0, ..., -1] > 0

    assert torch.equal(xpu_covered, cpu_covered)
    assert torch.allclose(xpu_rast[0, cpu_covered, :3], cpu_rast[0, cpu_covered, :3], atol=1e-5)
