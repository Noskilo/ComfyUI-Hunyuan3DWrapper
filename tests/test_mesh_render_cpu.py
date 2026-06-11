import torch

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
