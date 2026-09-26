
import torch
from torch import nn
import os, sys, math
from plyfile import PlyData
import numpy as np
import time
import viser
import viser.transforms as tf
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from utils import build_scaling_rotation, strip_symmetric

class GaussianModel:
    
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, path):
        self.active_sh_degree = 3
        self.setup_functions()
        
        self.load_ply(path)
    
    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        
        opac_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("opacity")]
        # Handles both standard 3DGS ("opacity") and indexed ("opacity_0", "opacity_1", ...) layouts
        opac_names = sorted(opac_names, key = lambda x: int(x.split('_')[-1]) if '_' in x else -1)
        opacities = np.zeros((xyz.shape[0], len(opac_names)))
        for idx, attr_name in enumerate(opac_names):
            opacities[:, idx] = np.asarray(plydata.elements[0][attr_name])
                    
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.active_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.active_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])
            
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        # self._p = nn.Parameter(torch.tensor(ps, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self._opacity
        return self.opacity_activation(self._opacity)
    
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera:
    # Minimal camera matching the fields the original 3DGS render() reads.
    # w2c is a 4x4 world-to-camera matrix in COLMAP convention (x right, y down, z forward).
    def __init__(self, w2c, FoVx, FoVy, width, height, znear=0.01, zfar=100.0):
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = width
        self.image_height = height

        self.world_view_transform = torch.tensor(w2c, dtype=torch.float).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear, zfar, FoVx, FoVy).transpose(0, 1).cuda()
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0)).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def render(viewpoint_camera, pc : GaussianModel, bg_color : torch.Tensor, scaling_modifier = 1.0, temporal_ctl = 0.):
    # Same as the original 3DGS gaussian_renderer.render, using SHs and scale/rotation.
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=True
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
    rendered_image, radii = rasterizer(
        means3D = pc.get_xyz,
        means2D = screenspace_points,
        shs = pc.get_features,
        colors_precomp = None,
        opacities = pc.get_opacity,
        scales = pc.get_scaling,
        rotations = pc.get_rotation,
        time=temporal_ctl,
        cov3D_precomp = None)

    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


def viser_camera(client_camera, max_res):
    # viser gives camera-to-world in OpenCV convention (x right, y down, z forward)
    c2w = np.eye(4)
    c2w[:3, :3] = tf.SO3(np.asarray(client_camera.wxyz)).as_matrix()
    c2w[:3, 3] = client_camera.position
    w2c = np.linalg.inv(c2w)

    aspect = client_camera.aspect
    if aspect >= 1:
        width, height = max_res, int(max_res / aspect)
    else:
        width, height = int(max_res * aspect), max_res

    fovy = client_camera.fov
    fovx = 2 * math.atan(math.tan(fovy / 2) * aspect)
    return Camera(w2c, fovx, fovy, width, height)

def render_loop(gaussians, port=8080, up=(0.0, 1.0, 1.0)):
    # COLMAP scenes are usually -y up; change `up` if the view starts tilted.
    server = viser.ViserServer(port=port)
    center = gaussians.get_xyz.detach().median(dim=0).values.cpu().numpy()

    gui_res = server.gui.add_slider("Max resolution", min=128, max=2048, step=64, initial_value=1024)
    gui_scale = server.gui.add_slider("Scale modifier", min=0.01, max=1.0, step=0.01, initial_value=1.0)
    gui_white = server.gui.add_checkbox("White background", initial_value=False)
    gui_temporal = server.gui.add_slider("Temporal control", min=0.0, max=1.0, step=0.01, initial_value=0.0)

    last_rendered = {}  # client id -> state the last frame was rendered with

    @server.on_client_connect
    def _(client):
        client.camera.up_direction = up
        client.camera.look_at = center

    with torch.no_grad():
        while True:
            state_gui = (gui_res.value, gui_scale.value, gui_white.value, gui_temporal.value)
            for client_id, client in server.get_clients().items():
                state = (client.camera.update_timestamp,) + state_gui
                if last_rendered.get(client_id) == state:
                    continue
                last_rendered[client_id] = state

                cam = viser_camera(client.camera, gui_res.value)
                bg_color = torch.tensor([1, 1, 1] if gui_white.value else [0, 0, 0], dtype=torch.float32, device="cuda")
                image = render(cam, gaussians, bg_color, scaling_modifier=gui_scale.value, temporal_ctl=gui_temporal.value)["render"]
                image = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                client.scene.set_background_image(image, format="jpeg")
            time.sleep(1 / 60)

from argparse import ArgumentParser
if __name__ == "__main__":
    torch.cuda.empty_cache()
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--data', type=str, default="/data/...")
    parser.add_argument('--port', type=int, default=8080)

    args = parser.parse_args(sys.argv[1:])


    gaussians = GaussianModel(args.data)
    render_loop(gaussians, args.port)
