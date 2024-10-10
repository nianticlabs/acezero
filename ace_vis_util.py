# Copyright © Niantic, Inc. 2022.

import os
import logging
import numpy as np
import trimesh
import pyrender
from PIL import Image, ImageOps
from scipy.linalg import svd
from matplotlib.colors import LinearSegmentedColormap
import torch
from torch.cuda.amp import autocast
from skimage import io, color
from skimage.transform import resize
from ace_util import get_pixel_grid, to_homogeneous
from bisect import insort

logging.getLogger('trimesh').setLevel(level=logging.WARNING)
_logger = logging.getLogger(__name__)

THICKNESS = 0.005  # controls how thick the frustum's 'bars' are

# define camera frustum geometry
origin_frustum_verts = np.array([
    (0., 0., 0.),
    (0.375, -0.375, -1.0),
    (0.375, 0.375, -1.0),
    (-0.375, 0.375, -1.0),
    (-0.375, -0.375, -1.0),
])

frustum_edges = np.array([
    (1, 2),
    (1, 3),
    (1, 4),
    (1, 5),
    (2, 3),
    (3, 4),
    (4, 5),
    (5, 2),
]) - 1


def normalise_vector(vect):
    """
    Returns vector with unit length.

    @param vect: Vector to be normalised.
    @return: Normalised vector.
    """
    length = np.sqrt((vect ** 2).sum())
    return vect / length


def cuboid_from_line(line_start, line_end, color=(255, 0, 255)):
    """Approximates a line with a long cuboid
    color is a 3-element RGB tuple, with each element a uint8 value
    """
    # create two vectors which are both (a) perpendicular to the direction of the line and
    # (b) perpendicular to each other.
    direction = normalise_vector(line_end - line_start)
    random_dir = normalise_vector(np.random.rand(3))
    perpendicular_x = normalise_vector(np.cross(direction, random_dir))
    perpendicular_y = normalise_vector(np.cross(direction, perpendicular_x))

    vertices = []
    for node in (line_start, line_end):
        for x_offset in (-1, 1):
            for y_offset in (-1, 1):
                vert = node + THICKNESS * (perpendicular_y * y_offset + perpendicular_x * x_offset)
                vertices.append(vert)

    faces = [
        (4, 5, 1, 0),
        (5, 7, 3, 1),
        (7, 6, 2, 3),
        (6, 4, 0, 2),
        (0, 1, 3, 2),  # end of tube
        (6, 7, 5, 4),  # other end of tube
    ]

    mesh = trimesh.Trimesh(vertices=np.array(vertices), faces=np.array(faces))

    for c in (0, 1, 2):
        mesh.visual.vertex_colors[:, c] = color[c]

    return mesh


def generate_frustum_marker(pose, color=(255, 0, 255), size=1.):
    frustum_vertices = np.array([
        [0., 0., 0., 1.],
        [1., 1., 3., 1.],
        [-1., 1., 3., 1.],
        [-1., -1., 3., 1.],
        [1., -1., 3., 1.]
    ]).T

    frustum_vertices[:3] *= size
    frustum_vertices[2, :] *= -1  # OpenCV to OpenGL
    frustum_vertices = pose @ frustum_vertices
    frustum_vertices = frustum_vertices[:3].T

    frustum_faces = np.array([
        [0, 4, 1],
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 4],
        [4, 2, 1],
        [4, 3, 2],
    ])

    mesh = trimesh.Trimesh(vertices=frustum_vertices, faces=frustum_faces)

    for c in (0, 1, 2):
        mesh.visual.vertex_colors[:, c] = color[c]

    return mesh


def get_image_box(
        image_path,
        frustum_pose,
        cam_marker_size=1.0,
        flip=False
):
    """ Gets a textured mesh of an image.

    @param image_path: File path of the image to be rendered.
    @param frustum_pose: 4x4 camera pose, OpenGL convention
    @param cam_marker_size: Scaling factor for the image object
    @param flip: flag whether to flip the image left/right
    @return: duple, trimesh mesh of the image and aspect ratio of the image
    """

    pil_image = Image.open(image_path)
    pil_image = ImageOps.flip(pil_image)  # flip top/bottom to align with scene space

    pil_image_w, pil_image_h = pil_image.size
    aspect_ratio = pil_image_w / pil_image_h

    height = 0.75
    width = height * aspect_ratio
    width *= cam_marker_size
    height *= cam_marker_size

    if flip:
        pil_image = ImageOps.mirror(pil_image)  # flips left/right
        width = -width

    vertices = np.zeros((4, 3))
    vertices[0, :] = [width / 2, height / 2, -cam_marker_size]
    vertices[1, :] = [width / 2, -height / 2, -cam_marker_size]
    vertices[2, :] = [-width / 2, -height / 2, -cam_marker_size]
    vertices[3, :] = [-width / 2, height / 2, -cam_marker_size]

    faces = np.zeros((2, 3))
    faces[0, :] = [0, 1, 2]
    faces[1, :] = [2, 3, 0]
    # faces[2,:] = [2,3]
    # faces[3,:] = [3,0]

    uvs = np.zeros((4, 2))

    uvs[0, :] = [1.0, 0]
    uvs[1, :] = [1.0, 1.0]
    uvs[2, :] = [0, 1.0]
    uvs[3, :] = [0, 0]

    face_normals = np.zeros((2, 3))
    face_normals[0, :] = [0.0, 0.0, 1.0]
    face_normals[1, :] = [0.0, 0.0, 1.0]

    material = trimesh.visual.texture.SimpleMaterial(
        image=pil_image,
        ambient=(1.0, 1.0, 1.0, 1.0),
        diffuse=(1.0, 1.0, 1.0, 1.0),
    )
    texture = trimesh.visual.TextureVisuals(
        uv=uvs,
        image=pil_image,
        material=material,
    )

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_normals=face_normals,
        visual=texture,
        validate=True,
        process=False
    )

    # from simple recon code
    def transform_trimesh(mesh, transform):
        """ Applies a transform to a trimesh. """
        np_vertices = np.array(mesh.vertices)
        np_vertices = (transform @ np.concatenate([np_vertices, np.ones((np_vertices.shape[0], 1))], 1).T).T
        np_vertices = np_vertices / np_vertices[:, 3][:, None]
        mesh.vertices[:, 0] = np_vertices[:, 0]
        mesh.vertices[:, 1] = np_vertices[:, 1]
        mesh.vertices[:, 2] = np_vertices[:, 2]

        return mesh

    return transform_trimesh(mesh, frustum_pose), aspect_ratio


def generate_frustum_at_position(rotation, translation, color, size, aspect_ratio):
    """Generates a frustum mesh at a specified (rotation, translation), with optional color
    : rotation is a 3x3 numpy array
    : translation is a 3-long numpy vector
    : color is a 3-long numpy vector or tuple or list; each element is a uint8 RGB value
    : aspect_ratio is a float of width/height
    """

    frustum_verts = origin_frustum_verts.copy()
    frustum_verts[:, 0] *= aspect_ratio

    transformed_frustum_verts = \
        size * rotation.dot(frustum_verts.T).T + translation[None, :]

    cuboids = []
    for edge in frustum_edges:
        line_cuboid = cuboid_from_line(line_start=transformed_frustum_verts[edge[0]],
                                       line_end=transformed_frustum_verts[edge[1]],
                                       color=color)
        cuboids.append(line_cuboid)

    return trimesh.util.concatenate(cuboids)


class LazyCamera:
    """Smooth and slightly delayed scene camera.

    Implements a rolling average of last few camera positions.
    Also zooms out to display the whole scene.
    """

    def __init__(self,
                 camera_buffer_size=40,
                 backwards_offset=4,
                 camera_buffer=None):
        """Constructor.

        Parameters:
            camera_buffer_size: Number of last few cameras to consider
            backwards_offset: Move observing camera backwards from current view, in meters
            camera_buffer: Optional array of camera positions to pre-fill the buffer
        """

        # buffer holding last m camera positions
        if camera_buffer is None:
            self.m_camera_buffer = []
        else:
            self.m_camera_buffer = camera_buffer

        self.m_camera_buffer_size = camera_buffer_size
        self.m_backwards_offset = backwards_offset

    def _orthonormalize_rotation(self, T):
        """Takes a 4x4 matrix and orthonormalizes the upper left 3x3 using SVD

        Returns:
            T with orthonormalized upper 3x3
        """

        R = T[:3, :3]
        t = T[:3, 3]

        # see https://arxiv.org/pdf/2006.14616.pdf Eq.2
        U, S, Vt = svd(R)
        Z = np.eye(3)
        Z[-1, -1] = np.sign(np.linalg.det(U @ Vt))
        R = U @ Z @ Vt

        T = np.eye(4)  # recreate the matrix to make sure that the forth row is [0 0 0 1]
        T[:3, :3] = R
        T[:3, 3] = t

        return T

    def update_camera(self, view):
        """Update lazy camera with new view.

        Parameters:
            view: New camera view, 4x4 matrix
        """

        observing_camera = view.copy()

        # push observing camera back in z-direction in camera space
        z_vec = np.zeros((3,))
        z_vec[2] = 1
        offset_vector = view[:3, :3] @ z_vec
        observing_camera[:3, 3] += offset_vector * self.m_backwards_offset

        # use moving avage of last X cameras (so that observing camera is smooth and follows with slight delay)
        self.m_camera_buffer.append(observing_camera)

        if len(self.m_camera_buffer) > self.m_camera_buffer_size:
            self.m_camera_buffer = self.m_camera_buffer[1:]

    def get_current_view(self):
        """Get current lazy camera view for rendering.

        Returns:
            4x4 matrix
        """

        # naive average of camera pose matrices
        smooth_camera_pose = np.zeros((4, 4))
        for camera_pose in self.m_camera_buffer:
            smooth_camera_pose += camera_pose
        smooth_camera_pose /= len(self.m_camera_buffer)

        return self._orthonormalize_rotation(smooth_camera_pose)

    def get_camera_buffer(self):
        """
        Return buffered camera views, e.g. for storing state.
        """
        return self.m_camera_buffer


class PointCloudBuffer:
    """Holds last N point clouds."""

    def __init__(self, pc_buffer_size=5):
        """Constructor.

        Parameters:
            pc_buffer_size: Number of last N point clouds to hold
        """

        self.pc_buffer_size = pc_buffer_size

        self.pc_xyz_buffer = []
        self.pc_clr_buffer = []
        self.pc_err_buffer = []

    def update_buffer(self, pc_xyz, pc_clr, pc_errs=None):
        """
        Add a new (partial) point cloud to the buffer.

        @param pc_xyz: N3, coordinates of points
        @param pc_clr: N3, RGB colors of points
        @param pc_errs: N1, scalar errors of points
        """
        self.pc_xyz_buffer.append(pc_xyz)
        self.pc_clr_buffer.append(pc_clr)

        if pc_errs is not None:
            self.pc_err_buffer.append(pc_errs)

        # remove oldest xyz and clr entries in the buffer if buffer is full
        if 0 < self.pc_buffer_size < len(self.pc_xyz_buffer):
            self.pc_xyz_buffer = self.pc_xyz_buffer[1:]
            self.pc_clr_buffer = self.pc_clr_buffer[1:]

        # errs handled separately, because optional
        if 0 < self.pc_buffer_size < len(self.pc_err_buffer):
            self.pc_err_buffer = self.pc_err_buffer[1:]

    def get_point_cloud(self):
        """
        Merges and returns all point clouds in the buffer.

        @return: triple, N3 xyz + N3 colors + N1 errors
        """
        # combine PC chunks of current frame to single PC
        merged_xyz = np.concatenate(self.pc_xyz_buffer)
        merged_clr = np.concatenate(self.pc_clr_buffer)

        if len(self.pc_err_buffer) > 0:
            merged_errs = np.concatenate(self.pc_err_buffer)
        else:
            merged_errs = None

        return merged_xyz, merged_clr, merged_errs

    def disable_buffer_cap(self):
        """
        Switch rolling buffer of fixed size to unconstrained buffer.
        """
        self.pc_buffer_size = -1


def get_retro_colors():
    """
    Create custom color map, dark magenta to bright cyan.

        if you like this color map and use it in your own work, let me know
        https://twitter.com/eric_brachmann
        looking forward to seeing what you do with it :)
        -- Eric

    @return: Color lookup table, 256x3
    """

    cdict = {'red': [
        [0.0, 0.073, 0.073],
        [0.4, 0.325, 0.325],
        [0.7, 0.286, 0.286],
        [0.85, 0.266, 0.266],
        [0.95, 0, 0],
        [1, 1, 1],
    ],
        'green': [
            [0.0, 0.0, 0.0],
            [0.4, 0.058, 0.058],
            [0.7, 0.470, 0.470],
            [0.85, 0.827, 0.827],
            [0.95, 1, 1],
            [1, 1, 1],
        ],
        'blue': [
            [0.0, 0.057, 0.057],
            [0.4, 0.223, 0.223],
            [0.7, 0.752, 0.752],
            [0.85, 0.988, 0.988],
            [0.95, 1, 1],
            [1, 1, 1],
        ]}

    retroColorMap = LinearSegmentedColormap('retroColors', segmentdata=cdict, N=256)

    return retroColorMap(np.linspace(0, 1, 257))[1:, :3]


def get_point_cloud_from_network(network, data_loader, filter_depth, dense_cloud=False):
    """
    Extract a point cloud from a fully trained network.

    @param network: scene coordinate regression network
    @param data_loader: loader for the mapping sequence
    @param filter_depth: in meters, remove points further from the camera
    @param dense_cloud: if True, return all points (good to initialise splats), otherwise filter based on repro error
    @return: tuple, N3 coordinates + N3 RGB colors
    """

    # remove points where scene coordinates change more than this threshold from one pixel to the next (in meters)
    # since scene can have vastly different scales, and scales are estimates, we try increasingly relaxed thresholds
    grad_thresholds = [0.1, 0.5, 1.0, torch.inf]

    # total number of points in the point cloud, at least min even with large re-projection errors
    # at most max, even if more points have small re-projection errors
    pc_points_min = 100000
    pc_points_max = 1000000

    # remove points with re-projection larger than threshold (in px) as long as we keep a min number of points
    repro_threshold = 1

    if dense_cloud:
        # disable checks to return random points per image
        grad_thresholds = [torch.inf]
        repro_threshold = torch.inf

    pc_points_per_image_min = int(pc_points_min / len(data_loader))
    pc_points_per_image_max = int(pc_points_max / len(data_loader))

    pixel_grid = get_pixel_grid(network.OUTPUT_SUBSAMPLE)  # Shape: 2x5000x5000

    pc_xyz = []
    pc_clr = []

    with torch.no_grad():

        # iterate over mapping sequence
        for image, _, gt_inv_pose, _, K, _, _, file, _ in data_loader:

            # predict scene coordinate
            image = image.cuda(non_blocking=True)
            gt_inv_pose = gt_inv_pose.cuda(non_blocking=True)
            K = K.cuda(non_blocking=True)

            with autocast():
                scene_coords = network(image)

            B, C, H, W = scene_coords.shape

            assert B == 1, "Batch size must be 1 for point cloud extraction."

            # scene coordinate to camera coordinates
            pred_scene_coords_B3HW = scene_coords.float()
            pred_scene_coords_B4N = to_homogeneous(pred_scene_coords_B3HW.flatten(2))
            pred_cam_coords_B3N = torch.matmul(gt_inv_pose[:, :3], pred_scene_coords_B4N)

            # project scene coordinates
            pred_px_B3N = torch.matmul(K, pred_cam_coords_B3N)
            pred_px_B3N[:, 2].clamp_(min=0.1)  # avoid division by zero
            pred_px_B2N = pred_px_B3N[:, :2] / pred_px_B3N[:, 2, None]

            # measure reprojection error
            pixel_positions_2HW = pixel_grid[:, :H, :W].clone()  # Crop to actual size
            pixel_positions_2N = pixel_positions_2HW.view(2, -1)

            reprojection_error_2N = pred_px_B2N.squeeze() - pixel_positions_2N.cuda()
            reprojection_error_1N = torch.norm(reprojection_error_2N, dim=0, keepdim=True, p=1)

            # filter based on gradient of scene coordinates
            grad_x_BHW = torch.linalg.norm(pred_scene_coords_B3HW[:, :, :, 1:] - pred_scene_coords_B3HW[:, :, :, :-1],
                                           dim=1)
            grad_x_BHW = torch.nn.functional.pad(grad_x_BHW, (1, 0), mode='reflect')
            grad_y_BHW = torch.linalg.norm(pred_scene_coords_B3HW[:, :, 1:, :] - pred_scene_coords_B3HW[:, :, :-1, :],
                                           dim=1)
            grad_y_BHW = torch.nn.functional.pad(grad_y_BHW, (0, 0, 1, 0), mode='reflect')

            grad_BHW = torch.max(grad_x_BHW, grad_y_BHW)
            grad_1N = grad_BHW.view(B, -1)

            # try different grad thresholds, keep the tightest one that still has enough points
            for grad_threshold in grad_thresholds:
                sc_grad_mask = grad_1N.squeeze() < grad_threshold
                if sc_grad_mask.sum() > pc_points_per_image_min:
                    break

            # filter predictions based on depth
            sc_depth_mask = pred_cam_coords_B3N[0, 2] < filter_depth

            sc_grad_and_depth_mask = torch.logical_and(sc_grad_mask, sc_depth_mask)

            # if no points survive, keep all
            if sc_grad_and_depth_mask.sum() == 0:
                sc_grad_and_depth_mask[:] = True

            # apply reprojection error
            sc_err_mask = reprojection_error_1N.squeeze() < repro_threshold
            sc_err_mask = torch.logical_and(sc_err_mask, sc_grad_and_depth_mask)

            # check whether enough point survive
            num_valid_points = int(sc_err_mask.sum())

            if num_valid_points < pc_points_per_image_min:
                # take min points with lowest reprojection error
                reprojection_error_within_range_and_smooth_1N = reprojection_error_1N.squeeze()[sc_grad_and_depth_mask]

                sorted_errors, _ = torch.sort(reprojection_error_within_range_and_smooth_1N)
                relaxed_filter_repro_error = sorted_errors[min(pc_points_per_image_min, sorted_errors.shape[0] - 1)]

                sc_err_mask = reprojection_error_1N.squeeze() < relaxed_filter_repro_error
                sc_err_mask = torch.logical_and(sc_grad_and_depth_mask, sc_err_mask)
            elif num_valid_points > pc_points_per_image_max:
                # sub-sample points
                keep_ratio = pc_points_per_image_max / num_valid_points
                sub_sample_mask = torch.randperm(num_valid_points) < int(keep_ratio * num_valid_points)
                sc_err_mask_subsampled = sc_err_mask.clone()
                sc_err_mask_subsampled[sc_err_mask] = sub_sample_mask.cuda()
                sc_err_mask = sc_err_mask_subsampled

            # load image file to extract colors
            rgb = io.imread(file[0])

            if len(rgb.shape) < 3:
                rgb = color.gray2rgb(rgb)

            # align RGB values with scene coordinate prediction
            rgb = rgb.astype('float64')
            # firstly, resize image to network input resolution
            rgb = resize(rgb, image.shape[2:])
            # secondly, sub-sampling to network output resolution
            # using nearest neighbour subsampling results in slightly crisper colors
            nn_stride = network.OUTPUT_SUBSAMPLE
            nn_offset = network.OUTPUT_SUBSAMPLE // 2
            rgb = rgb[nn_offset::nn_stride, nn_offset::nn_stride, :]
            # make sure the resolution fits (catch any striding mismatches)
            rgb = resize(rgb, scene_coords.shape[2:])
            rgb = torch.from_numpy(rgb).permute(2, 0, 1)
            rgb = rgb.contiguous().view(3, -1)

            # remove invalid map points
            rgb = rgb[:, sc_err_mask.cpu()]
            xyz = pred_scene_coords_B4N[0, :3, sc_err_mask].cpu()

            pc_xyz.append(xyz.numpy())
            pc_clr.append(rgb.numpy())

    # merge points
    pc_xyz = np.concatenate(pc_xyz, axis=1)
    pc_clr = np.concatenate(pc_clr, axis=1)

    # 3N to N3
    pc_xyz = np.transpose(pc_xyz)
    pc_clr = np.transpose(pc_clr)

    # OpenCV to OpenGL convention
    pc_xyz[:, 1] = -pc_xyz[:, 1]
    pc_xyz[:, 2] = -pc_xyz[:, 2]

    # return merged frame points
    return pc_xyz, pc_clr


def get_rendering_target_path(target_base_path, map_file_name):
    """
    Infer a folder for renderings from a base path and a map name.

    Creates target folder if it does not exist.

    @param target_base_path: Base path for all renderings.
    @param map_file_name: Map file name to infer folder name for renderings of this mapping run.
    @return: path to store renderings
    """
    target_path = map_file_name  # infer rendering folder from map file name
    target_path = os.path.basename(target_path)  # extract file name
    target_path = os.path.splitext(target_path)[0]  # remove extension
    target_path = target_base_path / target_path

    os.makedirs(target_path, exist_ok=True)

    return target_path


class CameraTrajectoryBuffer:
    """Incrementally builds a camera trajectory mesh."""

    def __init__(self,
                 frustum_skip,
                 frustum_scale):
        """
        Constructor.

        Initialises standard values.

        @param frustum_skip: minimum distance between placing frustums, in meters
        @param frustum_scale: Scaling factor for camera frustums
        """
        self.frustum_skip = frustum_skip
        self.frustum_scale = frustum_scale

        self.trajectory = []  # holds line segments to render the camera path of the mapping sequence
        self.frustums = []  # holds frustum geometry for the trajectory
        self.frustum_images = []  # frustum images need to be kept extra due to image texture

        self.trajectory_previous = None  # holds last camera position to skip segments if camera jumps
        self.frustum_positions = []  # holds accepted frustum placement positions to sparsify them
        self.trajectory_distances = []  # holds all previous distances in the trajectory to detect jumps

        self.trajectory_color = (255, 255, 255)
        self.aspect_ratio_buffer = 4 / 3  # default aspect ratio, overwritten as soon as a acutal image is loaded

    def grow_camera_path(self, new_camera):
        """
        Expand the camera trajectory line wrt new camera.

        Keeps track of camera movement statistics and skips the line if a camera jump is detected.

        @param new_camera: 4x4 camera pose, OpenGL convention
        """
        # get position of mapping camera
        current_pos = new_camera[:3, 3]

        # draw line from previous position to current position
        if self.trajectory_previous is not None:

            current_dist = np.linalg.norm(current_pos - self.trajectory_previous)
            # keep sorted list of previous camera distance
            insort(self.trajectory_distances, current_dist)
            # detect jump if current dist is more than X times the median
            line_skip = 10 * self.trajectory_distances[len(self.trajectory_distances) // 2]

            if 0.0001 < current_dist < line_skip:
                line_cuboid = cuboid_from_line(line_start=self.trajectory_previous,
                                               line_end=current_pos,
                                               color=self.trajectory_color)

                self.trajectory.append(line_cuboid)
            else:
                if current_dist > line_skip:
                    _logger.info(f"Detected jump: camera dist={current_dist:.3f}, threshold={line_skip:.3f}, "
                                 f"threshold estimated from {len(self.trajectory_distances)} estimates.")

        # update previous position for next iteration
        self.trajectory_previous = current_pos

    def add_position_marker(self, marker_pose, marker_color, marker_extent=0.015, frustum_maker=False):
        """
        Adds a cube to the trajectory mesh to signify a singular camera position.

        @param marker_pose: 4x4 camera pose, OpenGL convention
        @param marker_color: RGB color of the marker
        @param marker_extent: size of the marker, marker is a cube of this side length
        """
        if frustum_maker:
            current_pos_marker = generate_frustum_marker(marker_pose, marker_color, marker_extent)
        else:
            current_pos_marker = trimesh.primitives.Box(
                extents=(marker_extent, marker_extent, marker_extent),
                transform=marker_pose)
            for c in (0, 1, 2):
                current_pos_marker.visual.vertex_colors[:, c] = marker_color[c]

        self.trajectory.append(current_pos_marker)

    def _get_closest_frustum_distance(self, new_camera):
        """
        Calculate distance to the closest, previously placed frustum in the trajectory so far.

        @param new_camera: 4x4 camera, OpenGL convention
        @return: distance to the closest frustum in the trajectory
        """
        if len(self.frustum_positions) == 0:
            return self.frustum_skip + 1  # hack, return a distance that always accepts the new camera
        else:
            distances = [np.linalg.norm(pos - new_camera[:3, 3]) for pos in self.frustum_positions]
            return min(distances)

    def add_camera_frustum(self, camera, image_file=None, sparse=True, frustum_color=None):
        """
        Add a camera frustum object to the trajectory, minding distance to existing frustums.

        @param camera: 4x4 camera pose, OpenGL convention
        @param image_file: path to image to be displayed in frustum
        @param sparse: flag, if true a frustum is not placed if too close to existing frustums
        @param frustum_color: RGB color, if none default color is used
        """
        new_camera = camera.copy()

        if frustum_color is None:
            frustum_color = self.trajectory_color

        # place camera frustum all X centimeters (or overwrite via sparse flag)
        if (sparse == False) or (self._get_closest_frustum_distance(new_camera) > self.frustum_skip):

            if image_file is not None:
                image_mesh, self.aspect_ratio_buffer = get_image_box(image_path=image_file,
                                                                     frustum_pose=new_camera,
                                                                     flip=True,
                                                                     cam_marker_size=self.frustum_scale)

                image_mesh = pyrender.Mesh.from_trimesh(image_mesh)
                self.frustum_images.append(image_mesh)

            frustum = generate_frustum_at_position(rotation=new_camera[:3, :3],
                                                   translation=new_camera[:3, 3],
                                                   color=frustum_color,
                                                   size=self.frustum_scale,
                                                   aspect_ratio=self.aspect_ratio_buffer)
            self.frustums.append(frustum)
            self.frustum_positions.append(new_camera[:3, 3])

    def clear_frustums(self):
        """
        Clear all existing frustums in the trajectory.
        """
        self.frustums.clear()
        self.frustum_images.clear()
        self.frustum_previous = None

    def get_mesh(self):
        """
        Turn trajectory into pyrender mesh.

        Frustum images are returned separately since merging textured and non-textured objects creates artifacts.

        @return: tuple, trajectory mesh + list of frustum image objects
        """
        # concatenate line segments and frustums into a single mapping trajectory mesh
        trajectory_mesh = self.trajectory + self.frustums
        trajectory_mesh = trimesh.util.concatenate(trajectory_mesh)
        trajectory_mesh = pyrender.Mesh.from_trimesh(trajectory_mesh)

        return trajectory_mesh, self.frustum_images
