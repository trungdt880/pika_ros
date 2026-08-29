#!/usr/bin/env python3
"""Draw the READY / REST poses straight from the URDF, without a robot.

The point is to see a pose before the arm does it. arm_poses.nero_pika.yaml
holds seven numbers per pose and nothing about them tells you whether the
gripper ends up through the table -- this renders the actual collision meshes
at those joint angles, on a 10 cm ground grid, so you can look.

    # the two poses in the config, side and three-quarter views
    ros2 run pika_nero_teleop render_poses.py -o /root/pika_ros/arm_poses.png

    # try a pose without editing anything
    ros2 run pika_nero_teleop render_poses.py --pose 0 0.35 0 1.9 0 0.3 0

Needs no ROS, no display and no arm -- only Pillow and PyYAML, both already in
the container. It reads the URDF and the STL meshes off disk, so it works on
the host too:

    python3 pika_ros/src/pika_nero_teleop/scripts/render_poses.py \\
        --urdf pika_ros/src/agx_arm_ros/src/agx_arm_description/agx_arm_urdf/nero/urdf/nero_description.urdf \\
        --poses pika_ros/src/pika_nero_teleop/config/arm_poses.nero_pika.yaml \\
        -o docs/arm_poses.png

What it does NOT show: the Pika Gripper. agx_arm_urdf models the bare arm, so
every frame here ends at the joint7 flange. The gripper adds roughly another
0.2 m beyond it -- the flange heights printed under each view are the number to
judge REST by, minus that.

Rendering is a painter's-algorithm rasteriser: back-face cull, sort triangles
by depth, fill. ~200k triangles, well under a second. No z-buffer, so two
surfaces that interpenetrate can sort wrongly; for convex-ish robot links at
these sizes it does not show.
"""
import argparse
import math
import os
import struct
import sys
import xml.etree.ElementTree as ET

import yaml
from PIL import Image, ImageDraw

DEFAULT_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]

# Slightly different greys per link so the segments read apart; the flange is
# picked out in orange because it is the thing whose height you care about.
LINK_COLOUR = {
    "base_link": (118, 122, 130), "link1": (176, 180, 188), "link2": (150, 155, 165),
    "link3": (176, 180, 188), "link4": (150, 155, 165), "link5": (176, 180, 188),
    "link6": (150, 155, 165), "link7": (214, 128, 60),
}
LIGHT = (0.35, 0.55, 0.75)
_n = math.sqrt(sum(v * v for v in LIGHT))
LIGHT = tuple(v / _n for v in LIGHT)

BG = (247, 247, 245)
GRID = (223, 223, 218)
INK = (40, 42, 46)
MUTED = (120, 124, 132)


# --- small 3D helpers. Deliberately no numpy: this has to run anywhere. -----

def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def transform(rot, pos):
    return [[rot[0][0], rot[0][1], rot[0][2], pos[0]],
            [rot[1][0], rot[1][1], rot[1][2], pos[1]],
            [rot[2][0], rot[2][1], rot[2][2], pos[2]],
            [0.0, 0.0, 0.0, 1.0]]


def rpy_to_rot(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]]


def axis_angle_to_rot(axis, theta):
    norm = math.sqrt(sum(v * v for v in axis))
    x, y, z = [v / norm for v in axis]
    c, s, k = math.cos(theta), math.sin(theta), 1.0 - math.cos(theta)
    return [[c + x * x * k, x * y * k - z * s, x * z * k + y * s],
            [y * x * k + z * s, c + y * y * k, y * z * k - x * s],
            [z * x * k - y * s, z * y * k + x * s, c + z * z * k]]


def apply(mat, v):
    return (mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2] + mat[0][3],
            mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2] + mat[1][3],
            mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2] + mat[2][3])


# --- URDF ------------------------------------------------------------------

def load_urdf(path):
    """Joints in file order plus each link's collision mesh and its offset."""
    root = ET.parse(path).getroot()
    joints, links = [], {}

    def vec(elem, attr, default):
        raw = (elem.get(attr) if elem is not None else None) or default
        return [float(v) for v in raw.split()]

    for j in root.findall("joint"):
        origin, axis = j.find("origin"), j.find("axis")
        limit = j.find("limit")
        joints.append(dict(
            name=j.get("name"), type=j.get("type"),
            xyz=vec(origin, "xyz", "0 0 0"), rpy=vec(origin, "rpy", "0 0 0"),
            axis=[float(v) for v in axis.get("xyz").split()] if axis is not None else None,
            lower=float(limit.get("lower")) if limit is not None else None,
            upper=float(limit.get("upper")) if limit is not None else None,
            parent=j.find("parent").get("link"), child=j.find("child").get("link")))

    for link in root.findall("link"):
        col = link.find("collision")
        mesh = col.find("geometry/mesh") if col is not None else None
        origin = col.find("origin") if col is not None else None
        links[link.get("name")] = dict(
            mesh=mesh.get("filename") if mesh is not None else None,
            xyz=vec(origin, "xyz", "0 0 0"), rpy=vec(origin, "rpy", "0 0 0"))
    return joints, links


def forward_kinematics(joints, q):
    """World transform per link. Revolute joints consume q in file order."""
    frames = {joints[0]["parent"]: [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]}
    i = 0
    for j in joints:
        a = transform(rpy_to_rot(*j["rpy"]), j["xyz"])
        if j["type"] == "revolute":
            if i >= len(q):
                raise ValueError(f"URDF has more revolute joints than the {len(q)} given")
            a = mat_mul(a, transform(axis_angle_to_rot(j["axis"], q[i]), [0, 0, 0]))
            i += 1
        frames[j["child"]] = mat_mul(frames[j["parent"]], a)
    if i != len(q):
        raise ValueError(f"URDF has {i} revolute joints, {len(q)} values given")
    return frames


def check_limits(joints, q):
    """Joints outside the URDF's declared travel, as (name, value, lo, hi).

    Worth knowing because the real arm's stops are wider than this model's --
    a drag-taught pose can land outside it. A move_j target there still works
    (it goes to the firmware), but arm_ik_pose_node solves against this URDF
    and clamps, so teleop cannot hold such a pose and will jump out of it.
    """
    out = []
    for j, v in zip([j for j in joints if j["type"] == "revolute"], q):
        if j["lower"] is None:
            continue
        if not (j["lower"] <= v <= j["upper"]):
            out.append((j["name"], v, j["lower"], j["upper"]))
    return out


def resolve_mesh(uri, urdf_path):
    """package://<pkg>/<sub> -> <ancestor dir named pkg>/<sub>.

    Resolved by walking up from the URDF rather than through ament, so this
    runs on a host that has never sourced the workspace.
    """
    if not uri.startswith("package://"):
        return uri
    pkg, sub = uri[len("package://"):].split("/", 1)
    d = os.path.abspath(os.path.dirname(urdf_path))
    while d != os.path.dirname(d):
        if os.path.basename(d) == pkg:
            return os.path.join(d, sub)
        d = os.path.dirname(d)
    raise FileNotFoundError(f"cannot resolve {uri} from {urdf_path}")


def load_stl(path):
    """Binary STL -> list of triangles. These meshes are all binary."""
    with open(path, "rb") as fh:
        header = fh.read(84)
        if len(header) < 84:
            raise ValueError(f"{path}: too short for a binary STL")
        count = struct.unpack("<I", header[80:84])[0]
        body = fh.read(50 * count)
    if len(body) < 50 * count:
        raise ValueError(f"{path}: is this an ASCII STL? Only binary is supported")
    tris = []
    for k in range(count):
        f = struct.unpack_from("<12f", body, k * 50)
        tris.append(((f[3], f[4], f[5]), (f[6], f[7], f[8]), (f[9], f[10], f[11])))
    return tris


def world_triangles(joints, links, q, urdf_path, cache):
    frames = forward_kinematics(joints, q)
    out = []
    for name, spec in links.items():
        if not spec["mesh"] or name not in frames:
            continue
        path = resolve_mesh(spec["mesh"], urdf_path)
        if path not in cache:
            cache[path] = load_stl(path)
        mat = mat_mul(frames[name], transform(rpy_to_rot(*spec["rpy"]), spec["xyz"]))
        colour = LINK_COLOUR.get(name, (170, 174, 182))
        for tri in cache[path]:
            out.append((tuple(apply(mat, v) for v in tri), colour))
    return out, frames


# --- rendering -------------------------------------------------------------

class Camera:
    """Orthographic. azim 90 looks down -y (a side elevation); elev tilts up."""

    def __init__(self, azim, elev):
        a, e = math.radians(azim), math.radians(elev)
        self.right = (-math.sin(a), math.cos(a), 0.0)
        self.up = (-math.cos(a) * math.sin(e), -math.sin(a) * math.sin(e), math.cos(e))
        self.depth = (math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e))

    def project(self, p):
        return (sum(p[i] * self.right[i] for i in range(3)),
                sum(p[i] * self.up[i] for i in range(3)),
                sum(p[i] * self.depth[i] for i in range(3)))


def shade(rgb, normal):
    lit = max(0.0, sum(normal[i] * LIGHT[i] for i in range(3)))
    f = 0.34 + 0.66 * lit
    return tuple(min(255, int(c * f)) for c in rgb)


def frame_view(tris, cam, size, pad=0.10):
    """Fit the arm and the base of the grid into the panel.

    Auto-framing rather than fixed numbers because the two cameras see
    different extents, and --pose can be anything at all.
    """
    us, vs = [], []
    for tri, _ in tris:
        for v in tri:
            u, w, _ = cam.project(v)
            us.append(u)
            vs.append(w)
    for corner in ((0.0, 0.0, 0.0), (-0.45, 0.0, 0.0)):     # keep the base in shot
        u, w, _ = cam.project(corner)
        us.append(u)
        vs.append(w)
    span_u, span_v = max(us) - min(us), max(vs) - min(vs)
    centre = ((max(us) + min(us)) / 2.0, (max(vs) + min(vs)) / 2.0)
    scale = min(size[0] / (span_u * (1 + 2 * pad)), size[1] / (span_v * (1 + 2 * pad)))
    return scale, centre


def render_view(tris, cam, size, scale, centre, ss=2):
    """Return (supersampled image, to_px). Draw overlays, then downsample."""
    w, h = size[0] * ss, size[1] * ss
    img = Image.new("RGB", (w, h), BG)
    draw = ImageDraw.Draw(img)
    cx, cy = w / 2.0, h / 2.0
    s = scale * ss

    def to_px(p):
        u, v, d = cam.project(p)
        return ((u - centre[0]) * s + cx, cy - (v - centre[1]) * s, d)

    # 10 cm ground grid at z=0: the reference for "how far does it fall".
    for i in range(-8, 9):
        a, b = to_px((i * 0.1, -0.8, 0.0)), to_px((i * 0.1, 0.8, 0.0))
        draw.line([a[:2], b[:2]], fill=GRID, width=ss)
        a, b = to_px((-0.8, i * 0.1, 0.0)), to_px((0.8, i * 0.1, 0.0))
        draw.line([a[:2], b[:2]], fill=GRID, width=ss)

    faces = []
    for tri, colour in tris:
        e1 = tuple(tri[1][i] - tri[0][i] for i in range(3))
        e2 = tuple(tri[2][i] - tri[0][i] for i in range(3))
        n = (e1[1] * e2[2] - e1[2] * e2[1],
             e1[2] * e2[0] - e1[0] * e2[2],
             e1[0] * e2[1] - e1[1] * e2[0])
        length = math.sqrt(sum(v * v for v in n))
        if length < 1e-12:
            continue
        n = tuple(v / length for v in n)
        if sum(n[i] * cam.depth[i] for i in range(3)) <= 0.0:
            continue                                    # facing away
        pts = [to_px(v) for v in tri]
        faces.append((sum(p[2] for p in pts) / 3.0, [p[:2] for p in pts], shade(colour, n)))

    faces.sort(key=lambda f: f[0])
    for _, pts, colour in faces:
        draw.polygon(pts, fill=colour)
    return img, to_px


TOOL_AXES = {"x": (0, 1.0), "y": (1, 1.0), "z": (2, 1.0),
             "-x": (0, -1.0), "-y": (1, -1.0), "-z": (2, -1.0)}


def tool_direction(mat, tool_axis):
    """Unit vector the gripper sticks out along, in world coordinates.

    +x, verified on the hardware 2026-08-29. Sweeping joint7 at a fixed arm
    configuration swings where the gripper POINTS -- horizontal at joint7 = 0,
    straight down at +1.5708 -- which can only happen if the tool axis is
    perpendicular to joint7's rotation axis (link7 z). Model and photographs
    agree to a degree. Upstream's tool_translation_xyz: [0.1755, 0, -0.0235]
    says the same thing: 17.6 cm of flange->TCP along +x.

    Do not be misled by link7.stl being a 27 mm disc 9.6 mm thick along z, or
    by joint7's origin advancing 23.5 mm along that same z. Both suggest the
    tool should leave along z, and both are wrong about this arm. joint7 is a
    wrist pitch here, not a roll.

    --tool-axis overrides, for a different end effector.
    """
    col, sign = TOOL_AXES[tool_axis]
    return tuple(sign * mat[r][col] for r in range(3))


def draw_flange(draw, to_px, mat, gripper_length, tool_axis, ss):
    """Flange frame, plus a stick standing in for the gripper.

    The gripper is NOT in the URDF, and it is the part that actually hits the
    table -- so the tool axis is drawn out to gripper_length with a dot at the
    tip. It is a length, not a model: the real Pika Gripper is a body around
    that line, so treat the dot as the optimistic case.
    """
    origin = (mat[0][3], mat[1][3], mat[2][3])
    for col, colour in ((0, (214, 60, 60)), (1, (60, 165, 90)), (2, (60, 110, 220))):
        tip = tuple(origin[r] + 0.07 * mat[r][col] for r in range(3))
        draw.line([to_px(origin)[:2], to_px(tip)[:2]], fill=colour, width=2 * ss)
    if gripper_length > 0:
        d = tool_direction(mat, tool_axis)
        tip = tuple(origin[r] + gripper_length * d[r] for r in range(3))
        a, b = to_px(origin), to_px(tip)
        draw.line([a[:2], b[:2]], fill=(214, 128, 60), width=4 * ss)
        r = 5 * ss
        draw.ellipse([b[0] - r, b[1] - r, b[0] + r, b[1] + r],
                     fill=(214, 128, 60), outline=(150, 84, 30), width=ss)
    return origin


def panel(tris, frames, cam, size, scale, centre, title, subtitle,
          gripper_length, tool_axis, ss=2):
    if scale is None:
        scale, centre = frame_view(tris, cam, size)
    img, to_px = render_view(tris, cam, size, scale, centre, ss=ss)
    draw = ImageDraw.Draw(img)
    draw_flange(draw, to_px, frames["link7"], gripper_length, tool_axis, ss)
    img = img.resize(size, Image.LANCZOS)
    draw = ImageDraw.Draw(img)
    draw.text((14, 12), title, fill=INK)
    draw.text((14, 26), subtitle, fill=MUTED)
    draw.rectangle([0, 0, size[0] - 1, size[1] - 1], outline=(226, 226, 222))
    return img


def load_poses(path):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    params = cfg.get("/**", {}).get("ros__parameters", {})
    names = params.get("joint_names", DEFAULT_JOINTS)
    poses = [("READY", params["ready_pose"]), ("REST", params["rest_pose"])]
    return names, poses


def main(argv=None):
    here = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--urdf", default=None,
                        help="NERO URDF (default: found via the ament share dir)")
    parser.add_argument("--poses", default=os.path.join(
        here, "..", "config", "arm_poses.nero_pika.yaml"))
    parser.add_argument("--pose", nargs="+", type=float, default=None,
                        help="render these joint angles instead of the config")
    parser.add_argument("-o", "--out", default="arm_poses.png")
    parser.add_argument("--width", type=int, default=460, help="per panel")
    parser.add_argument("--height", type=int, default=520, help="per panel")
    parser.add_argument("--scale", type=float, default=None,
                        help="pixels per metre (default: fit the arm to the panel)")
    parser.add_argument("--gripper-length", type=float, default=0.20,
                        help="stand-in reach from the flange along the tool axis, "
                             "drawn as an orange stick (0 to hide it)")
    parser.add_argument("--tool-axis", default="x", choices=sorted(TOOL_AXES),
                        help="flange axis the gripper sticks out along "
                             "(default x, from upstream's tool_translation_xyz)")
    args = parser.parse_args(argv)

    urdf = args.urdf
    if urdf is None:
        try:
            from ament_index_python.packages import get_package_share_directory
            urdf = os.path.join(get_package_share_directory("agx_arm_description"),
                                "agx_arm_urdf", "nero", "urdf", "nero_description.urdf")
        except Exception:
            urdf = os.path.join(here, "..", "..", "agx_arm_ros", "src",
                                "agx_arm_description", "agx_arm_urdf", "nero",
                                "urdf", "nero_description.urdf")
    urdf = os.path.abspath(urdf)
    if not os.path.exists(urdf):
        print(f"URDF not found: {urdf}\nPass --urdf.", file=sys.stderr)
        return 1

    if args.pose:
        names, poses = DEFAULT_JOINTS, [("POSE", args.pose)]
    else:
        names, poses = load_poses(os.path.abspath(args.poses))

    joints, links = load_urdf(urdf)
    cache = {}
    size = (args.width, args.height)
    # Side elevation with a few degrees of tilt so the ground plane reads as a
    # plane, plus a three-quarter view for the parts the side view hides.
    views = [("side", Camera(90, 8)), ("3/4", Camera(126, 20))]

    rows = []
    for label, q in poses:
        tris, frames = world_triangles(joints, links, q, urdf, cache)
        flange = frames["link7"]
        pos = (flange[0][3], flange[1][3], flange[2][3])
        d = tool_direction(flange, args.tool_axis)
        tip = tuple(pos[r] + args.gripper_length * d[r] for r in range(3))
        angles = "  ".join(f"{n}={v:+.2f}" for n, v in zip(names, q))
        row = []
        for view_name, cam in views:
            row.append(panel(
                tris, frames, cam, size, args.scale, (0.0, 0.0),
                f"{label}   ({view_name})",
                f"flange x{pos[0]:+.3f} z{pos[2]:+.3f}    "
                f"tool tip z{tip[2]:+.3f} m (at {args.gripper_length:.2f} m)",
                args.gripper_length, args.tool_axis))
        rows.append((row, f"{label}   {angles}"))

    sheet = Image.new("RGB", (size[0] * len(views), (size[1] + 26) * len(rows)), BG)
    draw = ImageDraw.Draw(sheet)
    for r, (row, caption) in enumerate(rows):
        y = r * (size[1] + 26)
        for c, img in enumerate(row):
            sheet.paste(img, (c * size[0], y))
        draw.text((14, y + size[1] + 7), caption, fill=INK)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sheet.save(args.out)
    print(f"wrote {args.out}  ({sheet.size[0]}x{sheet.size[1]})")
    for label, q in poses:
        frames = forward_kinematics(joints, q)
        f = frames["link7"]
        pos = (f[0][3], f[1][3], f[2][3])
        d = tool_direction(f, args.tool_axis)
        tip = tuple(pos[r] + args.gripper_length * d[r] for r in range(3))
        print(f"  {label:6s} flange x={pos[0]:+.3f} y={pos[1]:+.3f} z={pos[2]:+.3f}"
              f"   tool tip z={tip[2]:+.3f}")
        for name, v, lo, hi in check_limits(joints, q):
            print(f"         ! {name}={v:+.4f} is outside the URDF limit "
                  f"[{lo:+.4f}, {hi:+.4f}]. move_j will still go there; "
                  f"arm_ik_pose_node will not -- do not use this as READY.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
