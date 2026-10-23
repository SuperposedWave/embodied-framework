"""Explicit physical transforms; never apply controller scaling here."""
import math
from .core.types import Pose, Transform
from .errors import ContractError


def _error(message):
    raise ContractError("invalid_geometry", "geometry", message)


def _unit(q):
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        _error("expected four finite quaternion_xyzw components")
    if not math.isclose(sum(v*v for v in q), 1.0, abs_tol=1e-6):
        _error("expected unit quaternion; normalize explicitly")
    return q


def _cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def _dot(a, b):
    return sum(x*y for x, y in zip(a, b))


def _normalize(v):
    n = math.sqrt(_dot(v, v))
    if n < 1e-10:
        _error("degenerate rotation representation")
    return tuple(x/n for x in v)


def normalize_quaternion(q):
    if len(q) != 4 or not all(math.isfinite(x) for x in q):
        _error("invalid quaternion")
    return _normalize(q)


def multiply_quaternions(a, b):
    ax, ay, az, aw = _unit(a)
    bx, by, bz, bw = _unit(b)
    return (
        aw*bx+ax*bw+ay*bz-az*by,
        aw*by-ax*bz+ay*bw+az*bx,
        aw*bz+ax*by-ay*bx+az*bw,
        aw*bw-ax*bx-ay*by-az*bz,
    )


def rotate_vector(q, v):
    if len(v) != 3 or not all(math.isfinite(component) for component in v):
        _error("expected three finite vector components")
    x, y, z, w = _unit(q)
    t = tuple(2*x for x in _cross((x, y, z), v))
    u = _cross((x, y, z), t)
    return tuple(v[i]+w*t[i]+u[i] for i in range(3))


def compose(a: Transform, b: Transform) -> Transform:
    """T_A_B @ T_B_C = T_A_C."""
    moved = rotate_vector(a.quaternion_xyzw, b.translation_m)
    return Transform(translation_m=tuple(x+y for x, y in zip(a.translation_m, moved)),
                     quaternion_xyzw=multiply_quaternions(a.quaternion_xyzw, b.quaternion_xyzw))


def inverse(t: Transform) -> Transform:
    x, y, z, w = _unit(t.quaternion_xyzw)
    q = (-x, -y, -z, w)
    return Transform(translation_m=rotate_vector(q, tuple(-x for x in t.translation_m)),
                     quaternion_xyzw=q)


def transform_pose(pose: Pose, transform: Transform, *, source_frame: str, target_frame: str) -> Pose:
    if pose.frame_id != source_frame:
        _error("pose source frame does not match the declared conversion")
    result = compose(transform, Transform(translation_m=pose.position_m, quaternion_xyzw=pose.quaternion_xyzw))
    return Pose(position_m=result.translation_m, quaternion_xyzw=result.quaternion_xyzw, frame_id=target_frame)


def apply_pose_delta(reference: Pose, delta: Transform, *, composition: str) -> Pose:
    ref = Transform(translation_m=reference.position_m, quaternion_xyzw=reference.quaternion_xyzw)
    if composition == "pre_multiply":
        result = compose(delta, ref)
    elif composition == "post_multiply":
        result = compose(ref, delta)
    else:
        _error("composition must be pre_multiply or post_multiply")
    return Pose(position_m=result.translation_m, quaternion_xyzw=result.quaternion_xyzw, frame_id=reference.frame_id)


def quaternion_to_rot6d(q):
    x, y, z, w = _unit(q)
    return (
        1-2*(y*y+z*z), 2*(x*y+w*z), 2*(x*z-w*y),
        2*(x*y-w*z), 1-2*(x*x+z*z), 2*(y*z+w*x),
    )


def rot6d_to_quaternion(values):
    if len(values) != 6 or not all(math.isfinite(v) for v in values):
        _error("rot6d requires six finite values (first two matrix columns)")
    col0 = _normalize(values[:3])
    dot = _dot(col0, values[3:])
    col1 = _normalize(tuple(values[i+3]-dot*col0[i] for i in range(3)))
    col2 = _cross(col0, col1)
    m = [[col0[i], col1[i], col2[i]] for i in range(3)]
    trace = m[0][0]+m[1][1]+m[2][2]
    if trace > 0:
        s = 2*math.sqrt(trace+1)
        q = ((m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s, s/4)
    else:
        i = max(range(3), key=lambda k: m[k][k])
        j, k = (i+1) % 3, (i+2) % 3
        s = 2*math.sqrt(max(0, 1+m[i][i]-m[j][j]-m[k][k]))
        if s < 1e-10:
            _error("degenerate matrix")
        xyz = [0.0, 0.0, 0.0]
        xyz[i], xyz[j], xyz[k] = s/4, (m[j][i]+m[i][j])/s, (m[k][i]+m[i][k])/s
        q = (*xyz, (m[k][j]-m[j][k])/s)
    return normalize_quaternion(q)


_UNIT_FACTORS = {
    "m": ("length", 1), "cm": ("length", .01), "mm": ("length", .001),
    "rad": ("angle", 1), "deg": ("angle", math.pi/180),
    "m/s": ("linear_velocity", 1), "cm/s": ("linear_velocity", .01),
    "rad/s": ("angular_velocity", 1), "deg/s": ("angular_velocity", math.pi/180),
    "N": ("force", 1), "Nm": ("torque", 1),
}


def convert_quantity(value: float, source_unit: str, target_unit: str) -> float:
    if source_unit not in _UNIT_FACTORS or target_unit not in _UNIT_FACTORS:
        _error("unsupported physical unit; controller_native is never a physical unit")
    source, target = _UNIT_FACTORS[source_unit], _UNIT_FACTORS[target_unit]
    if source[0] != target[0] or not math.isfinite(value):
        _error("incompatible dimension or non-finite value")
    return value*source[1]/target[1]


def frame_transform(space, observation, *, source_frame: str, target_frame: str, max_age_s: float | None = None):
    """Resolve T_target_source from the declared frame graph and current samples."""
    from .core.validation import validate_structure, quaternion
    validate_structure(space)
    for frame, sample in observation.frame_transforms.items():
        if frame not in space.frames or not space.frames[frame].dynamic or not 0 <= sample.captured_at_s <= observation.sim_time_s + 1e-9:
            _error("invalid dynamic frame sample")
        quaternion(sample.transform.quaternion_xyzw, frame)
    if source_frame not in space.frames or target_frame not in space.frames:
        _error("source and target frames must be declared")
    if max_age_s is not None and (not math.isfinite(max_age_s) or max_age_s < 0):
        _error("maximum transform age must be finite and nonnegative")
    def to_root(frame):
        chain = []
        while space.frames[frame].parent_id is not None:
            spec = space.frames[frame]
            if spec.dynamic:
                if frame not in observation.frame_transforms:
                    _error("required dynamic frame transform is missing")
                sample = observation.frame_transforms[frame]
                if max_age_s is not None and observation.sim_time_s - sample.captured_at_s > max_age_s + 1e-9:
                    _error("dynamic transform exceeds maximum age")
                transform = sample.transform
            else:
                transform = spec.static_transform
            chain.append(transform)
            frame = spec.parent_id
        result = Transform(translation_m=(0., 0., 0.), quaternion_xyzw=(0., 0., 0., 1.))
        for transform in reversed(chain):
            result = compose(result, transform)
        return frame, result
    source_root, to_source = to_root(source_frame)
    target_root, to_target = to_root(target_frame)
    if source_root != target_root:
        _error("source and target frames are disconnected")
    return compose(inverse(to_target), to_source)
