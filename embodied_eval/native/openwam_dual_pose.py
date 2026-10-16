"""Explicit inverse of the released dual EEF20 checkpoint's orientation encoding.

The pinned HDF5 writer stores physical world wxyz. The pinned training reader
and inference client feed those four numbers to an xyzw->rot6d conversion.
That is a model representation, NOT a physical rotation in another frame.

Input: public physical xyzw -> physical wxyz numbers -> native rot6d encoder.
Output: native rot6d decoder -> those wxyz numbers -> public physical xyzw.
No base/TCP transform, IK, normalization, controller change or gripper clipping.
The native decoder's float32 results are permuted only, never renormalized.
"""
import numpy as np

from ..core import types as leaf
from ..adapters.native_common import finite_array
from ..core.validation import require, quaternion
from . import contracts as c
from .native_catalog import GRIPPER_CONTROLLER, GRIPPER_PROTOCOL


PROFILE = "eef20_dual_wxyz_encoded_absolute"
TARGET_SLOTS = ("left_eef", "left_grip", "right_eef", "right_grip")


def encode_state(state, conversion):
    """state(key) returns a public inline state; native conversion is source-pinned."""
    poses, grippers = [], []
    for side in ("left", "right"):
        position = finite_array(state(side+"_position"), (3,), side+" position")
        q = finite_array(state(side+"_quaternion"), (4,), side+" quaternion")
        quaternion(q, side+" quaternion")
        grip = finite_array(state(side+"_grip"), (1,), side+" gripper")
        require(0. <= grip[0] <= 1., "state_out_of_bounds", side, "public gripper target must be in [0,1]")
        # Intentional model encoding. Do NOT replace this with physical xyzw:
        # that would change the input distribution of these exact weights.
        poses.append(np.concatenate((position, q[[3, 0, 1, 2]])))
        grippers.append(grip[0])
    return finite_array(conversion.robotwin_endpose_to_eef20d(
        poses[0], poses[1], grippers[0], grippers[1]), (20,), "native encoded EEF20", np.float32)


def decode_action(row, context, targets, conversion, *, acquire=True):
    require(set(targets) == set(TARGET_SLOTS) and len(set(targets.values())) == 4,
            "target_binding_mismatch", "dual pose", "both end effectors and both grippers must be bound exactly once")
    # The released policy transport and fixed decoder both produce float32.
    # Reject a lossy cast rather than silently narrowing a different protocol.
    original = finite_array(row, (20,), "native encoded EEF20")
    raw = finite_array(row, (20,), "native encoded EEF20", np.float32)
    require(np.array_equal(original, raw), "native_dtype_loss", "dual pose", "released EEF20 float32 representation required")
    for offset, side in ((0, "left"), (10, "right")):
        a, b = raw[offset+3:offset+6].astype(np.float64), raw[offset+6:offset+9].astype(np.float64)
        norm = np.linalg.norm(a)
        require(norm > 1e-8, "invalid_orientation", side, "first rot6d column must be nonzero")
        unit = a / norm
        require(np.linalg.norm(b - np.dot(unit, b) * unit) > 1e-8,
                "invalid_orientation", side, "rot6d columns must be linearly independent")
    native = finite_array(conversion.eef20d_to_ee16d(raw), (16,), "native decoded EE16")
    require(native.dtype == np.dtype("float32"), "native_dtype_mismatch", "dual pose", "pinned decoder returns float32")
    commands = []
    for offset, side in ((0, "left"), (8, "right")):
        # The decoder's nominal xyzw is in the training-encoded representation.
        # Its four values are physical wxyz; invert that encoding explicitly.
        physical_xyzw = native[offset+3:offset+7][[1, 2, 3, 0]]
        quaternion(physical_xyzw, side+" physical quaternion")
        commands.append(leaf.PoseTarget(command_id=context.message_id+":"+side+"_pose",
            target_id=targets[side+"_eef"], pose=leaf.Pose(
                position_m=tuple(float(v) for v in native[offset:offset+3]),
                quaternion_xyzw=tuple(float(v) for v in physical_xyzw), frame_id="world")))
        commands.append(leaf.ControllerCommand(command_id=context.message_id+":"+side+"_grip",
            target_id=targets[side+"_grip"], controller_profile_ref=GRIPPER_CONTROLLER,
            parameters={"requested_open_fraction": float(native[offset+7])}))
    return c.Action(commands=tuple(commands), acquire_observation_after=acquire,
                    native_protocol=GRIPPER_PROTOCOL, native_dtype="float32")
