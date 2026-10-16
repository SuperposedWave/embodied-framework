"""Verified physical frame paths; no model binding, controller or IK."""
from . import types as leaf
from .validation import require
from ..geometry import compose, inverse


def transform_between(space,view,source,target):
    identity=leaf.Transform(translation_m=(0.,0.,0.),quaternion_xyzw=(0.,0.,0.,1.))
    def to_root(frame):
        require(frame in space.frames,"unknown_frame",frame,"undeclared frame")
        transform=identity
        cursor=frame
        seen=set()
        while space.frames[cursor].parent_id is not None:
            require(cursor not in seen,"frame_cycle",cursor,"cyclic frame graph")
            seen.add(cursor)
            spec=space.frames[cursor]
            if spec.dynamic:
                require(cursor in view.frame_transforms,"missing_transform",cursor,"dynamic public transform unavailable")
                sample=view.frame_transforms[cursor]
                require(0<=sample.captured_at_s<=view.sim_time_s+1e-9,"invalid_time",cursor,"invalid transform acquisition time")
                step=sample.transform
            else:
                step=spec.static_transform
            require(step is not None,"missing_transform",cursor,"static transform unavailable")
            transform=compose(step,transform)
            cursor=spec.parent_id
        return cursor,transform
    root_a,a=to_root(source)
    root_b,b=to_root(target)
    require(root_a==root_b,"disconnected_frames","conversion","different root frames")
    return compose(inverse(b),a)
