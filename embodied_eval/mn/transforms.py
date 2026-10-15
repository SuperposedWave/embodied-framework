"""Shared adapter utility for explicit physical frame conversion, not IK."""
from dataclasses import replace
from .. import contracts as leaf
from ..geometry import compose, inverse, multiply_quaternions, rotate_vector
from ..validation import require


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


def prepare_input(view,cap,profile,space):
    states=dict(view.states)
    specs={r.key:space.state_fields[profile.binding.inputs[r.key]] for r in cap.inputs if r.kind=="state"}
    for req in cap.inputs:
        if req.kind!="state" or req.frame_id is None:
            continue
        source=specs[req.key].frame_id
        if source==req.frame_id:
            continue
        require((source,req.frame_id) in profile.binding.frame_conversions,"frame_mismatch",req.key,"conversion not declared")
        transform=transform_between(space,view,source,req.frame_id)
        data=states[req.key].payload.data
        require(data is not None,"payload_unavailable",req.key,"DataRef reader unavailable")
        if req.semantic in ("eef_position","base_position"):
            moved=rotate_vector(transform.quaternion_xyzw,data)
            data=[a+b for a,b in zip(moved,transform.translation_m)]
        elif req.semantic=="eef_orientation":
            data=list(multiply_quaternions(transform.quaternion_xyzw,tuple(data)))
        else:
            require(False,"unsupported_frame_conversion",req.key,"only physical pose fields have a declared conversion")
        states[req.key]=replace(states[req.key],payload=replace(states[req.key].payload,data=data))
        specs[req.key]=replace(specs[req.key],frame_id=req.frame_id)
    return replace(view,states=states,state_specs=specs)


def convert_pose_action(action,profile,space,view):
    commands=[]
    hints=dict(action.source_euler_xyz_rad)
    for cmd in action.commands:
        if isinstance(cmd,leaf.PoseTarget):
            destinations=[target for source,target in profile.binding.frame_conversions if source==cmd.pose.frame_id]
            require(len(destinations)<=1,"ambiguous_frame_conversion",cmd.command_id,"one output destination per frame")
            if destinations:
                destination=destinations[0]
                transform=transform_between(space,view,cmd.pose.frame_id,destination)
                pose=compose(transform,leaf.Transform(translation_m=cmd.pose.position_m,quaternion_xyzw=cmd.pose.quaternion_xyzw))
                cmd=replace(cmd,pose=leaf.Pose(position_m=pose.translation_m,quaternion_xyzw=pose.quaternion_xyzw,frame_id=destination))
                hints.pop(cmd.target_id,None) # old-frame Euler hint cannot survive a conversion
        commands.append(cmd)
    return replace(action,commands=tuple(commands),source_euler_xyz_rad=hints)
