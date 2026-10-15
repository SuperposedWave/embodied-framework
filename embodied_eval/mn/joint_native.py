"""A named scalar request, not an opaque whole-robot action array.

Clamping/interpolation belong to the declared simulator protocol. Both consumers
understand the same requested OPENING fraction; their native dynamics may differ.
"""
RAW_GRIPPER_PROTOCOL = "gripper-open-fraction-raw-request@1"
RAW_GRIPPER_SIGNATURE = "finite-unbounded-open-fraction-request;dtype-preserved;simulator-protocol-owns-clamping-and-interpolation"
RAW_GRIPPER_CONTROLLER = "gripper-open-fraction-raw-request@1"
