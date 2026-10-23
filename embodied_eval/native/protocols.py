"""Named native command meanings, shared without importing either adapter."""
OSC_PROTOCOL = 'osc-pose-normalized7@1'
OSC_SIGNATURE = 'xyz3+axisangle3+grip_close1;normalized[-1,1];environment-scales;world-delta;20Hz'
DUAL_PROTOCOL = 'dual-eef16-legacy-quaternion@1'
DUAL_SIGNATURE = 'left+right:xyz-m3+helper-xyzw4-interpreted-as-native-wxyz+open-fraction1;dtype-preserved'
GRIPPER_PROTOCOL = 'gripper-open-fraction-interpolate-then-clamp@1'
GRIPPER_SIGNATURE = 'finite-unbounded-open-fraction-request;interpolate-from-current-per-native-physics-path;clamp-each-step[0,1];dtype-preserved'
GRIPPER_CONTROLLER = 'gripper-open-fraction-raw-request@1'
