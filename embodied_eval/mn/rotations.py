"""Pure source-defined rotation conversions shared by adapter boundaries."""
import math
import numpy as np


def euler_quaternion_xyzw(euler):
    roll,pitch,yaw=np.asarray(euler,np.float64)
    cy,sy=np.cos(yaw*.5),np.sin(yaw*.5)
    cp,sp=np.cos(pitch*.5),np.sin(pitch*.5)
    cr,sr=np.cos(roll*.5),np.sin(roll*.5)
    return tuple(float(x) for x in (sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy))


def quat_axisangle(quaternion):
    q=np.asarray(quaternion).copy()
    q[3]=np.clip(q[3],-1.,1.)
    denominator=np.sqrt(1.-q[3]*q[3])
    return np.zeros(3) if math.isclose(denominator,0.) else q[:3]*2.*math.acos(q[3])/denominator
