## Direct Linear Transformation with Python
# implemented after https://me363.byu.edu/sites/me363.byu.edu/files/userfiles/5/DLTNotes.pdf
# functinalities:
# - estimate transformation from given UVs from two perspective 
# - calculate 3D point by giving UVs from two perspective and DLT result or Poses
# - also an nonlinear optimization for multiple views input

import numpy as np

def DLT_normalization(x):
    n = x.shape[1]
    m, s = np.mean(x, 0), np.std(x)
    if n == 2:
        T = np.array([
            [ s,  0,  m[0] ],
            [ 0,  s,  m[1] ],
            [ 0,  0,  1    ],
        ])
    elif n == 3:
        T = np.array([
            [ s,  0,  0,  m[0] ],
            [ 0,  s,  0,  m[1] ],
            [ 0,  0,  s,  m[2] ],
            [ 0,  0,  0,  1    ],
        ])
    else:
        raise ValueError("x.shape[1] must be 2 or 3")

    T_inv = np.linalg.inv(T)
    x = np.dot(T_inv, np.concatenate((x.T, np.ones((1, x.shape[0])))))
    x = x[0:n, :].T

    return T_inv, x

def DLT_calibration(uvs, p3ds):
    n_points = len(uvs)

    uvsT, uvsN = DLT_normalization(uvs)
    p3dsT, p3dsN = DLT_normalization(p3ds)

    g = uvsN.reshape(n_points * 2)
    F = []
    for i in range(n_points):
        x, y, z = p3dsN[i]
        u, v = uvsN[i]
        F.append(np.array([
            [x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z],
            [0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z],
        ]))
    F = np.vstack(F)

    L = np.linalg.pinv(F).dot(g) # Linear Transform Components: L1~L11
    H = np.hstack((L, [1.0])).reshape((3,4))
    H = np.linalg.pinv(uvsT).dot(H).dot(p3dsT)
    H = H * np.divide(1.0, H[-1,-1])

    uvs2 = DLT_project_2d(H.flatten()[:11], p3ds)

    MSE = np.mean(np.sum(np.power(uvs - uvs2, 2), axis=1))
    err = np.sqrt(np.mean(np.sum((uvs2 - uvs) ** 2, 1)))

    return H.flatten()[:11], MSE, err

# Project 3D points to UV space using DLT components
def DLT_project_2d(L, p3ds):
    L = L.flatten()
    # H = | L1  L2  L3  L4  |
    #     | L5  L6  L7  L8  |
    #     | L9  L10 L11 1.0 |
    H = np.hstack((L, [1.0])).reshape((3,4))
    # P = | x1  y1  z1  1.0 |
    #     | ... ... ... ... |
    #     | xn  yn  zn  1.0 |
    P = np.hstack((p3ds, np.ones((len(p3ds),1))))
    V = P.dot(H.T)
    return V[:,:2] * np.divide(1.0, V[:,-1,np.newaxis])

def get_DLT_constant(trans, intm):  
    trans = np.array(trans)
    intm = np.array(intm)

    ## decompose variables
    fx, _, ox, _, fy, oy, _, _, _ = intm.ravel()
    R1, R2, R3, T1, R4, R5, R6, T2, R7, R8, R9, T3, _, _, _, _ = trans.ravel()

    ## cannot handle object in camera plane
    assert(T3 != 0)

    ## return DLT constant
    return [
        (fx * R1 + ox * R7) / T3,   # L1 / R1
        (fx * R2 + ox * R8) / T3,   # L2 / R2
        (fx * R3 + ox * R9) / T3,   # L3 / R3
        (fx * T1 + ox * T3) / T3,   # L4 / R4
        (fy * R4 + oy * R7) / T3,   # L5 / R5
        (fy * R5 + oy * R8) / T3,   # L6 / R6
        (fy * R6 + oy * R9) / T3,   # L7 / R7
        (fy * T2 + oy * T3) / T3,   # L8 / R8
        R7 / T3,                    # L9 / R9
        R8 / T3,                    # L10 / R10
        R9 / T3,                    # L11 / R11
    ]


def solve_DLT(uv_pairs, intm, extm_pair):
    ## get constant
    L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11 = get_DLT_constant(extm_pair[0], intm)
    R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11 = get_DLT_constant(extm_pair[1], intm)

    p3ds = []
    Ls = np.array(uv_pairs[0])
    Rs = np.array(uv_pairs[1])

    ## calculate p3d for each p2d pairs
    for i in range(len(Ls)):
        u = Ls[i,0]
        v = Ls[i,1]
        u_R = Rs[i,0]
        v_R = Rs[i,1]

        # DLT Matrix
        Q = np.array([
            [L1-L9*u, L2-L10*u, L3-L11*u],
            [L5-L9*v, L6-L10*v, L7-L11*v],
            [R1-R9*u_R, R2-R10*u_R, R3-R11*u_R],
            [R5-R9*v_R, R6-R10*v_R, R7-R11*v_R],
        ])
        q = np.array([
            [u-L4],
            [v-L8],
            [u_R-R4],
            [v_R-R8],
        ])

        p3ds.append(np.linalg.inv(Q.T.dot(Q)).dot(Q.T).dot(q).ravel())

    return np.array(p3ds)


p3ds = np.array([[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755], [2951, 0.5, 9.755], [-4132, 0.5, 23.618],
           [-876, 0, 23.618]])
uvs = np.array([[76, 706], [702, 706], [1440, 706], [1867, 706], [264, 523], [625, 523]])

L1, MSE, err = DLT_calibration(uvs, p3ds)

from dlt import DLT
# Known 3D coordinates
xyz = [[-875, 0, 9.755], [442, 0, 9.755], [1921, 0, 9.755], [2951, 0.5, 9.755], [-4132, 0.5, 23.618],
           [-876, 0, 23.618]]
# Known pixel coordinates
uv = [[76, 706], [702, 706], [1440, 706], [1867, 706], [264, 523], [625, 523]]
L2, err, uv2 = DLT(xyz, uv)

L1 = np.asarray(L1)
L2 = np.asarray(L2)[:11]
print(L1 - L2)