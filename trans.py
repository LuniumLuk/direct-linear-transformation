## Direct Linear Transformation with Python
# implemented after https://me363.byu.edu/sites/me363.byu.edu/files/userfiles/5/DLTNotes.pdf
# functinalities:
# - estimate transformation from given UVs from two perspective 
# - calculate 3D point by giving UVs from two perspective and DLT result or Poses
# - also an nonlinear optimization for multiple views input

import numpy as np

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


def solve_DLT(p2d_pairs, intm, extm_pair):
    ## get constant
    L1, L2, L3, L4, L5, L6, L7, L8, L9, L10, L11 = get_DLT_constant(extm_pair[0], intm)
    R1, R2, R3, R4, R5, R6, R7, R8, R9, R10, R11 = get_DLT_constant(extm_pair[1], intm)

    p3ds = []
    Ls = np.array(p2d_pairs[0])
    Rs = np.array(p2d_pairs[1])

    ## calculate p3d for each p2d pairs
    for i in range(len(Ls)):
        u_L = Ls[i,0]
        v_L = Ls[i,1]
        u_R = Rs[i,0]
        v_R = Rs[i,1]

        # DLT Matrix
        Q = np.array([
            [L1-L9*u_L, L2-L10*u_L, L3-L11*u_L],
            [L5-L9*v_L, L6-L10*v_L, L7-L11*v_L],
            [R1-R9*u_R, R2-R10*u_R, R3-R11*u_R],
            [R5-R9*v_R, R6-R10*v_R, R7-R11*v_R],
        ])
        q = np.array([
            [u_L-L4],
            [v_L-L8],
            [u_R-R4],
            [v_R-R8],
        ])

        p3ds.append(np.linalg.inv(Q.T.dot(Q)).dot(Q.T).dot(q).ravel())

    return np.array(p3ds)