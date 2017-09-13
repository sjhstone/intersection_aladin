import numpy as np
from scipy.linalg import block_diag
import casadi as ca

from problem import QPOASES_SETTING
from helper.colorize import color_print

def step_4_solve_qp(qp_h_list, qp_g_list, qp_a, qp_b):
    qp_h = block_diag(qp_h_list[0], qp_h_list[1], qp_h_list[2], qp_h_list[3])
    qp_g = np.concatenate(qp_g_list, axis=0)

    qp = {
        'h': ca.DM(qp_h).sparsity(),
        'a': ca.DM(qp_a).sparsity(),
    }

    qp_solver = ca.conic('S', 'qpoases', qp, QPOASES_SETTING)

    # NOTE lba and uba as b
    opt_qp_soln = qp_solver(h=qp_h, g=qp_g, a=qp_a, lba=qp_b, uba=qp_b)

    opt_Δτ = opt_qp_soln['x'].full()
    opt_qp_λ = opt_qp_soln['lam_a'].full()

    return opt_Δτ, opt_qp_λ
