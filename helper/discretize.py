import numpy as np
import casadi as ca

def get_sampling_interval_length(t_in: ca.MX, t_out: ca.MX, n_1: int, n_2: int, has_entered: bool):
    """return `float` `length of time interval`.
    """
    return (t_out - t_in) / n_2 if has_entered else t_in / n_1

def get_discretized_dynamics(len_t_s: ca.MX):
    """return matrices A and B
    """
    mat_dis_A = ca.MX(
        len_t_s*ca.DM(
            [
                [0, 1],
                [0, 0]
            ]
        ) + np.eye(2)
    )
    mat_dis_B = ca.vertcat(len_t_s**2/2, len_t_s)
    return mat_dis_A, mat_dis_B
