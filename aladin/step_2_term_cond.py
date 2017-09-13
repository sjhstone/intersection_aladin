from pprint import pprint

import numpy as np
import casadi as ca

from problem import SUB_SYS_COUNT, SYMBOL_DEBUG, ALADIN_CFGS
from helper.colorize import color_print

def step_2_term_cond(opt_soln_list):
    """STEP 2 check termination condition

    First, form constraint A matrices by checking
    whether constraints of subproblems are active.

    """

    qp_mat_a_rows, qp_vec_b = [], []

    """
    TiC + ΔTiC = Ti+1In + ΔTi+1In
    Coordinate TiC of previous car and Tin of the following car.
    """
    for sub_index in range(SUB_SYS_COUNT-1):
        t_c__i = opt_soln_list[sub_index]['τ'][-1]
        t_in_n = opt_soln_list[sub_index+1]['τ'][0]
        qp_mat_a_row = [0]*3*sub_index + [0, 0, 1]
        if sub_index+1 == SUB_SYS_COUNT-1:
            # Reaching last car
            qp_mat_a_row += [-1, 0]
        else:
            qp_mat_a_row += [-1, 0, 0] + [0]*3*(SUB_SYS_COUNT-sub_index-3) + [0]*2
        qp_mat_a_rows.append(qp_mat_a_row)
        # NOTE symbol problem
        qp_vec_b += [t_in_n-t_c__i]

    """
    Tout + ΔTout = Tc + ΔTc
    Ensure copied variable is consistent with the original variable.
    NOTE Tc-Tout or Tout-Tc, λ
    """
    for sub_index in range(SUB_SYS_COUNT-1):
        t_out_val = opt_soln_list[sub_index]['τ'][1]
        t_c___val = opt_soln_list[sub_index]['τ'][2]
        if SYMBOL_DEBUG:
            c_diff = t_c___val - t_out_val
        else:
            c_diff = t_out_val - t_c___val
        if c_diff <= ALADIN_CFGS['COPIED_GAP']:
            # Tc appears to be smaller than Tout,
            # constraint is active
            qp_mat_a_row = [0]*3*sub_index
            qp_mat_a_row += [0, 1, -1]
            #              NOTE ~  ^^
            qp_mat_a_row += [0]*3*(SUB_SYS_COUNT-sub_index-2)
            qp_mat_a_row += [0]*2
            qp_mat_a_rows.append(qp_mat_a_row)
            qp_vec_b += [0]

    qp_mat_a = np.array([np.array(row) for row in qp_mat_a_rows])
    """
    $$ | \sum_{i=1}^N A_i y_i - b | \le \epsilon $$
    """
    ai_times_yi_minus_b = qp_mat_a@np.hstack((opt_soln_list[0]['τ'], opt_soln_list[1]['τ'], opt_soln_list[2]['τ'], opt_soln_list[3]['τ']))
    ai_times_yi_minus_b -= np.array(qp_vec_b)
    sum_ai_times_yi_minus_b = np.abs(np.sum(ai_times_yi_minus_b))

    """
    ρ|τi - zi| \le epsion
    """
    max_opt_movement = 0
    for sub_index in range(SUB_SYS_COUNT):
        max_opt_movement = max(max_opt_movement, max(opt_soln_list[sub_index]['opt_movement']))
    
    color_print('info', 2, 'residual')
    pprint(max(sum_ai_times_yi_minus_b, max_opt_movement))


    return max(sum_ai_times_yi_minus_b, max_opt_movement) <= ALADIN_CFGS['TOL'], qp_mat_a, qp_vec_b
