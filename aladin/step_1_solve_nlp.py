from pprint import pprint

import numpy as np
import casadi as ca

from problem import SYMBOL_DEBUG, IPOPT_SETTING
from helper.subsystype import SubSystemType, get_sub_system_type
from helper.colorize import color_print
from helper.subsystype import SubSystemType

def step_1_solve_nlp(nlp_struct, sub_index, var_u, var_τ, var_λ, param_ρ):

    goal_func = nlp_struct['cost']
    goal_func += param_ρ/2 * ca.transpose(nlp_struct['xt']-var_τ)@(nlp_struct['xt']-var_τ)
    if SYMBOL_DEBUG:
        goal_func += var_λ[sub_index]*nlp_struct['tc'] \
            if nlp_struct['type'] in (SubSystemType.head, SubSystemType.body) else 0
        goal_func -= var_λ[sub_index-1]*nlp_struct['tin'] \
            if nlp_struct['type'] in (SubSystemType.body, SubSystemType.tail) else 0
    else:
        goal_func -= var_λ[sub_index]*nlp_struct['tc'] \
            if nlp_struct['type'] in (SubSystemType.head, SubSystemType.body) else 0
        goal_func += var_λ[sub_index-1]*nlp_struct['tin'] \
            if nlp_struct['type'] in (SubSystemType.body, SubSystemType.tail) else 0
    
    nlp = {
        'x': nlp_struct['x'],
        'f': goal_func,
        'g': nlp_struct['g']
    }

    color_print('info', 1,
        'Running IPOPT to optimize subsystem {}'.format(sub_index+1)
    )

    nlp_solver = ca.nlpsol('S', 'ipopt', nlp, IPOPT_SETTING)

    color_print('debug', 3, 'Passing x to NLP solver')
    pprint(nlp_struct['x'], indent=2)

    color_print('debug', 3, 'Passing initializing point to NLP solver')
    pprint(ca.vertcat(var_τ, var_u))

    opt_sol = nlp_solver(
        x0=ca.vertcat(var_τ, var_u),
        lbx=nlp_struct['lbx'],
        ubx=nlp_struct['ubx'],
        lbg=nlp_struct['lbg'],
        ubg=nlp_struct['ubg'],
    )

    τ_opt = opt_sol['x'].full()[0:3, 0] \
        if nlp_struct['type'] in (SubSystemType.head, SubSystemType.body) \
        else opt_sol['x'].full()[0:2, 0]
    u_opt = opt_sol['x'].full()[3:, 0] \
        if nlp_struct['type'] in (SubSystemType.head, SubSystemType.body) \
        else opt_sol['x'].full()[2:, 0]
    λ_opt = opt_sol['lam_g'].full()

    # construct active constraint set
    # Din, Dout are assumed to be always active
    constraint_h = nlp_struct['gu']
    # gu are added as the last 2 constraints
    constraint_dual = opt_sol['lam_g'].full()[-2:]

    # check active constraint
    control_dual = opt_sol['lam_x'].full()[3:] \
        if nlp_struct['type'] in (SubSystemType.head, SubSystemType.body) \
        else opt_sol['lam_x'].full()[2:]

    for control_index, control in enumerate(control_dual):
        if np.abs(control) > 1e-8:
            # Assume to be active
            constraint_h = ca.vertcat(constraint_h, nlp_struct['xu'][control_index])
    
    # find all duals corresponding to active constraints
    control_dual = control_dual[np.abs(control_dual) > 1e-8]
    constraint_dual = np.append(constraint_dual, control_dual)

    color_print('debug', 3, 'ipopt result for car {}'.format(sub_index+1))
    color_print('debug', 3, '[{}] Tin, Tout, Tc'.format(sub_index+1))
    pprint(τ_opt)
    color_print('debug', 3, '[{}] Control'.format(sub_index+1))
    pprint(u_opt)
    color_print('debug', 3, '[{}] λ_opt'.format(sub_index+1))
    pprint(λ_opt)

    return {'τ': τ_opt, 'opt_movement': param_ρ*np.abs(τ_opt-var_τ),
            'u': u_opt, 'λ': λ_opt,
            'active_constraint': constraint_h, 'active_dual_val': constraint_dual
            }, goal_func
