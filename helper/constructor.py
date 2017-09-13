import numpy as np
import casadi as ca

from helper.colorize import color_print
from helper.subsystype import SubSystemType, get_sub_system_type
from helper.discretize import get_sampling_interval_length, get_discretized_dynamics

from problem import CONFIGS, SUB_SYS_COUNT, SAMPLE_N1, SAMPLE_N2, V_INITS, SYMBOL_DEBUG

def build_nlp_time_related(v_index, sub_sys_type):

    # form Tin and Tout
    t_in = ca.MX.sym('Tin_{}'.format(v_index+1))
    t_out = ca.MX.sym('Tout_{}'.format(v_index+1))

    # construct full τ and corresponding constraints
    # notice that time should be positive
    x_τ_xτ, x_τ_lb, x_τ_ub = ca.vertcat(t_in, t_out), [0]*2, [np.inf]*2
    g_τ_gτ, g_τ_lb, g_τ_ub = ca.MX.zeros(0, 0), [], []

    if sub_sys_type in (SubSystemType.head, SubSystemType.body):
        # add copy variable to sub-system
        t_c = ca.MX.sym('Tc_{}'.format(v_index+1))
        x_τ_xτ = ca.vertcat(x_τ_xτ, t_c)
        x_τ_lb += [0]
        x_τ_ub += [np.inf]

        # enforce constraint to make it a "copied" variable
        # NOTE `t_out-t_c` or `t_c-t_out` will lead to different symbol of λ
        if SYMBOL_DEBUG:
            g_τ_gτ = ca.vertcat(g_τ_gτ, t_out-t_c)
            g_τ_lb += [-np.inf]
            g_τ_ub += [-np.spacing(0)]
        else:
            g_τ_gτ = ca.vertcat(g_τ_gτ, t_c-t_out)
            g_τ_lb += [np.spacing(0)]
            g_τ_ub += [np.inf]
    else:
        t_c = None

    return x_τ_xτ, x_τ_lb, x_τ_ub, g_τ_gτ, g_τ_lb, g_τ_ub, t_in, t_out, t_c


def build_nlp_control_related(v_index, sub_sys_type, t_in, t_out):
    sub_known = V_INITS[v_index]

    # Read known values
    init_position = sub_known['P0']
    init_velocity = sub_known['V0']
    ref_velocity = sub_known['Vref']
    control_lb, control_ub = sub_known['Umin'], sub_known['Umax']
    d_in, d_out = sub_known['Din'], sub_known['Dout']

    # construct initial state
    state_i = ca.vertcat(init_position, init_velocity)

    cost_fn = (state_i[1]-ref_velocity)**2

    len_t_s = get_sampling_interval_length(t_in, t_out, SAMPLE_N1, SAMPLE_N2, False)
    dmat_a, dmat_b = get_discretized_dynamics(len_t_s)

    x_u_xu, x_u_lb, x_u_ub = ca.MX.zeros(0, 0), [], []
    g_u_gu, g_u_lb, g_u_ub = ca.MX.zeros(0, 0), [], []

    for u_index in range(SAMPLE_N1+SAMPLE_N2):
        control_i = ca.MX.sym('u{}_{}'.format(u_index, v_index+1))
        x_u_xu = ca.vertcat(x_u_xu, control_i)
        x_u_lb += [control_lb]
        x_u_ub += [control_ub]

        state_i = dmat_a@state_i + dmat_b@control_i

        cost_fn += (state_i[1]-ref_velocity)**2 + control_i**2

        # critical points, N1 and N2
        if u_index+1 == SAMPLE_N1:
            position_entry = state_i[0]
            g_u_gu = ca.vertcat(g_u_gu, position_entry-d_in)
            g_u_lb += [0]
            g_u_ub += [0]
            # obtain sampling time interval after entry
            len_t_s = get_sampling_interval_length(t_in, t_out, SAMPLE_N1, SAMPLE_N2, True)
            dmat_a, dmat_b = get_discretized_dynamics(len_t_s)
        elif u_index+1 == SAMPLE_N1+SAMPLE_N2:
            position_exit = state_i[0]
            g_u_gu = ca.vertcat(g_u_gu, position_exit-d_out)
            g_u_lb += [0]
            g_u_ub += [0]

    cost_fn /= t_out

    return x_u_xu, x_u_lb, x_u_ub, g_u_gu, g_u_lb, g_u_ub, cost_fn


def build_nlp_struct(v_index):
    """build_nlp_struct
    Given initial velocity and position,
    build variables and constraint needed in solving NLP
    """
    color_print(
        'info', 3,
        'Building NLP struct for vehicle {}.'.format(v_index+1)
    )

    sub_sys_type = get_sub_system_type(SUB_SYS_COUNT, v_index)

    # time related first
    x_τ_xτ, x_τ_lb, x_τ_ub, g_τ_gτ, g_τ_lb, g_τ_ub, t_in, t_out, t_c = build_nlp_time_related(v_index, sub_sys_type)
    # control related second
    x_u_xu, x_u_lb, x_u_ub, g_u_gu, g_u_lb, g_u_ub, cost_fn = build_nlp_control_related(v_index, sub_sys_type, t_in, t_out)

    x_xx = ca.vertcat(x_τ_xτ, x_u_xu)
    x_lb = x_τ_lb + x_u_lb
    x_ub = x_τ_ub + x_u_ub
    g_gx = ca.vertcat(g_τ_gτ, g_u_gu)
    g_lb = g_τ_lb + g_u_lb
    g_ub = g_τ_ub + g_u_ub

    color_print(
        'ok', 2,
        'NLP struct for vehicle {} is built.'.format(v_index+1)
    )

    return {'x': x_xx, 'lbx': x_lb, 'ubx': x_ub,
            'xt': x_τ_xτ,
            'tin': t_in, 'tout': t_out, 'tc': t_c,
            'xu': x_u_xu,
            'g': g_gx, 'lbg': g_lb, 'ubg': g_ub,
            'gt': g_τ_gτ, 'gu': g_u_gu,
            'cost': cost_fn, 'type': sub_sys_type}
