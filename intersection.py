"""
First, we define 3 system types:

    * head
    * body
    * tail

Throughout the solution, we have 2 different combination of optimization variables:

    head / body
    -----------
    variable:   [Tin, Tout, Tc]     (1)
    constraint: Pin = Din           (2)
                Pout = Dout         (3)
                Tc ≥ Tout           (4)

    tail
    ----
    variable:   [Tin, Tout]         (5)
    constraint: Pin = Din           (6)
                Pout = Dout         (7)

However, different cars have different initial velocity.
"""
from pprint import pprint

import numpy as np
import casadi as ca

from problem import ALADIN_CFGS, SUB_SYS_COUNT, SAMPLE_N1, SAMPLE_N2, CONFIGS, SYMBOL_DEBUG
from helper import constructor
from helper.subsystype import SubSystemType, get_sub_system_type
from helper.colorize import color_print
from aladin.step_1_solve_nlp import step_1_solve_nlp
from aladin.step_2_term_cond import step_2_term_cond
from aladin.step_3_derivatives import step_3_derivatives
from aladin.step_4_solve_qp import step_4_solve_qp

def welcome():
    print('==========================')
    print('ALADIN Intersection Solver')
    print('==========================')


def main():
    """
    Optimization Structurization
    ----------------------------
    To save time, we don't construct variables and
    constraint on every loop. Instead, we construct
    them before the first iteration.
    """
    welcome()
    
    """
        NLP Structurization
    """
    nlp_struct = [constructor.build_nlp_struct(sub_index) for sub_index in range(SUB_SYS_COUNT)]

    """
        QP Structurization
        *: only required when using IPOPT to solve QP.
    """
    # qp_struct = constructor.build_qp_struct(SUB_SYS_COUNT)

    """
    τ, u, λ should have initial value before first iteration
    TODO replace fixed value `helper.centralized_reference`
    """
    # @param var_τ
    #   size (3, 1) or (2, 1)
    # Main optimization variable
    #   * head: Tin, Tout, Tc
    #   * body: Tin, Tout, Tc
    #   * tail: Tin, Tout
    var_τ = [
        np.array([7.25105129939717, 7.33483311739565, 7.33483310753611]),
        np.array([7.33483310753611, 7.97749052696764, 7.97749051709728]),
        np.array([8.3749051709728, 8.98871120516430, 8.99871119535654]),
        np.array([12.3371119535654, 10.69449434539719]),
    ]

    # @param var_u
    #   size (SAMPLE_N1 + SAMPLE_N2, 1)
    # Sub-system optimization variable
    var_u = [
        np.array([1.622531378, 1.532418169, 1.388678581, 1.127354991, 0.546333175, -0.427024026, -1.051964747, -1.350808261, -0.515754497, -0.548483267, -2.92e-09]),
        np.array([0.440943499, 0.371596761, 0.300548885, 0.228051776, 0.15443599, 0.080098682, 0.005480858, -0.068963037, -0.039347146, -0.083367171, -6.25e-10]),
        np.array([-0.861005866, -0.666381045, -0.425623341, -0.150389793, 0.138192487, 0.414192525, 0.656296234, 0.852753533, 0.157146887, 0.120843793, 4.74e-10]),
        np.array([-1.726596536, -1.643441148, -1.49094536, -1.130687198, 0.140486844, 1.167191186, 1.507653314, 1.652923525, 0.750888127, 0.747020972, 4.88e-09]),
    ]

    # @param var_λ
    #   size (SUB_SYS_COUNT - 1, 1)
    # Dual variable of coupling constraints
    if SYMBOL_DEBUG:
        # var_λ = np.array([17.8768591674695,19.3575077012303,13.0531045254504])
        var_λ = np.array([1,1,1])
    else:
        var_λ = -1*np.array([17.8768591674695,19.3575077012303,13.0531045254504])

    param_ρ = CONFIGS['aladin']['para']['ρ']

    """
    Begin of Loop
    """
    opt_sol, nlp_goal_func = [None]*SUB_SYS_COUNT, [None]*SUB_SYS_COUNT
    qp_gradient, qp_hessian = [None]*SUB_SYS_COUNT, [None]*SUB_SYS_COUNT

    for iter_count in range(ALADIN_CFGS['MAX_ITER']):

        """
        STEP 1 Solve decoupled NLP
        """
        for sub_index in range(SUB_SYS_COUNT):
            sub_sys_type = get_sub_system_type(SUB_SYS_COUNT, sub_index)
            opt_sol[sub_index], nlp_goal_func[sub_index] = step_1_solve_nlp(
                nlp_struct=nlp_struct[sub_index],
                sub_index=sub_index,
                var_u=var_u[sub_index],
                var_τ=var_τ[sub_index],
                var_λ=var_λ,
                param_ρ=param_ρ
            )
        color_print('ok', 1, 'iter {} nlp'.format(iter_count))

        """
        STEP 2 Form Ai for QP and check termination condition
        """
        should_terminate, qp_a, qp_b = step_2_term_cond(opt_sol)
        if should_terminate:
            color_print('ok', 0, 'Tolerance of {} is satisfied. Problem is optimized.'.format(ALADIN_CFGS['TOL']))
            # TODO plot()
            break

        """
        STEP 3 Find gradient and Hessian matrix
        """
        for sub_index in range(SUB_SYS_COUNT):
            qp_gradient[sub_index], qp_hessian[sub_index] = step_3_derivatives(nlp_struct[sub_index], nlp_goal_func[sub_index], opt_sol[sub_index])
        color_print('ok', 1, 'iter {} find gradient and hessian'.format(iter_count))

        """
        STEP 4 Solve coupled concensus QP
        """
        opt_Δτ, opt_qp_λ = step_4_solve_qp(qp_gradient, qp_hessian, qp_a, qp_b)
        color_print('ok', 1, 'iter {} con qp'.format(iter_count))

        """
        TODO STEP 5 Do line search 
        """

        """
        STEP 6 Update variables
        """
        for sub_index in range(SUB_SYS_COUNT-1):
            # Update τ
            color_print('debug', 2, 'updating value for car {}'.format(sub_index+1))
            color_print('debug', 3, '[{}] τ prev'.format(sub_index+1))
            pprint(var_τ[sub_index])
            color_print('debug', 3, '[{}] τ updated'.format(sub_index+1))
            pprint(opt_sol[sub_index]['τ'] + opt_Δτ[sub_index*3:(sub_index+1)*3,0])

            var_τ[sub_index] = opt_sol[sub_index]['τ'] + opt_Δτ[sub_index*3:(sub_index+1)*3,0]

            # Update u
            color_print('debug', 3, '[{}] u prev'.format(sub_index+1))
            pprint(var_u[sub_index])
            color_print('debug', 3, '[{}] u updated'.format(sub_index+1))
            pprint(opt_sol[sub_index]['u'])

            var_u[sub_index] = opt_sol[sub_index]['u']
        # Update for the last
        color_print('debug', 2, 'updating value for last car')
        color_print('debug', 3, '[last] τ prev')
        pprint(var_τ[-1])
        color_print('debug', 3, '[last] τ updated')
        pprint(opt_sol[-1]['τ'] + opt_Δτ[-2:,0])

        var_τ[-1] = opt_sol[-1]['τ'] + opt_Δτ[-2:,0]
        var_u[-1] = opt_sol[-1]['u']

        # Update λ
        color_print('debug', 2, 'updating λ')
        pprint(opt_qp_λ[-3:])

        var_λ = opt_qp_λ[-3:]

        color_print('ok', 0, '-----------------------')
        color_print('ok', 0, 'ITER {} COMPLETED'.format(iter_count))
        print('\n\n\n\n')
    
    # max iteration warning
    if iter_count+1 == ALADIN_CFGS['MAX_ITER']:
        color_print('warning', 0, 'max iteration reached, tolerance isn\'t met.')

if __name__ == '__main__':
    main()
