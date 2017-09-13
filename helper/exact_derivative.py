def exact_hessian(f, h, x, y, x_val, y_val,kappa_val):

    # this function compute the gradient and hessian of g(x) defined as
    # g(x) = min_y f(x,y) s.t. h(x,y) = 0 | kappa
    # here, y^*(x) and kappa^*(x) can be consdiered as functions of x

    import numpy as np
    import casadi as ca

    # % define Lagrangian
    L = f + ca.mtimes(ca.transpose(kappa_val), h)
    
    # % compute partial first order derivative
    fx = ca.gradient(f, x)
    eval_fx = ca.Function('fx', [x, y], [fx])
    val_fx = eval_fx(x_val, y_val).full()

    fy = ca.gradient(f, y)
    eval_fy = ca.Function('fx', [x, y], [fy])
    val_fy = eval_fy(x_val,y_val).full()

    # % compute partial second order derivative
    Lxx = ca.hessian(L, x)[0]
    eval_Lxx = ca.Function('Lxx', [x, y], [Lxx])
    val_Lxx = eval_Lxx(x_val,y_val)

    Lyy = ca.hessian(L, y)[0]
    eval_Lyy = ca.Function('Lyy', [x, y], [Lyy])
    val_Lyy = eval_Lyy(x_val,y_val)

    Lx = ca.gradient(L, x)
    Ly = ca.gradient(L, y)

    Lyx = ca.jacobian(Ly, x)
    eval_Lyx = ca.Function('Lyx', [x, y], [Lyx])
    val_Lyx = eval_Lyx(x_val,y_val).full()

    Lxy = ca.jacobian(Lx, y)
    eval_Lxy = ca.Function('Lxy', [x, y], [Lxy])
    val_Lxy = eval_Lxy(x_val,y_val).full()

    hx = ca.jacobian(h, x)
    eval_hx = ca.Function('hx', [x, y], [hx])
    val_hx = eval_hx(x_val,y_val).full()

    hy = ca.jacobian(h, y)
    eval_hy = ca.Function('hy', [x, y], [hy])
    val_hy = eval_hy(x_val,y_val).full()

    # % comput dy/dx, dkppa/dx
    val_Lyy_inv = np.linalg.pinv(val_Lyy.full())
    dkdx = np.dot(np.linalg.pinv(val_hy @ val_Lyy_inv @ ca.transpose(val_hy)),(val_hx - val_hy @ val_Lyy_inv @ val_Lyx))
    dydx = -val_Lyy_inv @ (ca.transpose(val_hy) @ dkdx + val_Lyx)

    # % gradient 
    g = val_fx + ca.transpose(dydx) @ val_fy
    # % Hessian
    H = val_Lxx.full() + val_Lxy @ dydx.full() - dydx.T @ val_hy.T @ dkdx 

    H, g = H.full(), g.full()

    return H, g
