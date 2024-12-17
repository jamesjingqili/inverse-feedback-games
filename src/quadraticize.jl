using Infiltrator
"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

A convencience implementation of the cost quadraticization.
"""
function _quadraticize_ad2(pc::PlayerCost, g::GeneralGame, x::SVector, u::SVector,
                          t::AbstractFloat)
    @warn "You are using the fallback quadraticization using ForwardDiff.
    Consider implementing a custom `quadraticize` for your `ControlSystem`
    type." maxlog=1

    x_cost = x->pc(g, x, u, t)
    u_cost = u->pc(g, x, u, t)
    nx = length(x)
    nu = length(u)
    # we can compute the gradient and the hessian in one go
    diff_x = DiffResults.HessianResult(x)
    diff_x = ForwardDiff.hessian!(diff_x, x_cost, x)
    # the linear state component of the cost is the gradient in x
    l = DiffResults.gradient(diff_x)
    # the quadratic state component of the cost is the hessian in x
    Q = DiffResults.hessian(diff_x)
    # @infiltrate
    diff_u = DiffResults.HessianResult(u)
    diff_u = ForwardDiff.hessian!(diff_u, u_cost, u)
    # the linear control cost is the gradient in u
    r = DiffResults.gradient(diff_u)
    # the quadratic control component o the cost is the hessian in u
    R = DiffResults.hessian(diff_u)
    S = SMatrix{nx, nu}(zeros(nx, nu))
    # @infiltrate
    return QuadraticPlayerCost(l, Q, r, R, S)
end
function _quadraticize_ad(pc::PlayerCost, g::GeneralGame, x::SVector, u::SVector,
    t::AbstractFloat)
    @warn "You are using the fallback quadraticization using ForwardDiff.
    Consider implementing a custom `quadraticize` for your `ControlSystem`
    type." maxlog=1

    # concatenate x and u into a single vector
    nx = length(x)
    nu = length(u)
    xu = vcat(x, u)
    xu_cost = xu_vec->pc(g, xu_vec[1:nx], xu_vec[nx+1:end], t)
    # we can compute the gradient and the hessian in one go
    diff_xu = DiffResults.HessianResult(xu)
    diff_xu = ForwardDiff.hessian!(diff_xu, xu_cost, xu)
    jacobian = DiffResults.gradient(diff_xu)
    hessian = DiffResults.hessian(diff_xu)
    l = SVector{nx}(jacobian[1:nx])
    Q = SMatrix{nx, nx}(hessian[1:nx, 1:nx])
    r = SVector{nu}(jacobian[nx+1:end])
    R = SMatrix{nu, nu}(hessian[nx+1:end, nx+1:end])
    S = SMatrix{nx, nu}(hessian[1:nx, nx+1:end])
    # @infiltrate
    return QuadraticPlayerCost(l, Q, r, R, S)
end

quadraticize!(qcache::QuadCache, pc::PlayerCost, g::GeneralGame, x::SVector,
              u::SVector, t::AbstractFloat) = _quadraticize_ad(pc, g, x, u, t)
