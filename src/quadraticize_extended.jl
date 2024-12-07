"""
    $FUNCTIONNAME(pc::PlayerCost, x::SVector{nx}, u::SVector{nu}, t::AbstractFloat)

A convencience implementation of the cost quadraticization.
"""
function _quadraticize_ad(pc::PlayerCost, g::GeneralGame, x::SVector, u::SVector,
                          t::AbstractFloat)
    @warn "You are using the fallback quadraticization using ForwardDiff.
    Consider implementing a custom `quadraticize` for your `ControlSystem`
    type." maxlog=1

    # concatenate x and u into a single vector
    xu = vcat(x, u)
    xu_cost = xu -> pc(g, xu[1:g.nx], xu[g.nx+1:end], t)
    # we can compute the gradient and the hessian in one go
    diff_xu = DiffResults.HessianResult(xu)
    diff_xu = ForwardDiff.hessian!(diff_xu, xu_cost, xu)
    jacobian = DiffResults.gradient(diff_xu)
    hessian = DiffResults.hessian(diff_xu)
    l = jacobian[1:g.nx]
    Q = hessian[1:g.nx, 1:g.nx]
    r = jacobian[g.nx+1:end]
    R = hessian[g.nx+1:end, g.nx+1:end]
    return QuadraticPlayerCost(l, Q, r, R)
end

quadraticize!(qcache::QuadCache, pc::PlayerCost, g::GeneralGame, x::SVector,
              u::SVector, t::AbstractFloat) = _quadraticize_ad(pc, g, x, u, t)
