# ------------------------------Markdown Language Header-----------------------
# # 3D Compressible Navier-Stokes Equations
#
#
#-
#
#-
# ## Introduction
#
# This example shows how to solve the 3D compressible Navier-Stokes equations using vanilla DG.
#
# ## Continuous Governing Equations
# We solve the following equation:
#
# ```math
# \frac{\partial \rho}{\partial t} + \nabla \cdot \mathbf{U} = 0 \; \; (1.1)
# ```
# ```math
# \frac{\partial \mathbf{U}}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \otimes \mathbf{U}}{\rho} + P \mathbf{I}_2 \right) + \rho g \hat{\mathbf{k}}= \nabla \cdot \mathbf{F}_U^{visc} \; \; (1.2)
# ```
# ```math
# \frac{\partial E}{\partial t} + \nabla \cdot \left( \frac{\mathbf{U} \left(E+P \right)}{\rho} \right = \nabla \cdot \mathbf{F}_E^{visc} \; \; (1.3)
# ```
# where $\mathbf{u}=(u,v,w)$ is the velocity, $\mathbf{U}=\rho \mathbf{u}$, is the momentum, with $\rho$ the total density and $E=(\gamma-1) \rho \left( c_v T + \frac{1}{2} \mathbf{u} \cdot \mathbf{u} + g z \right)$ the total energy (internal $+$ kinetic $+$ potential).
# The viscous fluxes are defined as follows
# ```math
# \mathbf{F}_U^{visc} = \mu \left\[ \nabla \mathbf{u} +  \lambda \left( \nabla \mathbf{u} \right)^T + \nabla \cdot \mathbf{u}  \mathbf{I}_2 \right\]
# ```
# and
# ```math
# \mathbf{F}_E^{visc} =  \mathbf{u} \cdot \mathbf{F}_U^{visc} + \frac{c_p/Pr} \nabla T
# ```
# where $\mu$ is the kinematic (or artificial) viscosity, $\lambda=-\frac{2}{3}$ is the Stokes hypothesis, $Pr \approx 0.71$ is the Prandtl number for air and $T$ is the temperature.
# We employ periodic boundary conditions in the horizontaland no-flux boundary conditions in the vertical.  At the bottom and top of the domain, we need to impose no-flux boundary conditions in $\nabla T$ to avoid a (artificial) thermal boundary layer.
#
#-
# ## Discontinous Galerkin Method
# To solve Eq.\ (1) we use the discontinuous Galerkin method with basis functions comprised of Lagrange polynomials based on Lobatto points. Multiplying Eq.\ (1) by a test function $\psi$ and integrating within each element $\Omega_e$ such that $\Omega = \bigcup_{e=1}^{N_e} \Omega_e$ we get
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Omega_e} \psi \nabla \cdot \mathbf{F}^{(e)}_N d\Omega_e =  \int_{\Omega_e} \psi S\left( q^{(e)}_N} \right) d\Omega_e \; \; (2)
# ```
# where $\mathbf{q}^{(e)}_N=\sum_{i=1}^{(N+1)^{dim}} \psi_i(\mathbf{x}) \mathbf{q}_i(t)$ is the finite dimensional expansion with basis functions $\psi(\mathbf{x})$, where $\mathbf{q}=\left( \rho, \mathbf{U}^T, E \right)^T$,
# ```math
# \mathbf{F}=\left( \mathbf{U}, \frac{\mathbf{U} \otimes \mathbf{U}}{\rho} + P \mathbf{I}_2,   \frac{\mathbf{U} \left(E+P \right)}{\rho} \right).
# ```
# and
# ```math
#  S\left( q^{(e)}_N} \right)  = \nu \left( \nabla^2 \rho, \nabla^2 \mathbf{U}, \nabla^2 E \right).
# ```
# Integrating Eq.\ (2) by parts yields
#
# ```math
# \int_{\Omega_e} \psi \frac{\partial \mathbf{q}^{(e)}_N}{\partial t} d\Omega_e + \int_{\Gamma_e} \psi \mathbf{n} \cdot \mathbf{F}^{(*,e)}_N d\Gamma_e - \int_{\Omega_e} \nabla \psi \cdot \mathbf{F}^{(e)}_N d\Omega_e = \int_{\Omega_e} \psi S\left( q^{(e)}_N} \right) d\Omega_e \; \; (3)
# ```
#
# where the second term on the left denotes the flux integral term (computed in "function flux\_rhs") and the third term denotes the volume integral term (computed in "function volume\_rhs").  The superscript $(*,e)$ in the flux integral term denotes the numerical flux. Here we use the Rusanov flux.
#
#-
# ## Local Discontinous Galerkin Method
# To approximate the second order terms on the right hand side of Eq.\ (1) we use the local discontinuous Galerkin (LDG) method, which we described in LDG2d.jl. We will highlight the main steps below for completeness. We employ the following two-step process: first we approximate the gradient of $q$ as follows
# ```math
# \mathbf{Q}(\mathbf{x}) = \nabla \vc{q}(\mathbf{x}) \; \; (2)
# ```
# where $\mathbf{Q}$ is an auxiliary vector function, followed by
# ```math
# \nabla \cdot \left \mathbf{F}^{visc}\left( \mathbf{Q} \right) \; \; (3)
# ```
# which completes the approximation of the second order derivatives.
#
#-
# ## Commented Program
#
#--------------------------------Markdown Language Header-----------------------
include(joinpath(@__DIR__,"InitPackages.jl"))

using ..PlanetParameters
using MPI
using ..Canary
using Roots
using DelimitedFiles
using Dierckx
using Printf: @sprintf
const HAVE_CUDA = try
    using CUDAnative
    using CUDAdrv
    true
catch
    false
end
if HAVE_CUDA
    macro hascuda(ex)
        return :($(esc(ex)))
    end
else
    macro hascuda(ex)
        return :()
    end
end

# {{{ reshape for CuArray
@hascuda function Base.reshape(A::CuArray, dims::NTuple{N, Int}) where {N}
    @assert prod(dims) == prod(size(A))
    CuArray{eltype(A), length(dims)}(dims, A.buf)
end
# }}}


const _nsd = 3 # number of spatial dimensions
_icase = 1003
# note the order of the fields below is also assumed in the code.
#
#
# FIND A WAY TO CLEAR PREVIPOUSLY COMPILED CONSTS
#
#
# DEFINE CASE AND PRE_COMPILED QUANTITIES:
#

# _icase = 1    #RTB
# _icase = 1001
# _icase = 1003
 _icase = 1010
# _icase = 1201
if(_icase < 1000) 
	DRY_CASE = true
else
	DRY_CASE = false 
end 
# }}}
# -------- ASR -------- # 

# {{{ constants
# note the order of the fields below is also assumed in the code.

if  DRY_CASE
	const _ntracers = 0
	const _nstate = _ntracers + _nsd + 2 
        const _U, _V, _W, _ρ, _E = 1:_nstate
	const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E)
        _xmin, _xmax = 0.0, 1000.0
        _ymin, _ymax = 0.0, 1000.0
        _zmin, _zmax = 0.0, 1000.0

    elseif (_icase == 1001)  # Single Tracer 
	const _ntracers = 1
        const _nstate = (_nsd + 2) + _ntracers
	const _U, _V, _W, _ρ, _E, _qt = 1:_nstate
	const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E, qt = _qt)
        _xmin, _xmax = 0.0, 1000.0
        _ymin, _ymax = 0.0, 1000.0
        _zmin, _zmax = 0.0, 1000.0
    
    elseif (_icase == 1002) # Three Tracer Dry Test
        const _ntracers = 3
        const _nstate = (_nsd + 2) + _ntracers
        const _U, _V, _W, _ρ, _E, _qt1, _qt2, _qt3 = 1:_nstate
        const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E, qt1 = _qt1, qt2 = _qt2, qt3 = _qt3)
        _xmin, _xmax = 0.0, 1000.0
        _ymin, _ymax = 0.0, 1000.0
        _zmin, _zmax = 0.0, 1000.0
    elseif (_icase == 1003)
        const _ntracers = 3
        const _nstate = _nsd + 2 + _ntracers
        const _U, _V, _W, _ρ, _E, _qt1, _qt2, _qt3 = 1:_nstate
        const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E, qt1 = _qt1, qt2 = _qt2, qt3 = _qt3)
        _xmin, _xmax = 0.0, 1000.0
        _ymin, _ymax = 0.0, 1000.0
        _zmin, _zmax = 0.0, 1000.0
    elseif (_icase == 1010)
        const _ntracers = 3
        const _nstate = (_nsd + 2) + _ntracers
        const _U, _V, _W, _ρ, _E, _qt1, _qt2, _qt3 = 1:_nstate
        const stateid = (U = _U, V = _V, W = _W, ρ = _ρ, E = _E, qt1 = _qt1, qt2 = _qt2, qt3 = _qt3)
        _xmin, _xmax = 0.0, 20000.0
        _ymin, _ymax = 0.0, 10000.0
        _zmin, _zmax = 0.0, 10000.0
    else
        error("Please enter a valid", _icase, "case id. Currently, <900, 1003, 1010 are valid entries")
end

const _nvgeo = 14
const _ξx, _ηx, _ζx, _ξy, _ηy, _ζy, _ξz, _ηz, _ζz, _MJ, _MJI,
_x, _y, _z = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx, ζx = _ζx,
                ξy = _ξy, ηy = _ηy, ζy = _ζy,
                ξz = _ξz, ηz = _ηz, ζz = _ζz,
                MJ = _MJ, MJI = _MJI,
                x = _x,   y = _y,   z = _z)

const _nsgeo = 5
const _nx, _ny, _nz, _sMJ, _vMJI = 1:_nsgeo
const sgeoid = (nx = _nx, ny = _ny, nz = _nz, sMJ = _sMJ, vMJI = _vMJI)

const _γ        = 14  // 10
const _p0       = 100000
const _R_gas    = 28717 // 100
const _c_p      = 100467 // 100
const _c_v      = 7175 // 10
const _gravity  = 10
const _Prandtl  = 71 // 100
const _Stokes   = -2 // 3
# }}}



# {{{ courant
function courantnumber(::Val{dim}, ::Val{N}, vgeo, Q, mpicomm) where {dim, N}
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)

    dt = [floatmax(DFloat)]
    Courant = - [floatmax(DFloat)]

    # Allocate tracers
    q_tr = zeros(DFloat, max(3,_ntracers))
    q_l, q_i = zeros(DFloat, 1), zeros(DFloat, 1)
    
    #Compute DT
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
        ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
        ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]
        z = vgeo[n, _z, e]
        P = (R_gas/c_v) * (E - (U^2 + V^2 + W^2)/(2*ρ) - ρ * gravity * z)
        u, v, w = U/ρ, V/ρ, W/ρ
        dx              = sqrt( (1.0/(2*ξx))^2 + 0*(1.0/(2*ηy))^2  + (1.0/(2*ζz))^2 )
        vel             = sqrt( u^2 + v^2 + w^2)
        wave_speed      = (vel + sqrt(γ * P / ρ))
        loc_dt          = 1.0*dx/wave_speed/N
        dt[1]           = min(dt[1], loc_dt)

    end
    dt_min=MPI.Allreduce(dt[1], MPI.MIN, mpicomm)

    #Compute Courant
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        ξx, ξy, ξz = vgeo[n, _ξx, e], vgeo[n, _ξy, e], vgeo[n, _ξz, e]
        ηx, ηy, ηz = vgeo[n, _ηx, e], vgeo[n, _ηy, e], vgeo[n, _ηz, e]
        ζx, ζy, ζz = vgeo[n, _ζx, e], vgeo[n, _ζy, e], vgeo[n, _ζz, e]
        z = vgeo[n, _z, e]
        
        u, v, w = U/ρ, V/ρ, W/ρ
        dx=sqrt( (1.0/(2*ξx))^2 + 0*(1.0/(2*ηy))^2  + (1.0/(2*ζz))^2 )
        vel=sqrt( u^2 + v^2 + w^2)
        P = (R_gas/c_v)*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ*gravity*z)
        wave_speed = (vel + sqrt(γ * P / ρ))
        loc_Courant = wave_speed*dt_min*N/dx
        Courant[1] = max(Courant[1], loc_Courant)
    end
 
    Courant_max=MPI.Allreduce(Courant[1], MPI.MAX, mpicomm)
    (dt_min, Courant_max)
end
# }}}

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM) where dim
    # Compute metric terms
    Nq = size(D, 1)
    DFloat = eltype(D)

    (nface, nelem) = size(mesh.elemtoelem)

    crd = creategrid(Val(dim), mesh.elemtocoord, ξ)

    vgeo = zeros(DFloat, Nq^dim, _nvgeo, nelem)
    sgeo = zeros(DFloat, _nsgeo, Nq^(dim-1), nface, nelem)

    (ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, MJ, MJI, x, y, z) =
        ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
    J = similar(x)
    (nx, ny, nz, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
    sJ = similar(sMJ)

    X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
    creategrid!(X..., mesh.elemtocoord, ξ)

    @inbounds for j = 1:length(x)
        (x[j], y[j], z[j]) = meshwarp(x[j], y[j], z[j])
    end

    # Compute the metric terms
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)

    M = kron(1, ntuple(j->ω, dim)...)
    MJ .= M .* J
    MJI .= 1 ./ MJ
    vMJI .= MJI[vmapM]

    sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
    sMJ .= sM .* sJ

    (vgeo, sgeo)
end
# }}}

# {{{ CPU Kernels
#
# --------ASR --------#
function compute_wave_speed(nxM, nyM, nzM, ρMinv, UM, VM, WM, PM, ρPinv, UP, VP, WP, PP, γ)
        
        λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + sqrt(ρMinv * γ * PM)
        λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + sqrt(ρPinv * γ * PP)
        λ  =  max(λM, λP)
        return λ
end

# --------ASR --------#
function integrate_volume_rhs!(rhs, e, D, Nq, s_F, s_G, s_H)

          # loop of ξ-grid lines 
          for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
              rhs[i, j, k, s, e] += D[n, i] * s_F[n, j, k, s]
          end
          # loop of η-grid lines 
          for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
              rhs[i, j, k, s, e] += D[n, j] * s_G[i, n, k, s]
          end
          # loop of ζ-grid lines 
          for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
              rhs[i, j, k, s, e] += D[n, k] * s_H[i, j, n, s]
          end         
end

# --------ASR --------#
function build_volume_fluxes_ijke( MJ,
                                   ξx,fluxQ_x,
                                   ξy,fluxQ_y,
                                   ξz,fluxQ_z)
        return MJ * (ξx * fluxQ_x + ξy * fluxQ_y + ξz * fluxQ_z)
        #return volume_fluxes
end

# ------ ASR --------- # 
function build_surface_fluxes_ijke( nxM, fluxQM_x, fluxQP_x,
                                    nyM, fluxQM_y, fluxQP_y,
                                    nzM, fluxQM_z, fluxQP_z,
                                    λ, QM,QP)
        return (nxM * (fluxQM_x + fluxQP_x) + nyM * (fluxQM_y + fluxQP_y) + nzM * (fluxQM_z + fluxQP_z) - λ * (QP -QM)) / 2
        #return surface_flux 
end

# }}}

# {{{ CPU Kernels
# Volume RHS
function volume_rhs!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where {dim, N}
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Nq = N + 1
    nelem = size(Q)[end]

    Q           = reshape(Q, Nq, Nq, Nq, _nstate+3, nelem)
    rhs         = reshape(rhs, Nq, Nq, Nq, _nstate, nelem)
    vgeo        = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)
    

    # Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)

    # Tracer quantity initialisation ASR 
    fluxQT_x	 = zeros(DFloat, _ntracers)
    fluxQT_y	 = zeros(DFloat, _ntracers)
    fluxQT_z	 = zeros(DFloat, _ntracers)

    # Allocate at least 3 spaces to q_tr
    ntracers     = max(3,_ntracers)
    q_tr         = zeros(DFloat, ntracers)

    @inbounds for e in elems
        for k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
            z = vgeo[i,j,k,_z,e]
            
            #=Moist air quantities: Rm
            for itracer = 1:_ntracers
                istate = itracer + (_nsd + 2)
                
                q_tr[itracer] = Q[i,j,k,istate,e]
            end
            =#
            R_gas   = MoistThermodynamics.gas_constant_air(0.0,0.0,0.0)
            U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
            ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]
            P = (R_gas/c_v)*(E - (U^2 + V^2+ W^2)/(2*ρ) - ρ * gravity * z)
            ρinv = 1 / ρ
            fluxρ_x = U
            fluxU_x = ρinv * U * U + P
            fluxV_x = ρinv * V * U
            fluxW_x = ρinv * W * U
            fluxE_x = ρinv * U * (E+P)
	    
            fluxρ_y = V
            fluxU_y = ρinv * U * V
            fluxV_y = ρinv * V * V + P
            fluxW_y = ρinv * W * V
            fluxE_y = ρinv * V * (E+P)
	       
	    fluxρ_z = W
            fluxU_z = ρinv * U * W
            fluxV_z = ρinv * V * W
            fluxW_z = ρinv * W * W + P
            fluxE_z = ρinv * W * (E+P)

            #--------- ASR ----------#
            s_F[i, j, k, _ρ] = build_volume_fluxes_ijke(MJ, ξx, fluxρ_x, ξy, fluxρ_y, ξz, fluxρ_z)
            s_F[i, j, k, _U] = build_volume_fluxes_ijke(MJ, ξx, fluxU_x, ξy, fluxU_y, ξz, fluxU_z)
            s_F[i, j, k, _V] = build_volume_fluxes_ijke(MJ, ξx, fluxV_x, ξy, fluxV_y, ξz, fluxV_z)
            s_F[i, j, k, _W] = build_volume_fluxes_ijke(MJ, ξx, fluxW_x, ξy, fluxW_y, ξz, fluxW_z)
            s_F[i, j, k, _E] = build_volume_fluxes_ijke(MJ, ξx, fluxE_x, ξy, fluxE_y, ξz, fluxE_z)

            s_G[i, j, k, _ρ] = build_volume_fluxes_ijke(MJ, ηx, fluxρ_x, ηy, fluxρ_y, ηz, fluxρ_z)
            s_G[i, j, k, _U] = build_volume_fluxes_ijke(MJ, ηx, fluxU_x, ηy, fluxU_y, ηz, fluxU_z)
            s_G[i, j, k, _V] = build_volume_fluxes_ijke(MJ, ηx, fluxV_x, ηy, fluxV_y, ηz, fluxV_z)
            s_G[i, j, k, _W] = build_volume_fluxes_ijke(MJ, ηx, fluxW_x, ηy, fluxW_y, ηz, fluxW_z)
            s_G[i, j, k, _E] = build_volume_fluxes_ijke(MJ, ηx, fluxE_x, ηy, fluxE_y, ηz, fluxE_z)

            s_H[i, j, k, _ρ] = build_volume_fluxes_ijke(MJ, ζx, fluxρ_x, ζy, fluxρ_y, ζz, fluxρ_z)
            s_H[i, j, k, _U] = build_volume_fluxes_ijke(MJ, ζx, fluxU_x, ζy, fluxU_y, ζz, fluxU_z)
            s_H[i, j, k, _V] = build_volume_fluxes_ijke(MJ, ζx, fluxV_x, ζy, fluxV_y, ζz, fluxV_z)
            s_H[i, j, k, _W] = build_volume_fluxes_ijke(MJ, ζx, fluxW_x, ζy, fluxW_y, ζz, fluxW_z)
            s_H[i, j, k, _E] = build_volume_fluxes_ijke(MJ, ζx, fluxE_x, ζy, fluxE_y, ζz, fluxE_z)

            #-------- ASR ----------#    
	    # Tracers elementwise loop ASR 
	    @inbounds for itracer = 1:_ntracers
                istate = itracer + (_nsd+2) 
                
                QT     = Q[i, j, k, istate, e]
                fluxQT_x = ρinv * U * QT 
                fluxQT_y = ρinv * V * QT
                fluxQT_z = ρinv * W * QT
                
                s_F[i,j,k,istate] = build_volume_fluxes_ijke(MJ, ξx, fluxQT_x, ξy, fluxQT_y, ξz, fluxQT_z)
                s_G[i,j,k,istate] = build_volume_fluxes_ijke(MJ, ηx, fluxQT_x, ηy, fluxQT_y, ηz, fluxQT_z)
                s_H[i,j,k,istate] = build_volume_fluxes_ijke(MJ, ζx, fluxQT_x, ζy, fluxQT_y, ζz, fluxQT_z)
            end

            # buoyancy term
            rhs[i, j, k, _W, e] -= MJ * ρ * gravity
        end
            integrate_volume_rhs!(rhs, e, D, Nq, s_F, s_G, s_H)
    end
end

# Flux RHS
function flux_rhs!(::Val{dim}, ::Val{N}, rhs::Array, Q, sgeo, vgeo, elems, vmapM,
                  vmapP, elemtobndy) where {dim, N}
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np          = (N+1)^dim
    Nfp         = (N+1)^(dim-1)
    nface       = 2*dim
    
    ntracers = max(3,_ntracers)
    q_tr     = zeros(DFloat,ntracers)

    # Allocate and initialize to zero tracer flux quantities
    QTM		= zeros(DFloat, _ntracers)
    QTP		= zeros(DFloat, _ntracers)
    fluxQTM_x	= zeros(DFloat, _ntracers)
    fluxQTM_y	= zeros(DFloat, _ntracers)
    fluxQTM_z	= zeros(DFloat, _ntracers)
    fluxQTP_x	= zeros(DFloat, _ntracers)
    fluxQTP_y	= zeros(DFloat, _ntracers)
    fluxQTP_z	= zeros(DFloat, _ntracers)
    
    
    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1
                
                # M - left variables and # P - right variables 
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]
                EM = Q[vidM, _E, eM]
                um, vm, wm = UM/ρM, VM/ρM, WM/ρM
                zM = vgeo[vidM, _z, eM]

                R_gas   = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
                PM = (R_gas/c_v) * (EM - (UM^2 + VM^2 + WM^2)/(2*ρM) - ρM * gravity * zM)
                
                #Tracer variables
                @inbounds for itracer = 1:_ntracers
                    istate = itracer + (_nsd+2)
                    QTM[itracer] = Q[vidM, istate, eM]
                    QTP[itracer] = zero(eltype(Q))
                end
                
                bc = elemtobndy[f, e]
                ρP = UP = VP = WP = EP = PP = zero(eltype(Q))

                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    WP = Q[vidP, _W, eP]
                    EP = Q[vidP, _E, eP]
                    up, vp, wp = UP/ρP, VP/ρP, WP/ρP
                    zP = vgeo[vidP, _z, eP]
                    
                    # Tracers
                    @inbounds for itracer = 1:_ntracers
                        istate = itracer + (_nsd + 2) 
                        QTP[itracer] = Q[vidP, istate, eP]
                    end
                    
                    PP = (R_gas/c_v)*(EP - (UP^2 + VP^2 + WP^2)/(2*ρP) - ρP * gravity * zP)

                elseif bc == 1
                    UnM = nxM * UM + nyM * VM + nzM * WM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    WP = WM - 2 * UnM * nzM
                    ρP = ρM
                    EP = EM
                    PP = PM

                    @inbounds for itracer = 1:_ntracers
                        QTP[itracer] = QTM[itracer] 
                    end

                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end
                
                #Left fluxes
                ρMinv = 1 / ρM
                fluxρM_x = UM
                fluxUM_x = ρMinv * UM * UM + PM
                fluxVM_x = ρMinv * VM * UM
                fluxWM_x = ρMinv * WM * UM
                fluxEM_x = ρMinv * UM * (EM+PM)

                fluxρM_y = VM
                fluxUM_y = ρMinv * UM * VM
                fluxVM_y = ρMinv * VM * VM + PM
                fluxWM_y = ρMinv * WM * VM
                fluxEM_y = ρMinv * VM * (EM+PM)

                fluxρM_z = WM
                fluxUM_z = ρMinv * UM * WM
                fluxVM_z = ρMinv * VM * WM
                fluxWM_z = ρMinv * WM * WM + PM
                fluxEM_z = ρMinv * WM * (EM+PM)

                ρPinv = 1 / ρP
                fluxρP_x = UP
                fluxUP_x = ρPinv * UP * UP + PP
                fluxVP_x = ρPinv * VP * UP
                fluxWP_x = ρPinv * WP * UP
                fluxEP_x = ρPinv * UP * (EP+PP)

                fluxρP_y = VP
                fluxUP_y = ρPinv * UP * VP
                fluxVP_y = ρPinv * VP * VP + PP
                fluxWP_y = ρPinv * WP * VP
                fluxEP_y = ρPinv * VP * (EP+PP)

                fluxρP_z = WP
                fluxUP_z = ρPinv * UP * WP
                fluxVP_z = ρPinv * VP * WP
                fluxWP_z = ρPinv * WP * WP + PP
                fluxEP_z = ρPinv * WP * (EP+PP)

                # ------- ASR --------- # 
                λ  = compute_wave_speed(nxM, nyM, nzM, ρMinv, UM, VM, WM, PM, ρPinv, UP, VP, WP, PP, γ)
                # ------- ASR --------- #

                # Compute using generic function, the flux integrand
                fluxρS = build_surface_fluxes_ijke(nxM, fluxρM_x, fluxρP_x, nyM, fluxρM_y, fluxρP_y, nzM, fluxρM_z, fluxρP_z, λ, ρM, ρP)
                fluxUS = build_surface_fluxes_ijke(nxM, fluxUM_x, fluxUP_x, nyM, fluxUM_y, fluxUP_y, nzM, fluxUM_z, fluxUP_z, λ, UM, UP)
                fluxVS = build_surface_fluxes_ijke(nxM, fluxVM_x, fluxVP_x, nyM, fluxVM_y, fluxVP_y, nzM, fluxVM_z, fluxVP_z, λ, VM, VP)
                fluxWS = build_surface_fluxes_ijke(nxM, fluxWM_x, fluxWP_x, nyM, fluxWM_y, fluxWP_y, nzM, fluxWM_z, fluxWP_z, λ, WM, WP)
                fluxES = build_surface_fluxes_ijke(nxM, fluxEM_x, fluxEP_x, nyM, fluxEM_y, fluxEP_y, nzM, fluxEM_z, fluxEP_z, λ, EM, EP)
    
                # ------- ASR -------- #
                
                #Update RHS
                rhs[vidM, _ρ, eM] -= sMJ * fluxρS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
                rhs[vidM, _V, eM] -= sMJ * fluxVS
                rhs[vidM, _W, eM] -= sMJ * fluxWS
                rhs[vidM, _E, eM] -= sMJ * fluxES
                
                #Tracers
                @inbounds for itracer = 1:_ntracers
                    istate = itracer + (_nsd + 2) 

                    fluxQTM_x = ρMinv * UM * QTM[itracer] 
                    fluxQTM_y = ρMinv * VM * QTM[itracer] 
                    fluxQTM_z = ρMinv * WM * QTM[itracer] 

                    fluxQTP_x = ρPinv * UP * QTP[itracer] 
                    fluxQTP_y = ρPinv * VP * QTP[itracer] 
                    fluxQTP_z = ρPinv * WP * QTP[itracer] 
                    
                    fluxQTS = build_surface_fluxes_ijke(nxM, fluxQTM_x, fluxQTP_x, 
                                                        nyM, fluxQTM_y, fluxQTP_y, 
                                                        nzM, fluxQTM_z, fluxQTP_z, 
                                                        λ, QTM[itracer], QTP[itracer])
                    # Update RHS (Integration) 
                    rhs[vidM, istate, eM] -= sMJ * fluxQTS               
                end
            end
        end
    end
end
# }}}

# {{{ Volume grad(Q)
function volume_grad!(::Val{dim}, ::Val{N}, rhs::Array, Q, vgeo, D, elems) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Nq = N + 1
    nelem = size(Q)[end]
    @show(size(Q))
    Q           = reshape(Q, Nq, Nq, Nq, _nstate+3, nelem)
    rhs         = reshape(rhs, Nq, Nq, Nq, _nstate, dim, nelem)
    vgeo        = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate, dim)
    s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate, dim)
    s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate, dim)
    
    # Allocate at least 3 spaces to q_tr
    q_tr = zeros(DFloat, max(3,_ntracers))

    @inbounds for e in elems
        for k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]
            z = vgeo[i,j,k,_z,e]

            U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
            ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]
           
            # Moist air quantities
            #=
            for itracer = 1:_ntracers
                istate = itracer + (_nsd + 2)
                q_tr[itracer]  = Q[i, j, k, istate, e]
            end
            =#
            R_gas  = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
            P = (R_gas/c_v)*(E - (U^2 + V^2 + W^2)/(2*ρ) - ρ * gravity * z)
            
            #Primite variables
            u = U/ρ
            v = V/ρ
            w = W/ρ
            T = P/(R_gas * ρ)

            #Compute fluxes
            fluxρ = ρ
            fluxU = u
            fluxV = v
            fluxW = w
            fluxE = T
            
            s_F[i, j, k, _ρ, 1], s_F[i, j, k, _ρ, 2], s_F[i, j, k, _ρ, 3] = build_volume_fluxes_ijke(MJ,ξx,fluxρ,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ξy,fluxρ,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ξz,fluxρ)
            s_F[i, j, k, _U, 1], s_F[i, j, k, _U, 2], s_F[i, j, k, _U, 3] = build_volume_fluxes_ijke(MJ,ξx,fluxU,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ξy,fluxU,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ξz,fluxU)
            s_F[i, j, k, _V, 1], s_F[i, j, k, _V, 2], s_F[i, j, k, _V, 3] = build_volume_fluxes_ijke(MJ,ξx,fluxV,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ξy,fluxV,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ξz,fluxV)
            s_F[i, j, k, _W, 1], s_F[i, j, k, _W, 2], s_F[i, j, k, _W, 3] = build_volume_fluxes_ijke(MJ,ξx,fluxW,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ξy,fluxW,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ξz,fluxW)
            s_F[i, j, k, _E, 1], s_F[i, j, k, _E, 2], s_F[i, j, k, _E, 3] = build_volume_fluxes_ijke(MJ,ξx,fluxE,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ξy,fluxE,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ξz,fluxE)

            s_G[i, j, k, _ρ, 1], s_G[i, j, k, _ρ, 2], s_G[i, j, k, _ρ, 3] = build_volume_fluxes_ijke(MJ,ηx,fluxρ,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ηy,fluxρ,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ηz,fluxρ)
            s_G[i, j, k, _U, 1], s_G[i, j, k, _U, 2], s_G[i, j, k, _U, 3] = build_volume_fluxes_ijke(MJ,ηx,fluxU,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ηy,fluxU,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ηz,fluxU)
            s_G[i, j, k, _V, 1], s_G[i, j, k, _V, 2], s_G[i, j, k, _V, 3] = build_volume_fluxes_ijke(MJ,ηx,fluxV,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ηy,fluxV,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ηz,fluxV)
            s_G[i, j, k, _W, 1], s_G[i, j, k, _W, 2], s_G[i, j, k, _W, 3] = build_volume_fluxes_ijke(MJ,ηx,fluxW,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ηy,fluxW,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ηz,fluxW)
            s_G[i, j, k, _E, 1], s_G[i, j, k, _E, 2], s_G[i, j, k, _E, 3] = build_volume_fluxes_ijke(MJ,ηx,fluxE,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ηy,fluxE,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ηz,fluxE)

            s_H[i, j, k, _ρ, 1], s_H[i, j, k, _ρ, 2], s_H[i, j, k, _ρ, 3] = build_volume_fluxes_ijke(MJ,ζx,fluxρ,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ζy,fluxρ,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ζz,fluxρ)
            s_H[i, j, k, _U, 1], s_H[i, j, k, _U, 2], s_H[i, j, k, _U, 3] = build_volume_fluxes_ijke(MJ,ζx,fluxU,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ζy,fluxU,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ζz,fluxU)
            s_H[i, j, k, _V, 1], s_H[i, j, k, _V, 2], s_H[i, j, k, _V, 3] = build_volume_fluxes_ijke(MJ,ζx,fluxV,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ζy,fluxV,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ζz,fluxV)
            s_H[i, j, k, _W, 1], s_H[i, j, k, _W, 2], s_H[i, j, k, _W, 3] = build_volume_fluxes_ijke(MJ,ζx,fluxW,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ζy,fluxW,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ζz,fluxW)
            s_H[i, j, k, _E, 1], s_H[i, j, k, _E, 2], s_H[i, j, k, _E, 3] = build_volume_fluxes_ijke(MJ,ζx,fluxE,0.0,0.0,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ζy,fluxE,0.0,0.0),
                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ζz,fluxE)        
            
            # Tracers
            for itracer = 1:_ntracers
                istate = itracer + (_nsd + 2) 
                
                # Compute fluxes
                QT = Q[i, j, k, istate, e]
                fluxQT = QT 
 
                s_F[i, j, k, istate, 1], s_F[i, j, k, istate, 2], s_F[i, j, k, istate, 3] = build_volume_fluxes_ijke(MJ,ξx,fluxQT,0.0,0.0,0.0,0.0),
                                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ξy,fluxQT,0.0,0.0),
                                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ξz,fluxQT)
                s_G[i, j, k, istate, 1], s_G[i, j, k, istate, 2], s_G[i, j, k, istate, 3] = build_volume_fluxes_ijke(MJ,ηx,fluxQT,0.0,0.0,0.0,0.0),
                                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ηy,fluxQT,0.0,0.0),
                                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ηz,fluxQT)
                s_H[i, j, k, istate, 1], s_H[i, j, k, istate, 2], s_H[i, j, k, istate, 3] = build_volume_fluxes_ijke(MJ,ζx,fluxQT,0.0,0.0,0.0,0.0),
                                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,ζy,fluxQT,0.0,0.0),
                                                                                            build_volume_fluxes_ijke(MJ,0.0,0.0,0.0,0.0,ζz,fluxQT)
            end        
        end

        # loop of ξ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, i] * s_F[n, j, k, s, 1]
            rhs[i, j, k, s, 2, e] -= D[n, i] * s_F[n, j, k, s, 2]
            rhs[i, j, k, s, 3, e] -= D[n, i] * s_F[n, j, k, s, 3]
        end
        # loop of η-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, j] * s_G[i, n, k, s, 1]
            rhs[i, j, k, s, 2, e] -= D[n, j] * s_G[i, n, k, s, 2]
            rhs[i, j, k, s, 3, e] -= D[n, j] * s_G[i, n, k, s, 3]
        end
        # loop of ζ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, k] * s_H[i, j, n, s, 1]
            rhs[i, j, k, s, 2, e] -= D[n, k] * s_H[i, j, n, s, 2]
            rhs[i, j, k, s, 3, e] -= D[n, k] * s_H[i, j, n, s, 3]
        end
    end
end
# }}}

# Flux grad(Q)
function flux_grad!(::Val{dim}, ::Val{N}, rhs::Array,  Q, sgeo, vgeo, elems, vmapM, vmapP, elemtobndy) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    Nfp = (N+1)^(dim-1)
    nface = 2*dim
        
    QTM = zeros(DFloat, _ntracers)
    QTP = zeros(DFloat, _ntracers)
    
    q_tr = zeros(DFloat, max(3,_ntracers))

    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Left variables
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]
                EM = Q[vidM, _E, eM]
                uM, vM, wM = UM/ρM, VM/ρM, WM/ρM
                zM = vgeo[vidM, _z, eM]
                
                #=
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd + 2) 
                    q_tr[itracer] = Q[vidM, istate, eM]
                end
                =#
                R_gas  = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)           
                PM     = (R_gas/c_v) * (EM - (UM^2 + VM^2 + WM^2)/(2*ρM)- ρM * gravity * zM) 
                uM     = UM / ρM
                vM     = VM / ρM
                wM     = WM / ρM
                TM     = PM/(R_gas*ρM)

                #Right variables
                bc = elemtobndy[f, e]
                ρP = UP = VP = WP = EP = PP = zero(eltype(Q))
                
                # Tracers
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd + 2)      
                    QTM[itracer] = Q[vidM, istate, eM]
                    QTP[itracer] = zero(eltype(Q))
                end

                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    WP = Q[vidP, _W, eP]
                    EP = Q[vidP, _E, eP]
                    zP = vgeo[vidP, _z, eP]
                    PP     = (R_gas/c_v) * (EP - (UP^2 + VP^2 + WP^2)/(2*ρP)- ρP * gravity * zP) 
                    TP      = PP/(R_gas * ρP) 
                    uP=UP/ρP
                    vP=VP/ρP
                    wP=WP/ρP
                    
                    for itracer = 1:_ntracers
                        istate = itracer + (_nsd + 2) 
                        QTP[itracer] = Q[vidP, istate, eP]
                    end
                    
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM + nzM * WM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    WP = WM - 2 * UnM * nzM
                    ρP = ρM
                    EP = EM
                    PP = PM
                    uP = UP/ρP
                    vP = VP/ρP
                    wP = WP/ρP
                    TP = TM

                    for itracer = 1:_ntracers
                        istate = itracer + (_nsd + 2) 
                        QTP[itracer] = QTM[itracer]
                    end

                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                fluxρM = ρM
                fluxUM = uM
                fluxVM = vM
                fluxWM = wM
                fluxEM = TM

                #Right Fluxes
                fluxρP = ρP
                fluxUP = uP
                fluxVP = vP
                fluxWP = wP
                fluxEP = TP

                #Compute Numerical/Rusanov Flux
                fluxρS = 0.5*(fluxρM + fluxρP)
                fluxUS = 0.5*(fluxUM + fluxUP)
                fluxVS = 0.5*(fluxVM + fluxVP)
                fluxWS = 0.5*(fluxWM + fluxWP)
                fluxES = 0.5*(fluxEM + fluxEP)

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * nxM*fluxρS
                rhs[vidM, _ρ, 2, eM] += sMJ * nyM*fluxρS
                rhs[vidM, _ρ, 3, eM] += sMJ * nzM*fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * nxM*fluxUS
                rhs[vidM, _U, 2, eM] += sMJ * nyM*fluxUS
                rhs[vidM, _U, 3, eM] += sMJ * nzM*fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * nxM*fluxVS
                rhs[vidM, _V, 2, eM] += sMJ * nyM*fluxVS
                rhs[vidM, _V, 3, eM] += sMJ * nzM*fluxVS
                rhs[vidM, _W, 1, eM] += sMJ * nxM*fluxWS
                rhs[vidM, _W, 2, eM] += sMJ * nyM*fluxWS
                rhs[vidM, _W, 3, eM] += sMJ * nzM*fluxWS
                rhs[vidM, _E, 1, eM] += sMJ * nxM*fluxES
                rhs[vidM, _E, 2, eM] += sMJ * nyM*fluxES
                rhs[vidM, _E, 3, eM] += sMJ * nzM*fluxES
            
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd + 2) 
                    
                    # Compute numerical flux 
                    fluxQTM = QTM[itracer]
                    fluxQTP = QTP[itracer] 
                    fluxQTS = 0.5 * (fluxQTM + fluxQTP)
                    
                    # Update RHS (integration step)
                    rhs[vidM, istate, 1, eM] += sMJ * nxM * fluxQTS
                    rhs[vidM, istate, 2, eM] += sMJ * nyM * fluxQTS 
                    rhs[vidM, istate, 3, eM] += sMJ * nzM * fluxQTS 
                end
            end
        end
    end
end
# }}}

# {{{ Volume div(grad(Q))
function volume_div!(::Val{dim}, ::Val{N}, rhs::Array, gradQ, Q, vgeo, D, elems) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    Pr::DFloat = _Prandtl
    lambda::DFloat = _Stokes

    Nq = N + 1
    nelem = size(Q)[end]

    Q     = reshape(Q, Nq, Nq, Nq, _nstate+3, nelem)
    gradQ = reshape(gradQ, Nq, Nq, Nq, _nstate+3, dim, nelem)
    rhs   = reshape(rhs, Nq, Nq, Nq, _nstate, dim, nelem)
    vgeo  = reshape(vgeo, Nq, Nq, Nq, _nvgeo, nelem)

    #Initialize RHS vector
    fill!( rhs, zero(rhs[1]))

    #Allocate Arrays
    s_F = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_G = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)
    s_H = Array{DFloat}(undef, Nq, Nq, Nq, _nstate)

    @inbounds for e in elems
        for k = 1:Nq, j = 1:Nq, i = 1:Nq
            MJ = vgeo[i, j, k, _MJ, e]
            ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
            ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
            ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

            ρx, ρy, ρz = gradQ[i,j,k,_ρ,1,e], gradQ[i,j,k,_ρ,2,e], gradQ[i,j,k,_ρ,3,e]
            Ux, Uy, Uz = gradQ[i,j,k,_U,1,e], gradQ[i,j,k,_U,2,e], gradQ[i,j,k,_U,3,e]
            Vx, Vy, Vz = gradQ[i,j,k,_V,1,e], gradQ[i,j,k,_V,2,e], gradQ[i,j,k,_V,3,e]
            Wx, Wy, Wz = gradQ[i,j,k,_W,1,e], gradQ[i,j,k,_W,2,e], gradQ[i,j,k,_W,3,e]
            Ex, Ey, Ez = gradQ[i,j,k,_E,1,e], gradQ[i,j,k,_E,2,e], gradQ[i,j,k,_E,3,e]
            ρ, U, V, W = Q[i,j,k,_ρ,e], Q[i,j,k,_U,e], Q[i,j,k,_V,e], Q[i,j,k,_W,e]

            #Compute primitive variables
            ux, uy, uz  = Ux, Uy, Uz
            vx, vy, vz  = Vx, Vy, Vz
            wx, wy, wz  = Wx, Wy, Wz
            Tx, Ty, Tz  = Ex, Ey, Ez
            div_u       = ux + vy + wz
            u           = U/ρ
            v           = V/ρ
            w           = W/ρ

            #Compute fluxes
            fluxρ_x = 0*ρx
            fluxρ_y = 0*ρy
            fluxρ_z = 0*ρz
            fluxU_x = 2*ux + lambda*div_u
            fluxU_y = uy + vx
            fluxU_z = uz + wx
            fluxV_x = vx + uy
            fluxV_y = 2*vy + lambda*div_u
            fluxV_z = vz + wy
            fluxW_x = wx + uz
            fluxW_y = wy + vz
            fluxW_z = 2*wz + lambda*div_u
            fluxE_x = u*(2*ux + lambda*div_u) + v*(uy + vx) + w*(uz + wx) + c_p/Pr*Tx
            fluxE_y = u*(vx + uy) + v*(2*vy + lambda*div_u) + w*(vz + wy) + c_p/Pr*Ty
            fluxE_z = u*(wx + uz) + v*(wy + vz) + w*(2*wz + lambda*div_u) + c_p/Pr*Tz
            
            # ----- ASR ------- # 
            
            s_F[i, j, k, _ρ] = build_volume_fluxes_ijke(MJ, ξx, fluxρ_x, ξy, fluxρ_y, ξz, fluxρ_z)
            s_F[i, j, k, _U] = build_volume_fluxes_ijke(MJ, ξx, fluxU_x, ξy, fluxU_y, ξz, fluxU_z)
            s_F[i, j, k, _V] = build_volume_fluxes_ijke(MJ, ξx, fluxV_x, ξy, fluxV_y, ξz, fluxV_z)
            s_F[i, j, k, _W] = build_volume_fluxes_ijke(MJ, ξx, fluxW_x, ξy, fluxW_y, ξz, fluxW_z)
            s_F[i, j, k, _E] = build_volume_fluxes_ijke(MJ, ξx, fluxE_x, ξy, fluxE_y, ξz, fluxE_z)

            s_G[i, j, k, _ρ] = build_volume_fluxes_ijke(MJ, ηx, fluxρ_x, ηy, fluxρ_y, ηz, fluxρ_z)
            s_G[i, j, k, _U] = build_volume_fluxes_ijke(MJ, ηx, fluxU_x, ηy, fluxU_y, ηz, fluxU_z)
            s_G[i, j, k, _V] = build_volume_fluxes_ijke(MJ, ηx, fluxV_x, ηy, fluxV_y, ηz, fluxV_z)
            s_G[i, j, k, _W] = build_volume_fluxes_ijke(MJ, ηx, fluxW_x, ηy, fluxW_y, ηz, fluxW_z)
            s_G[i, j, k, _E] = build_volume_fluxes_ijke(MJ, ηx, fluxE_x, ηy, fluxE_y, ηz, fluxE_z)

            s_H[i, j, k, _ρ] = build_volume_fluxes_ijke(MJ, ζx, fluxρ_x, ζy, fluxρ_y, ζz, fluxρ_z)
            s_H[i, j, k, _U] = build_volume_fluxes_ijke(MJ, ζx, fluxU_x, ζy, fluxU_y, ζz, fluxU_z)
            s_H[i, j, k, _V] = build_volume_fluxes_ijke(MJ, ζx, fluxV_x, ζy, fluxV_y, ζz, fluxV_z)
            s_H[i, j, k, _W] = build_volume_fluxes_ijke(MJ, ζx, fluxW_x, ζy, fluxW_y, ζz, fluxW_z)
            s_H[i, j, k, _E] = build_volume_fluxes_ijke(MJ, ζx, fluxE_x, ζy, fluxE_y, ζz, fluxE_z)

            # ------ ASR ------ #   
            for itracer = 1:_ntracers
                istate = itracer + (_nsd + 2) 
                
                QTx = gradQ[i, j, k, istate, 1, e]
                QTy = gradQ[i, j, k, istate, 2, e]
                QTz = gradQ[i, j, k, istate, 3, e]
                QT =  Q[i, j, k, istate, e] 
            
                fluxQT_x = 1*QTx # SGS model requires eddy viscosity term so that  κ * QTx , SGS not implemented yet 
                fluxQT_y = 1*QTy 
                fluxQT_z = 1*QTz 
            
                s_F[i, j, k, istate] = build_volume_fluxes_ijke(MJ, ξx, fluxQT_x, ξy, fluxQT_y, ξz, fluxQT_z)
                s_G[i, j, k, istate] = build_volume_fluxes_ijke(MJ, ηx, fluxQT_x, ηy, fluxQT_y, ηz, fluxQT_z)
                s_H[i, j, k, istate] = build_volume_fluxes_ijke(MJ, ζx, fluxQT_x, ζy, fluxQT_y, ζz, fluxQT_z)

            end
        end

            # ------- ASR ------- # 
            
        # loop of ξ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, i] * s_F[n, j, k, s]
        end
        # loop of η-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, j] * s_G[i, n, k, s]
        end
        # loop of ζ-grid lines
        for s = 1:_nstate, k = 1:Nq, j = 1:Nq, i = 1:Nq, n = 1:Nq
            rhs[i, j, k, s, 1, e] -= D[n, k] * s_H[i, j, n, s]
        end
    end
end
# }}}

# Flux div(grad(Q))
function flux_div!(::Val{dim}, ::Val{N}, rhs::Array,  gradQ, Q, sgeo, elems, vmapM, vmapP, elemtobndy) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    Pr::DFloat = _Prandtl
    lambda::DFloat = _Stokes

    Np = (N+1)^dim
    Nfp = (N+1)^(dim-1)
    nface = 2*dim
    
    # Tracers
    QTxM        = zeros(DFloat, _ntracers)
    QTyM        = zeros(DFloat, _ntracers)
    QTzM        = zeros(DFloat, _ntracers)
    QTxP        = zeros(DFloat, _ntracers)
    QTyP        = zeros(DFloat, _ntracers)
    QTzP        = zeros(DFloat, _ntracers)
    QTM         = zeros(DFloat, _ntracers)
    QTP         = zeros(DFloat, _ntracers)


    @inbounds for e in elems
        for f = 1:nface
            for n = 1:Nfp
                (nxM, nyM, nzM, sMJ, ~) = sgeo[:, n, f, e]
                idM, idP = vmapM[n, f, e], vmapP[n, f, e]

                eM, eP = e, ((idP - 1) ÷ Np) + 1
                vidM, vidP = ((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1

                #Left variables
                ρxM = gradQ[vidM, _ρ, 1, eM]
                ρyM = gradQ[vidM, _ρ, 2, eM]
                ρzM = gradQ[vidM, _ρ, 3, eM]
                UxM = gradQ[vidM, _U, 1, eM]
                UyM = gradQ[vidM, _U, 2, eM]
                UzM = gradQ[vidM, _U, 3, eM]
                VxM = gradQ[vidM, _V, 1, eM]
                VyM = gradQ[vidM, _V, 2, eM]
                VzM = gradQ[vidM, _V, 3, eM]
                WxM = gradQ[vidM, _W, 1, eM]
                WyM = gradQ[vidM, _W, 2, eM]
                WzM = gradQ[vidM, _W, 3, eM]
                ExM = gradQ[vidM, _E, 1, eM]
                EyM = gradQ[vidM, _E, 2, eM]
                EzM = gradQ[vidM, _E, 3, eM]
                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]

                uM=UM/ρM
                vM=VM/ρM
                wM=WM/ρM
                uxM, uyM, uzM = UxM, UyM, UzM
                vxM, vyM, vzM = VxM, VyM, VzM
                wxM, wyM, wzM = WxM, WyM, WzM
                TxM, TyM, TzM = ExM, EyM, EzM

                #Right variables
                bc = elemtobndy[f, e]
                ρxP = ρyP = ρzP = zero(eltype(Q))
                UxP = UyP = UzP = zero(eltype(Q))
                VxP = VyP = VzP = zero(eltype(Q))
                WxP = WyP = WzP = zero(eltype(Q))
                ExP = EyP = EzP = zero(eltype(Q))
                
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd + 2) 
                 
                    QTxM[itracer] = gradQ[vidM, istate, 1, eM]
                    QTyM[itracer] = gradQ[vidM, istate, 2, eM]
                    QTzM[itracer] = gradQ[vidM, istate, 3, eM]
                    
                    QTxP[itracer] = zero(eltype(Q))
                    QTyP[itracer] = zero(eltype(Q))
                    QTzP[itracer] = zero(eltype(Q))
               
                end

                if bc == 0
                    ρxP = gradQ[vidP, _ρ, 1, eP]
                    ρyP = gradQ[vidP, _ρ, 2, eP]
                    ρzP = gradQ[vidP, _ρ, 3, eP]
                    UxP = gradQ[vidP, _U, 1, eP]
                    UyP = gradQ[vidP, _U, 2, eP]
                    UzP = gradQ[vidP, _U, 3, eP]
                    VxP = gradQ[vidP, _V, 1, eP]
                    VyP = gradQ[vidP, _V, 2, eP]
                    VzP = gradQ[vidP, _V, 3, eP]
                    WxP = gradQ[vidP, _W, 1, eP]
                    WyP = gradQ[vidP, _W, 2, eP]
                    WzP = gradQ[vidP, _W, 3, eP]
                    ExP = gradQ[vidP, _E, 1, eP]
                    EyP = gradQ[vidP, _E, 2, eP]
                    EzP = gradQ[vidP, _E, 3, eP]
                    
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    WP = Q[vidP, _W, eP]
                    uP=UP/ρP
                    vP=VP/ρP
                    wP=WP/ρP

                    uxP, uyP, uzP = UxP, UyP, UzP
                    vxP, vyP, vzP = VxP, VyP, VzP
                    wxP, wyP, wzP = WxP, WyP, WzP
                    TxP, TyP, TzP = ExP, EyP, EzP
                    
                    # For each tracer
                    for itracer = 1:_ntracers
                        istate = itracer + (_nsd + 2) 
                        
                        QTxP[itracer] = gradQ[vidP, istate, 1, eP]
                        QTyP[itracer] = gradQ[vidP, istate, 2, eP]
                        QTzP[itracer] = gradQ[vidP, istate, 3, eP]
                        
                        QTP[itracer]  = Q[vidP, istate, eP]
                    
                    end
                    
                elseif bc == 1
                    
                    ρnM = nxM * ρxM + nyM * ρyM + nzM * ρzM
                    ρxP = ρxM - 2 * ρnM * nxM
                    ρyP = ρyM - 2 * ρnM * nyM
                    ρzP = ρzM - 2 * ρnM * nzM
                    
                    UnM = nxM * UxM + nyM * UyM + nzM * UzM
                    UxP = UxM - 2 * UnM * nxM
                    UyP = UyM - 2 * UnM * nyM
                    UzP = UzM - 2 * UnM * nzM
                    
                    VnM = nxM * VxM + nyM * VyM + nzM * VzM
                    VxP = VxM - 2 * VnM * nxM
                    VyP = VyM - 2 * VnM * nyM
                    VzP = VzM - 2 * VnM * nzM
                    
                    WnM = nxM * WxM + nyM * WyM + nzM * WzM
                    WxP = WxM - 2 * WnM * nxM
                    WyP = WyM - 2 * WnM * nyM
                    WzP = WzM - 2 * WnM * nzM
                    
                    EnM = nxM * ExM + nyM * EyM + nzM * EzM
                    ExP = ExM - 2 * EnM * nxM
                    EyP = EyM - 2 * EnM * nyM
                    EzP = EzM - 2 * EnM * nzM

                    unM = nxM * uM + nyM * vM + nzM * wM
                    uP = uM - 2 * unM * nxM
                    vP = vM - 2 * unM * nyM
                    wP = wM - 2 * unM * nzM
                    uxP, uyP, uzP = UxP, UyP, UzP #FXG: Not sure about this BC
                    vxP, vyP, vzP = VxP, VyP, VzP #FXG: Not sure about this BC
                    wxP, wyP, wzP = WxP, WyP, WzP #FXG: Not sure about this BC
                    #TxP, TyP, TzP = ExP, EyP, EzP #Produces thermal boundary layer
                    TxP, TyP, TzP = TxM, TyM, TzM
                    
                    # Tracers   
                    for itracer = 1:_ntracers
                        istate          = itracer + (_nsd + 2) 
                        
                        QTnM            = nxM * QTxM[itracer] + nyM * QTyM[itracer] + nzM * QTzM[itracer]
                        QTxP[itracer]   = QTxM[itracer] - 2 * QTnM * nxM 
                        QTyP[itracer]   = QTyM[itracer] - 2 * QTnM * nyM 
                        QTzP[itracer]   = QTzM[itracer] - 2 * QTnM * nzM 
                    end
                
                else
                    error("Invalid boundary conditions $bc on face $f of element $e")
                end

                #Left Fluxes
                div_uM   = uxM + vyM + wzM
                fluxρM_x = 0*ρxM
                fluxρM_y = 0*ρyM
                fluxρM_z = 0*ρzM
                fluxUM_x = 2*uxM + lambda*div_uM
                fluxUM_y = uyM + vxM
                fluxUM_z = uzM + wxM
                fluxVM_x = vxM + uyM
                fluxVM_y = 2*vyM + lambda*div_uM
                fluxVM_z = vzM + wyM
                fluxWM_x = wxM + uzM
                fluxWM_y = wyM + vzM
                fluxWM_z = 2*wzM + lambda*div_uM
                fluxEM_x = uM*(2*uxM + lambda*div_uM) + vM*(uyM + vxM) + wM*(uzM + wxM) + c_p/Pr*TxM
                fluxEM_y = uM*(vxM + uyM) + vM*(2*vyM + lambda*div_uM) + wM*(vzM + wyM) + c_p/Pr*TyM
                fluxEM_z = uM*(wxM + uzM) + vM*(wyM + vzM) + wM*(2*wzM + lambda*div_uM) + c_p/Pr*TzM

                #Right Fluxes
                div_uP   = uxP + vyP + wzP
                fluxρP_x = 0*ρxP
                fluxρP_y = 0*ρyP
                fluxρP_z = 0*ρzP
                fluxUP_x = 2*uxP + lambda*div_uP
                fluxUP_y = uyP + vxP
                fluxUP_z = uzP + wxP
                fluxVP_x = vxP + uyP
                fluxVP_y = 2*vyP + lambda*div_uP
                fluxVP_z = vzP + wyP
                fluxWP_x = wxP + uzP
                fluxWP_y = wyP + vzP
                fluxWP_z = 2*wzP + lambda*div_uP
                fluxEP_x = uP*(2*uxP + lambda*div_uP) + vP*(uyP + vxP) + wP*(uzP + wxP) + c_p/Pr*TxP
                fluxEP_y = uP*(vxP + uyP) + vP*(2*vyP + lambda*div_uP) + wP*(vzP + wyP) + c_p/Pr*TyP
                fluxEP_z = uP*(wxP + uzP) + vP*(wyP + vzP) + wP*(2*wzP + lambda*div_uP) + c_p/Pr*TzP

                #Compute Numerical Flux
                fluxρS = 0.5*(nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) + nzM * (fluxρM_z + fluxρP_z))
                fluxUS = 0.5*(nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) + nzM * (fluxUM_z + fluxUP_z))
                fluxVS = 0.5*(nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) + nzM * (fluxVM_z + fluxVP_z))
                fluxWS = 0.5*(nxM * (fluxWM_x + fluxWP_x) + nyM * (fluxWM_y + fluxWP_y) + nzM * (fluxWM_z + fluxWP_z))
                fluxES = 0.5*(nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) + nzM * (fluxEM_z + fluxEP_z))

                #Update RHS
                rhs[vidM, _ρ, 1, eM] += sMJ * fluxρS
                rhs[vidM, _U, 1, eM] += sMJ * fluxUS
                rhs[vidM, _V, 1, eM] += sMJ * fluxVS
                rhs[vidM, _W, 1, eM] += sMJ * fluxWS
                rhs[vidM, _E, 1, eM] += sMJ * fluxES
                
                for itracer = 1:_ntracers
                    istate = itracer + (_nsd + 2) 
                    
                    # Left, Right fluxes
                    fluxQTM_x = QTxM[itracer] 
                    fluxQTM_y = QTyM[itracer] 
                    fluxQTM_z = QTzM[itracer]
                    fluxQTP_x = QTxP[itracer] 
                    fluxQTP_y = QTyP[itracer] 
                    fluxQTP_z = QTzP[itracer]
                    # Compute numerical flux
                    fluxQTS = 0.5*(nxM * (fluxQTM_x + fluxQTP_x) + nyM * (fluxQTM_y + fluxQTP_y) + nzM * (fluxQTM_z + fluxQTP_z))    
                    # Update RHS
                    rhs[vidM, istate, 1 , eM] += sMJ * fluxQTS           
                end
            end
        end
    end
end
# }}}

# {{{ Update grad Q solution
function update_gradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        Q[i, s, 1, e] = rhs[i, s, 1, e] * vgeo[i, _MJI, e]
        Q[i, s, 2, e] = rhs[i, s, 2, e] * vgeo[i, _MJI, e]
        Q[i, s, 3, e] = rhs[i, s, 3, e] * vgeo[i, _MJI, e]
    end

end
# }}}

# {{{ Update grad Q solution
function update_divgradQ!(::Val{dim}, ::Val{N}, Q, rhs, vgeo, elems) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        Q[i, s, e] = rhs[i, s, 1, e] * vgeo[i, _MJI, e]
    end

end
# }}}

# {{{ Update solution (for all dimensions)
function updatesolution!(::Val{dim}, ::Val{N}, rhs::Array, rhs_gradQ, Q, vgeo, elems, rka,
                         rkb, dt, visc) where {dim, N}

    Nq=(N+1)^dim

    @inbounds for e = elems, s = 1:_nstate, i = 1:Nq
        rhs[i, s, e] += visc*rhs_gradQ[i,s,1,e]
        Q[i, s, e] += rkb * dt * rhs[i, s, e] * vgeo[i, _MJI, e]
        rhs[i, s, e] *= rka
    end
end

# }}}


# {{{ improved GPU kernles

# {{{ Volume RHS for 3D
@hascuda function knl_volume_rhs!(::Val{3}, ::Val{N}, rhs, Q, vgeo, D, nelem) where N
    DFloat = eltype(D)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Nq = N + 1

    (i, j, k) = threadIdx()
    e = blockIdx().x

    s_D = @cuStaticSharedMem(eltype(D), (Nq, Nq))
    s_F = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
    s_G = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))
    s_H = @cuStaticSharedMem(eltype(Q), (Nq, Nq, Nq, _nstate))

    rhsU = rhsV = rhsW = rhsρ = rhsE = zero(eltype(rhs))
    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
        # Load derivative into shared memory
        if k == 1
            s_D[i, j] = D[i, j]
        end

        # Load values will need into registers
        MJ = vgeo[i, j, k, _MJ, e]
        ξx, ξy, ξz = vgeo[i,j,k,_ξx,e], vgeo[i,j,k,_ξy,e], vgeo[i,j,k,_ξz,e]
        ηx, ηy, ηz = vgeo[i,j,k,_ηx,e], vgeo[i,j,k,_ηy,e], vgeo[i,j,k,_ηz,e]
        ζx, ζy, ζz = vgeo[i,j,k,_ζx,e], vgeo[i,j,k,_ζy,e], vgeo[i,j,k,_ζz,e]

        U, V, W = Q[i, j, k, _U, e], Q[i, j, k, _V, e], Q[i, j, k, _W, e]
        ρ, E = Q[i, j, k, _ρ, e], Q[i, j, k, _E, e]

        P = p0 * CUDAnative.pow(R_gas * E / p0, c_p / c_v)

        ρinv = 1 / ρ
        fluxρ_x = U
        fluxU_x = ρinv * U * U + P
        fluxV_x = ρinv * V * U
        fluxW_x = ρinv * W * U
        fluxE_x = E * ρinv * U

        fluxρ_y = V
        fluxU_y = ρinv * U * V
        fluxV_y = ρinv * V * V + P
        fluxW_y = ρinv * W * V
        fluxE_y = E * ρinv * V

        fluxρ_z = W
        fluxU_z = ρinv * U * W
        fluxV_z = ρinv * V * W
        fluxW_z = ρinv * W * W + P
        fluxE_z = E * ρinv * W

        s_F[i, j, k, _ρ] = MJ * (ξx * fluxρ_x + ξy * fluxρ_y + ξz * fluxρ_z)
        s_F[i, j, k, _U] = MJ * (ξx * fluxU_x + ξy * fluxU_y + ξz * fluxU_z)
        s_F[i, j, k, _V] = MJ * (ξx * fluxV_x + ξy * fluxV_y + ξz * fluxV_z)
        s_F[i, j, k, _W] = MJ * (ξx * fluxW_x + ξy * fluxW_y + ξz * fluxW_z)
        s_F[i, j, k, _E] = MJ * (ξx * fluxE_x + ξy * fluxE_y + ξz * fluxE_z)

        s_G[i, j, k, _ρ] = MJ * (ηx * fluxρ_x + ηy * fluxρ_y + ηz * fluxρ_z)
        s_G[i, j, k, _U] = MJ * (ηx * fluxU_x + ηy * fluxU_y + ηz * fluxU_z)
        s_G[i, j, k, _V] = MJ * (ηx * fluxV_x + ηy * fluxV_y + ηz * fluxV_z)
        s_G[i, j, k, _W] = MJ * (ηx * fluxW_x + ηy * fluxW_y + ηz * fluxW_z)
        s_G[i, j, k, _E] = MJ * (ηx * fluxE_x + ηy * fluxE_y + ηz * fluxE_z)

        s_H[i, j, k, _ρ] = MJ * (ζx * fluxρ_x + ζy * fluxρ_y + ζz * fluxρ_z)
        s_H[i, j, k, _U] = MJ * (ζx * fluxU_x + ζy * fluxU_y + ζz * fluxU_z)
        s_H[i, j, k, _V] = MJ * (ζx * fluxV_x + ζy * fluxV_y + ζz * fluxV_z)
        s_H[i, j, k, _W] = MJ * (ζx * fluxW_x + ζy * fluxW_y + ζz * fluxW_z)
        s_H[i, j, k, _E] = MJ * (ζx * fluxE_x + ζy * fluxE_y + ζz * fluxE_z)

        rhsU, rhsV, rhsW = (rhs[i, j, k, _U, e],
                            rhs[i, j, k, _V, e],
                            rhs[i, j, k, _W, e])
        rhsρ, rhsE = rhs[i, j, k, _ρ, e], rhs[i, j, k, _E, e]

        # buoyancy term
        rhsW -= MJ * ρ * gravity
    end

    sync_threads()

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
        # loop of ξ-grid lines
        for n = 1:Nq
            Dni = s_D[n, i]
            Dnj = s_D[n, j]
            Dnk = s_D[n, k]

            rhsρ += Dni * s_F[n, j, k, _ρ]
            rhsρ += Dnj * s_G[i, n, k, _ρ]
            rhsρ += Dnk * s_H[i, j, n, _ρ]

            rhsU += Dni * s_F[n, j, k, _U]
            rhsU += Dnj * s_G[i, n, k, _U]
            rhsU += Dnk * s_H[i, j, n, _U]

            rhsV += Dni * s_F[n, j, k, _V]
            rhsV += Dnj * s_G[i, n, k, _V]
            rhsV += Dnk * s_H[i, j, n, _V]

            rhsW += Dni * s_F[n, j, k, _W]
            rhsW += Dnj * s_G[i, n, k, _W]
            rhsW += Dnk * s_H[i, j, n, _W]

            rhsE += Dni * s_F[n, j, k, _E]
            rhsE += Dnj * s_G[i, n, k, _E]
            rhsE += Dnk * s_H[i, j, n, _E]
        end

        rhs[i, j, k, _U, e] = rhsU
        rhs[i, j, k, _V, e] = rhsV
        rhs[i, j, k, _W, e] = rhsW
        rhs[i, j, k, _ρ, e] = rhsρ
        rhs[i, j, k, _E, e] = rhsE
    end
    nothing
end
# }}}

# {{{ Face RHS (all dimensions)
@hascuda function knl_flux_rhs!(::Val{dim}, ::Val{N}, rhs, Q, sgeo, vgeo, nelem, vmapM,
                               vmapP, elemtobndy) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1) * (N+1) * (N+1)
    nface = 6

    (i, j, k) = threadIdx()
    e = blockIdx().x

    Nq = N+1
    half = convert(eltype(Q), 0.5)

    @inbounds if i <= Nq && j <= Nq && k == 1 && e <= nelem
        n = i + (j-1) * Nq
        for lf = 1:2:nface
            for f = lf:lf+1
                (nxM, nyM) = (sgeo[_nx, n, f, e], sgeo[_ny, n, f, e])
                (nzM, sMJ) = (sgeo[_nz, n, f, e], sgeo[_sMJ, n, f, e])

                (idM, idP) = (vmapM[n, f, e], vmapP[n, f, e])

                (eM, eP) = (e, ((idP - 1) ÷ Np) + 1)
                (vidM, vidP) = (((idM - 1) % Np) + 1,  ((idP - 1) % Np) + 1)

                ρM = Q[vidM, _ρ, eM]
                UM = Q[vidM, _U, eM]
                VM = Q[vidM, _V, eM]
                WM = Q[vidM, _W, eM]
                EM = Q[vidM, _E, eM]

                bc = elemtobndy[f, e]
                PM = p0 * CUDAnative.pow(R_gas * EM / p0, c_p / c_v)
                ρP = UP = VP = WP = EP = PP = zero(eltype(Q))
                if bc == 0
                    ρP = Q[vidP, _ρ, eP]
                    UP = Q[vidP, _U, eP]
                    VP = Q[vidP, _V, eP]
                    WP = Q[vidP, _W, eP]
                    EP = Q[vidP, _E, eP]
                    PP = p0 * CUDAnative.pow(R_gas * EP / p0, c_p / c_v)
                elseif bc == 1
                    UnM = nxM * UM + nyM * VM + nzM * WM
                    UP = UM - 2 * UnM * nxM
                    VP = VM - 2 * UnM * nyM
                    WP = WM - 2 * UnM * nzM
                    ρP = ρM
                    EP = EM
                    PP = PM
                end

                ρMinv = 1 / ρM
                fluxρM_x = UM
                fluxUM_x = ρMinv * UM * UM + PM
                fluxVM_x = ρMinv * VM * UM
                fluxWM_x = ρMinv * WM * UM
                fluxEM_x = ρMinv * UM * EM

                fluxρM_y = VM
                fluxUM_y = ρMinv * UM * VM
                fluxVM_y = ρMinv * VM * VM + PM
                fluxWM_y = ρMinv * WM * VM
                fluxEM_y = ρMinv * VM * EM

                fluxρM_z = WM
                fluxUM_z = ρMinv * UM * WM
                fluxVM_z = ρMinv * VM * WM
                fluxWM_z = ρMinv * WM * WM + PM
                fluxEM_z = ρMinv * WM * EM

                ρPinv = 1 / ρP
                fluxρP_x = UP
                fluxUP_x = ρPinv * UP * UP + PP
                fluxVP_x = ρPinv * VP * UP
                fluxWP_x = ρPinv * WP * UP
                fluxEP_x = ρPinv * UP * EP

                fluxρP_y = VP
                fluxUP_y = ρPinv * UP * VP
                fluxVP_y = ρPinv * VP * VP + PP
                fluxWP_y = ρPinv * WP * VP
                fluxEP_y = ρPinv * VP * EP

                fluxρP_z = WP
                fluxUP_z = ρPinv * UP * WP
                fluxVP_z = ρPinv * VP * WP
                fluxWP_z = ρPinv * WP * WP + PP
                fluxEP_z = ρPinv * WP * EP

                λM = ρMinv * abs(nxM * UM + nyM * VM + nzM * WM) + CUDAnative.sqrt(ρMinv * γ * PM)
                λP = ρPinv * abs(nxM * UP + nyM * VP + nzM * WP) + CUDAnative.sqrt(ρPinv * γ * PP)
                λ  =  max(λM, λP)

                #Compute Numerical Flux and Update
                fluxρS = (nxM * (fluxρM_x + fluxρP_x) + nyM * (fluxρM_y + fluxρP_y) +
                          nzM * (fluxρM_z + fluxρP_z) - λ * (ρP - ρM)) / 2
                fluxUS = (nxM * (fluxUM_x + fluxUP_x) + nyM * (fluxUM_y + fluxUP_y) +
                          nzM * (fluxUM_z + fluxUP_z) - λ * (UP - UM)) / 2
                fluxVS = (nxM * (fluxVM_x + fluxVP_x) + nyM * (fluxVM_y + fluxVP_y) +
                          nzM * (fluxVM_z + fluxVP_z) - λ * (VP - VM)) / 2
                fluxWS = (nxM * (fluxWM_x + fluxWP_x) + nyM * (fluxWM_y + fluxWP_y) +
                          nzM * (fluxWM_z + fluxWP_z) - λ * (WP - WM)) / 2
                fluxES = (nxM * (fluxEM_x + fluxEP_x) + nyM * (fluxEM_y + fluxEP_y) +
                          nzM * (fluxEM_z + fluxEP_z) - λ * (EP - EM)) / 2

                #Update RHS
                rhs[vidM, _ρ, eM] -= sMJ * fluxρS
                rhs[vidM, _U, eM] -= sMJ * fluxUS
                rhs[vidM, _V, eM] -= sMJ * fluxVS
                rhs[vidM, _W, eM] -= sMJ * fluxWS
                rhs[vidM, _E, eM] -= sMJ * fluxES
            end
            sync_threads()
        end
    end
nothing
end
# }}}

# {{{ Update solution (for all dimensions)
@hascuda function knl_updatesolution!(::Val{dim}, ::Val{N}, rhs, Q, vgeo, nelem, rka,
                                      rkb, dt) where {dim, N}
    (i, j, k) = threadIdx()
    e = blockIdx().x

    Nq = N+1
    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
        n = i + (j-1) * Nq + (k-1) * Nq * Nq
        MJI = vgeo[n, _MJI, e]
        for s = 1:_nstate
            Q[n, s, e] += rkb * dt * rhs[n, s, e] * MJI
            rhs[n, s, e] *= rka
        end
    end
    nothing
end
# }}}

# }}}

# {{{ Fill sendQ on device with Q (for all dimensions)
@hascuda function knl_fillsendQ!(::Val{dim}, ::Val{N}, sendQ, Q,
                                 sendelems) where {N, dim}
    Nq = N + 1
    (i, j, k) = threadIdx()
    e = blockIdx().x

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= length(sendelems)
        n = i + (j-1) * Nq + (k-1) * Nq * Nq
        re = sendelems[e]
        for s = 1:_nstate
            sendQ[n, s, e] = Q[n, s, re]
        end
    end
    nothing
end
# }}}

# {{{ Fill Q on device with recvQ (for all dimensions)
@hascuda function knl_transferrecvQ!(::Val{dim}, ::Val{N}, Q, recvQ, nelem,
                                     nrealelem) where {N, dim}
    Nq = N + 1
    (i, j, k) = threadIdx()
    e = blockIdx().x

    @inbounds if i <= Nq && j <= Nq && k <= Nq && e <= nelem
        n = i + (j-1) * Nq + (k-1) * Nq * Nq
        for s = 1:_nstate
            Q[n, s, nrealelem + e] = recvQ[n, s, e]
        end
    end
    nothing
end
# }}}

# {{{ MPI Buffer handling
function fillsendQ!(::Val{dim}, ::Val{N}, sendQ, d_sendQ::Array, Q,
                    sendelems) where {dim, N}
    sendQ[:, :, :] .= Q[:, :, sendelems]
end

@hascuda function fillsendQ!(::Val{dim}, ::Val{N}, sendQ, d_sendQ::CuArray,
                             d_QL, d_sendelems) where {dim, N}
    nsendelem = length(d_sendelems)
    if nsendelem > 0
        @cuda(threads=ntuple(j->N+1, dim), blocks=nsendelem,
              knl_fillsendQ!(Val(dim), Val(N), d_sendQ, d_QL, d_sendelems))
        sendQ .= d_sendQ
    end
end

@hascuda function transferrecvQ!(::Val{dim}, ::Val{N}, d_recvQ::CuArray, recvQ,
                                 d_QL, nrealelem) where {dim, N}
    nrecvelem = size(recvQ)[end]
    if nrecvelem > 0
        d_recvQ .= recvQ
        @cuda(threads=ntuple(j->N+1, dim), blocks=nrecvelem,
              knl_transferrecvQ!(Val(dim), Val(N), d_QL, d_recvQ, nrecvelem,
                                 nrealelem))
    end
end

function transferrecvQ!(::Val{dim}, ::Val{N}, d_recvQ::Array, recvQ, Q,
                        nrealelem) where {dim, N}
    Q[:, :, nrealelem+1:end] .= recvQ[:, :, :]
end
# }}}

# {{{ GPU kernel wrappers
@hascuda function volume_rhs!(::Val{dim}, ::Val{N}, d_rhsC::CuArray, d_QC,
                             d_vgeoC, d_D, elems) where {dim, N}
    nelem = length(elems)
    @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
          knl_volume_rhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, nelem))
end

@hascuda function flux_rhs!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL, d_sgeo,
                           d_vgeoL, elems, d_vmapM, d_vmapP, d_elemtobndy) where {dim, N}
    nelem = length(elems)
    @cuda(threads=(ntuple(j->N+1, dim-1)..., 1), blocks=nelem,
          knl_flux_rhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, d_vgeoL, nelem, d_vmapM,
                       d_vmapP, d_elemtobndy))
end

@hascuda function updatesolution!(::Val{dim}, ::Val{N}, d_rhsL::CuArray, d_QL,
                                  d_vgeoL, elems, rka, rkb, dt) where {dim, N}
    nelem = length(elems)
    @cuda(threads=ntuple(j->N+1, dim), blocks=nelem,
          knl_updatesolution!(Val(dim), Val(N), d_rhsL, d_QL, d_vgeoL, nelem, rka,
                              rkb, dt))
end
# }}}

# {{{ L2 Energy (for all dimensions)
function L2energysquared(::Val{dim}, ::Val{N}, Q, vgeo, elems) where {dim, N}
    DFloat = eltype(Q)
    Np = (N+1)^dim
    (~, nstate, nelem) = size(Q)

    energy = zero(DFloat)

    @inbounds for e = elems, q = 1:nstate, i = 1:Np
        energy += vgeo[i, _MJ, e] * Q[i, q, e]^2
    end

    energy
end
# }}}

# {{{ Send Data Q
function senddata_Q(::Val{dim}, ::Val{N}, mesh, sendreq, recvreq, sendQ,
                  recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                  ArrType=ArrType) where {dim, N}
    mpirank = MPI.Comm_rank(mpicomm)

    # Create send and recv request array
    nnabr = length(mesh.nabrtorank)
    d_sendelems = ArrType(mesh.sendelems)
    nrealelem = length(mesh.realelems)

    # post MPI receives
    for n = 1:nnabr
        recvreq[n] = MPI.Irecv!((@view recvQ[:, :, mesh.nabrtorecv[n]]),
                                mesh.nabrtorank[n], 777, mpicomm)
    end

    # wait on (prior) MPI sends
    MPI.Waitall!(sendreq)

    # pack data from d_QL into send buffer
#    fillsendQ!(Val(dim), Val(N), sendQ, d_sendQ, d_QL, d_sendelems)
    sendQ[:, :, :] .= d_QL[:, :, d_sendelems]

    # post MPI sends
    for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
    end
end
# }}}

# {{{ Send Data Grad(Q)
function senddata_gradQ(::Val{dim}, ::Val{N}, mesh, sendreq, recvreq, sendQ,
                  recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                  ArrType=ArrType) where {dim, N}
    mpirank = MPI.Comm_rank(mpicomm)

    # Create send and recv request array
    nnabr = length(mesh.nabrtorank)
    d_sendelems = ArrType(mesh.sendelems)
    nrealelem = length(mesh.realelems)

    # post MPI receives
    for n = 1:nnabr
        recvreq[n] = MPI.Irecv!((@view recvQ[:, :, :, mesh.nabrtorecv[n]]),
                                mesh.nabrtorank[n], 777, mpicomm)
    end

    # wait on (prior) MPI sends
    MPI.Waitall!(sendreq)

    # pack data from d_QL into send buffer
#    fillsendQ!(Val(dim), Val(N), sendQ, d_sendQ, d_QL, d_sendelems)
    sendQ[:, :, :, :] .= d_QL[:, :, :, d_sendelems]

    # post MPI sends
    for n = 1:nnabr
        sendreq[n] = MPI.Isend((@view sendQ[:, :, :, mesh.nabrtosend[n]]),
                               mesh.nabrtorank[n], 777, mpicomm)
    end
end
# }}}

# {{{ Receive Data Q
function receivedata_Q!(::Val{dim}, ::Val{N}, mesh, recvreq, recvQ,
                        d_recvQ, d_QL) where {dim, N}
    nrealelem = length(mesh.realelems)

    # wait on MPI receives
    MPI.Waitall!(recvreq)

    # copy data to state vector d_QL
    #transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)
    d_QL[:, :, nrealelem+1:end] .= recvQ[:, :, :]

end
# }}}

# {{{ Receive Data Grad(Q)
function receivedata_gradQ!(::Val{dim}, ::Val{N}, mesh, recvreq, recvQ,
                            d_recvQ, d_QL) where {dim, N}
    nrealelem = length(mesh.realelems)

    # wait on MPI receives
    MPI.Waitall!(recvreq)

    # copy data to state vector d_QL
    #transferrecvQ!(Val(dim), Val(N), d_recvQ, recvQ, d_QL, nrealelem)
    d_QL[:, :, :, nrealelem+1:end] .= recvQ[:, :, :, :]
end
# }}}

# {{{ RK loop
function lowstorageRK(::Val{dim}, ::Val{N}, mesh, vgeo, sgeo, Q, rhs, D,
                      dt, nsteps, tout, vmapM, vmapP, mpicomm, iplot, visc;
                      ArrType=ArrType, plotstep=0) where {dim, N}
    DFloat = eltype(Q)
    mpirank = MPI.Comm_rank(mpicomm)

    # Fourth-order, low-storage, Runge–Kutta scheme of Carpenter and Kennedy
    # (1994) ((5,4) 2N-Storage RK scheme.
    #
    # Ref:
    # @TECHREPORT{CarpenterKennedy1994,
    #   author = {M.~H. Carpenter and C.~A. Kennedy},
    #   title = {Fourth-order {2N-storage} {Runge-Kutta} schemes},
    #   institution = {National Aeronautics and Space Administration},
    #   year = {1994},
    #   number = {NASA TM-109112},
    #   address = {Langley Research Center, Hampton, VA},
    # }
    RKA = (DFloat(0),
           DFloat(-567301805773)  / DFloat(1357537059087),
           DFloat(-2404267990393) / DFloat(2016746695238),
           DFloat(-3550918686646) / DFloat(2091501179385),
           DFloat(-1275806237668) / DFloat(842570457699 ))

    RKB = (DFloat(1432997174477) / DFloat(9575080441755 ),
           DFloat(5161836677717) / DFloat(13612068292357),
           DFloat(1720146321549) / DFloat(2090206949498 ),
           DFloat(3134564353537) / DFloat(4481467310338 ),
           DFloat(2277821191437) / DFloat(14882151754819))

    RKC = (DFloat(0),
           DFloat(1432997174477) / DFloat(9575080441755),
           DFloat(2526269341429) / DFloat(6820363962896),
           DFloat(2006345519317) / DFloat(3224310063776),
           DFloat(2802321613138) / DFloat(2924317926251))

    # Create send and recv request array
    nnabr = length(mesh.nabrtorank)
    sendreq = fill(MPI.REQUEST_NULL, nnabr)
    recvreq = fill(MPI.REQUEST_NULL, nnabr)
    @show(size(sendreq),size(recvreq))
    # Create send and recv buffer
    sendQ = zeros(DFloat, (N+1)^dim, size(Q,2), length(mesh.sendelems))
    recvQ = zeros(DFloat, (N+1)^dim, size(Q,2), length(mesh.ghostelems))
    @show(size(sendQ),size(recvQ))
    # Create send and recv LDG buffer
    sendgradQ = zeros(DFloat, (N+1)^dim, size(Q,2), dim, length(mesh.sendelems))
    recvgradQ = zeros(DFloat, (N+1)^dim, size(Q,2), dim, length(mesh.ghostelems))
    @show(size(sendgradQ),size(recvgradQ))
    # Store Constants
    nrealelem = length(mesh.realelems)
    nsendelem = length(mesh.sendelems)
    nrecvelem = length(mesh.ghostelems)
    nelem = length(mesh.elems)

    #Create Device Arrays
    d_QL, d_rhsL                = ArrType(Q), ArrType(rhs)
    d_vgeoL, d_sgeo             = ArrType(vgeo), ArrType(sgeo)
    d_vmapM, d_vmapP            = ArrType(vmapM), ArrType(vmapP)
    d_sendelems, d_elemtobndy   = ArrType(mesh.sendelems), ArrType(mesh.elemtobndy)
    d_sendQ, d_recvQ            = ArrType(sendQ), ArrType(recvQ)
    d_D                         = ArrType(D)
    
    #Create Device LDG Arrays
    d_gradQL                    = zeros(DFloat, (N+1)^dim, _nstate+3, dim, nelem)
    d_rhs_gradQL                = zeros(DFloat, (N+1)^dim, _nstate, dim, nelem)
    d_sendgradQ, d_recvgradQ    = ArrType(sendgradQ), ArrType(recvgradQ)
    
    #Create Device SGS Arrays
    #=
    d_rhs_sgs   = zeros(DFloat, (N+1)^dim, _nstate, nelem)
    d_visc_sgs  = zeros(DFloat, 3, nelem)
    visc_sgsL   = zeros(DFloat, (N+1)^dim, 3, nelem)
    =#

    #Template Reshape Arrays
    Qshape      = (fill(N+1, dim)..., size(Q, 2), size(Q, 3))
    vgeoshape   = (fill(N+1, dim)..., _nvgeo, size(Q, 3))
    gradQshape  = (fill(N+1, dim)..., size(d_gradQL,2), size(d_gradQL,3), size(d_gradQL,4))
    
    #Reshape Device Arrays 
    d_QC = reshape(d_QL, Qshape)
    d_rhsC = reshape(d_rhsL, Qshape...)
    d_vgeoC = reshape(d_vgeoL, vgeoshape)

    #Reshape Device LDG Arrays
    d_gradQC = reshape(d_gradQL, gradQshape)
    d_rhs_gradQC = reshape(d_rhs_gradQL, gradQshape...)
    
    #Start Time Loop
    start_time = t1 = time_ns()
    for step = 1:nsteps
        for s = 1:length(RKA)

            #---------------1st Order Operators--------------------------#
            # Send Data Q
            senddata_Q(Val(dim), Val(N), mesh, sendreq, recvreq, sendQ,
                       recvQ, d_sendelems, d_sendQ, d_recvQ, d_QL, mpicomm;
                       ArrType=ArrType)

            # volume RHS computation
            volume_rhs!(Val(dim), Val(N), d_rhsC, d_QC, d_vgeoC, d_D, mesh.realelems)

            # Receive Data Q
            receivedata_Q!(Val(dim), Val(N), mesh, recvreq, recvQ, d_recvQ, d_QL)

            # face RHS computation
            flux_rhs!(Val(dim), Val(N), d_rhsL, d_QL, d_sgeo, d_vgeoL, mesh.realelems, d_vmapM,
                     d_vmapP, d_elemtobndy)

            #---------------2nd Order Operators--------------------------#
            if (visc > 0)
                # volume grad Q computation
                volume_grad!(Val(dim), Val(N), d_rhs_gradQC, d_QC, d_vgeoC, d_D, mesh.realelems)

                # flux grad Q computation
                flux_grad!(Val(dim), Val(N), d_rhs_gradQL, d_QL, d_sgeo, d_vgeoL, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)

                # Construct grad Q
                update_gradQ!(Val(dim), Val(N), d_gradQL, d_rhs_gradQL, d_vgeoL, mesh.realelems)

                # Send Data grad(Q)
                senddata_gradQ(Val(dim), Val(N), mesh, sendreq, recvreq, sendgradQ,
                               recvgradQ, d_sendelems, d_sendgradQ, d_recvgradQ,
                               d_gradQL, mpicomm;ArrType=ArrType)

                # volume div(grad Q) computation
                volume_div!(Val(dim), Val(N), d_rhs_gradQC, d_gradQC, d_QC, d_vgeoC, d_D, mesh.realelems)

                # Receive Data grad(Q)
                receivedata_gradQ!(Val(dim), Val(N), mesh, recvreq, recvgradQ, d_recvgradQ, d_gradQL)

                # flux div(grad Q) computation
                flux_div!(Val(dim), Val(N), d_rhs_gradQL, d_gradQL, d_QL, d_sgeo, mesh.realelems, d_vmapM, d_vmapP, d_elemtobndy)
            end

            #---------------Update Solution--------------------------#
            # update solution and scale RHS
            updatesolution!(Val(dim), Val(N), d_rhsL, d_rhs_gradQL, d_QL, d_vgeoL, mesh.realelems,
                            RKA[s%length(RKA)+1], RKB[s], dt, visc)
        end
        if step == 1
            @hascuda synchronize()
            start_time = time_ns()
        end
        if mpirank == 0 && (time_ns() - t1)*1e-9 > tout
            @hascuda synchronize()
            t1 = time_ns()
            avg_stage_time = (time_ns() - start_time) * 1e-9 / ((step-1) * length(RKA))
            @show (step, nsteps, avg_stage_time)
        end

        # Write VTK file
        if mod(step,iplot) == 0
            Q .= d_QL
            convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q)
            X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)..., nelem), dim)
            ρ = reshape((@view Q[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
            U = reshape((@view Q[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
            V = reshape((@view Q[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
            W = reshape((@view Q[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
            E = reshape((@view Q[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
            E = E .- 300.0
            #E_ref = reshape((@view Q[:, _nstate, :]), ntuple(j->(N+1),dim)..., nelem)
            #E = E .- E_ref
            writemesh(@sprintf("viz/nse%dD_set3c_%s_rank_%04d_step_%05d",dim, ArrType, mpirank, step), X...;
                      fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)), realelems=mesh.realelems)
        
            @inbounds for itracer = 1:_ntracers

                istate = itracer + _nsd + 2
                Qtracers = reshape((@view Q[:, istate, :]), ntuple(j->(N+1),dim)..., nelem)
                writemesh(@sprintf("viz/tracer_%04d_nse3d-tracer_%s_rank_%04d_step_%05d",
                                   itracer, ArrType, mpirank, step), X...;
                          fields = (("Rho", ρ), ("U", U), ("V", V), ("E", E), ("QT",Qtracers)),
                          realelems = mesh.realelems)
            end
        end
       #= if mpirank == 0
            avg_stage_time = (time_ns() - start_time) * 1e-9 / ((nsteps-1) * length(RKA))
            @show (nsteps, avg_stage_time)
        end =#
    Q .= d_QL
    rhs .= d_rhsL
    end
end

# }}}

# {{{ convert_variables
function convert_set2nc_to_set2c(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)

    println("[CPU] converting variables (CPU)...")
    @inbounds for e = 1:nelem, n = 1:Np
        ρ, u, v, w, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        Q[n, _U, e] = ρ*u
        Q[n, _V, e] = ρ*v
        Q[n, _W, e] = ρ*w
        Q[n, _E, e] = ρ*E
    end
end
# }}}

function convert_set2c_to_set2nc(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)

    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        u=U/ρ
        v=V/ρ
        w=W/ρ
        E=E/ρ
        Q[n, _U, e] = u
        Q[n, _V, e] = v
        Q[n, _W, e] = w
        Q[n, _E, e] = E
    end
end

function convert_set2nc_to_set3c(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)
    q_tr = zeros(DFloat, _ntracers)

    @inbounds for e = 1:nelem, n = 1:Np
        ρ, u, v, w, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        z = vgeo[n, _z, e]
       
        #Moist air constant: Rm

        for itracer = 1:_ntracers
            istate      = itracer + (_nsd + 2) 
            q_tr[itracer] = Q[n, istate, e]
        end 
        R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
        P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
        T = P/(ρ*R_gas)
        E = c_v*T + 0.5*(u^2 + v^2 + w^2) + gravity * z
        Q[n, _U, e] = ρ*u
        Q[n, _V, e] = ρ*v
        Q[n, _W, e] = ρ*w
        Q[n, _E, e] = ρ*E
    end
end

function convert_set3c_to_set2nc(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)

    #Allocate at least 3 spaces to q_tr
    q_tr = zeros(DFloat, _ntracers)

    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        z = vgeo[n, _z, e]

        # Moist air constant: Rm
        #=
        for itracer = 1:_ntracers
            istate = itracer + (_nsd+2)
            q_tr[itracer]   = Q[n, istate, e]
        end
        =#
        u, v, w, E = U/ρ, V/ρ, W/ρ, E/ρ
        R_gas = MoistThermodynamics.gas_constant_air(0.0, 0.0, 0.0)
        P = (R_gas/c_v) * ρ * (E - (u^2 + v^2 + w^2)/2 - gravity * z)
        θ = p0/(ρ * R_gas)*( P/p0 )^(c_v/c_p)
        E = p0/(ρ * R_gas) * ( P/p0)^(c_v/c_p)
        Q[n, _U, e] = u
        Q[n, _V, e] = v
        Q[n, _W, e] = w
        Q[n, _E, e] = E
    end
end
# }}}


function convert_set2nc_to_set4c(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)

    @inbounds for e = 1:nelem, n = 1:Np
        ρ, u, v, w, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
        T = P/(ρ*R_gas)
        E = c_v*T
        Q[n, _U, e] = ρ*u
        Q[n, _V, e] = ρ*v
        Q[n, _W, e] = ρ*w
        Q[n, _E, e] = ρ*E
    end
end

function convert_set4c_to_set2nc(::Val{dim}, ::Val{N}, vgeo, Q) where {dim, N}
    DFloat = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    Np = (N+1)^dim
    (~, ~, nelem) = size(Q)

    @inbounds for e = 1:nelem, n = 1:Np
        ρ, U, V, W, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        u=U/ρ
        v=V/ρ
        w=W/ρ
        T=E/(ρ*c_v)
        P=ρ*R_gas*T
        E=p0/(ρ* R_gas)*(P/p0)^(c_v/c_p)
        Q[n, _U, e] = u
        Q[n, _V, e] = v
        Q[n, _W, e] = w
        Q[n, _E, e] = E
    end
end


function convert_set2nc_to_set3c_scalar(x_ndim, Q)
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity

    q_tr = zeros(DFloat, _ntracers)

    ρ, u, v, w, E = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

    # Moist air constant: Rm
    # Get q from q*ρ
    q_tr[1] = 0.0
    q_tr[2] = 0.0
    q_tr[3] = 0.0
    for itracer = 1:_ntracers
        istate = itracer + (_nsd+2)
        q_tr[itracer] = Q[istate]
        Q[istate]     = q_tr[itracer] /ρ

    end
    R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])
    P = p0 * (ρ * R_gas * E / p0)^(c_p / c_v)
    T = P/(ρ*R_gas)

    E = c_v*T + 0.5*(u^2 + v^2 + w^2) + gravity * x[dim]

    Q[_U] = ρ*u
    Q[_V] = ρ*v
    Q[_W] = ρ*w
    Q[_E] = ρ*E

    return Q

end


function convert_set3c_to_set2nc_scalar(x_ndim, Q)
    DFloat          = eltype(Q)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    
    z               = x_ndim    
    q_tr            = zeros(DFloat, _ntracers)
    ρ, U, V, W, E      = Q[_ρ], Q[_U], Q[_V], Q[_W], Q[_E]

    q_tr  = zeros(DFloat, _ntracers)
    
    # Calculate air constant R_gas for moist air:
    q_tr[1] = 0.0
    q_tr[2] = 0.0
    q_tr[3] = 0.0
    for itracer = 1:_ntracers
        istate = itracer + (_nsd+2)
        q_tr[itracer] = Q[istate]
        Q[istate]     = q_tr[itracer] /ρ
        
    end    
    #R_gas = MoistThermodynamics.gas_constant_air(q_tr[1], q_tr[2], q_tr[3])
    
    u = U/ρ
    v = V/ρ
    E = E/ρ
    
    P = (R_gas/c_v)*ρ*(E - 0.5*(u^2 + v^2+ w^2) - gravity*z)
    #@show(P, E, y, y*gravity)
    θ = 300 #p0/(ρ * R_gas)*( P/p0 )^(c_v/c_p)
    
    Q[_U] = 0.0 #u
    Q[_V] = 0.0 #v
    Q[_W] = 0.0 #w
    Q[_E] = E #θ
    
end


function convert_velo_to_mome_ene_and_to_ENE(::Val{dim}, ::Val{N}, vgeo, Q) where {dim,N}
    DFloat      = eltype(Q)
    γ::DFloat   = _γ
    p0::DFloat  = _p0
    R_gas::DFloat= _R_gas
    c_p::DFloat = _c_p
    c_v::DFloat = _c_v
    gravity::DFloat= _gravity   

    Np = (N+1)^dim
    (~,~,nelem) = size(Q)

    @inbounds for e = 1:nelem, n = 1:Np
        ρ, u, v, w, E = Q[n, _ρ, e], Q[n, _U, e], Q[n, _V, e], Q[n, _W, e], Q[n, _E, e]
        Q[n, _U, e] = ρ * u
        Q[n, _V, e] = ρ * v
        Q[n, _W, e] = ρ * w
        Q[n, _E, e] = ρ * E
    end
end

# }}}

# {{{ nse driver
function nse(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend, iplot, visc;
               meshwarp=(x...)->identity(x),
               tout = 1, ArrType=Array, plotstep=0) where {dim, N}
    DFloat = typeof(tend)

    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)

    # Partion the mesh using a Hilbert curve based partitioning
    mpirank == 0 && println("[CPU] partitioning mesh...")
    mesh = partition(mpicomm, mesh...)

    # Connect the mesh in parallel
    mpirank == 0 && println("[CPU] connecting mesh...")
    mesh = connectmesh(mpicomm, mesh...)

    # Get the vmaps
    mpirank == 0 && println("[CPU] computing mappings...")
    (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface,
                              mesh.elemtoordr)

    # Create 1-D operators
    (ξ, ω) = lglpoints(DFloat, N)
    D = spectralderivative(ξ)

    # Compute the geometry
    mpirank == 0 && println("[CPU] computing metrics...")
    (vgeo, sgeo) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM)
    (nface, nelem) = size(mesh.elemtoelem)

    # Storage for the solution, rhs, and error
    mpirank == 0 && println("[CPU] creating fields (CPU)...")
    Q = zeros(DFloat, (N+1)^dim, _nstate+3, nelem)
    rhs = zeros(DFloat, (N+1)^dim, _nstate, nelem)
    
    # setup the initial condition
    mpirank == 0 && println("[CPU] computing initial conditions (CPU)...")
    @inbounds for e = 1:nelem, i = 1:(N+1)^dim
        x, y, z = vgeo[i, _x, e], vgeo[i, _y, e], vgeo[i, _z, e]
        Qinit = ic(x, y, z)
        Q[i, _U, e] = Qinit[1]
        Q[i, _V, e] = Qinit[2]
        Q[i, _W, e] = Qinit[3]
        Q[i, _ρ, e] = Qinit[4]
        Q[i, _E, e] = Qinit[5]
        
        #= Saturation adjustment (if applied) FIXME
        Q[i, _nstate + 1, e] = Qinit[_nstate + 1] #T 
        Q[i, _nstate + 2, e] = Qinit[_nstate + 2] #E_ref
        Q[i, _nstate + 3, e] = Qinit[_nstate + 3] #P
        =#
        @inbounds for istate = 6:_nstate
            Q[i,istate,e] = Qinit[istate]
        end

    end
    
    # Convert to proper variables
    mpirank == 0 && println("[CPU] converting variables (CPU)...")
    convert_set2nc_to_set3c(Val(dim), Val(N), vgeo, Q)

    # Compute time step
    mpirank == 0 && println("[CPU] computing dt (CPU)...")
    (base_dt, Courant) = courantnumber(Val(dim), Val(N), vgeo, Q, mpicomm)
    #base_dt=0.02
    mpirank == 0 && @show (base_dt, Courant)

    nsteps = ceil(Int64, tend / base_dt)
    dt = tend / nsteps
    mpirank == 0 && @show (dt, nsteps, dt * nsteps, tend)

    # Do time stepping
    stats = zeros(DFloat, 2)
    mpirank == 0 && println("[CPU] computing initial energy...")
    Q_temp=copy(Q)
    convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q_temp)
    stats[1] = L2energysquared(Val(dim), Val(N), Q_temp, vgeo, mesh.realelems)

    # Write VTK file: plot the initial condition
    mkpath("viz")
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    W = reshape((@view Q_temp[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E = E .- 300
    writemesh(@sprintf("viz/nse%dD_set3c_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, 0), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
              realelems=mesh.realelems)

    # Tracer
    @inbounds for itracer = 1:_ntracers
            istate = itracer + _nsd + 2
            Qtracers = reshape((@view Q[:, istate, :]), ntuple(j->(N+1),dim)..., nelem)
            writemesh(@sprintf("viz/tracer_%04d_nse3d-tracer_%s_rank_%04d_step_%05d",
                               itracer, ArrType, mpirank, 0), X...;
                      fields = (("Rho", ρ), ("U", U), ("V", V), ("W", W), ("E", E), ("QT",Qtracers)),
                      realelems = mesh.realelems)
    end
    
    mpirank == 0 && println("[DEV] starting time stepper...")
    lowstorageRK(Val(dim), Val(N), mesh, vgeo, sgeo, Q, rhs, D, dt, nsteps, tout,
                 vmapM, vmapP, mpicomm, iplot, visc; ArrType=ArrType, plotstep=plotstep)

    # Write VTK: final solution
    Q_temp=copy(Q)
    convert_set3c_to_set2nc(Val(dim), Val(N), vgeo, Q_temp)
    X = ntuple(j->reshape((@view vgeo[:, _x+j-1, :]), ntuple(j->N+1,dim)...,
                          nelem), dim)
    ρ = reshape((@view Q_temp[:, _ρ, :]), ntuple(j->(N+1),dim)..., nelem)
    U = reshape((@view Q_temp[:, _U, :]), ntuple(j->(N+1),dim)..., nelem)
    V = reshape((@view Q_temp[:, _V, :]), ntuple(j->(N+1),dim)..., nelem)
    W = reshape((@view Q_temp[:, _W, :]), ntuple(j->(N+1),dim)..., nelem)
    E = reshape((@view Q_temp[:, _E, :]), ntuple(j->(N+1),dim)..., nelem)
    E = E .- 300.0
    writemesh(@sprintf("viz/nse%dD_set3c_%s_rank_%04d_step_%05d",
                       dim, ArrType, mpirank, nsteps), X...;
              fields=(("ρ", ρ), ("U", U), ("V", V), ("W", W), ("E", E)),
              realelems=mesh.realelems)
    # Tracer
    @inbounds for itracer = 1:_ntracers
            istate = itracer + _nsd + 2
            Qtracers = reshape((@view Q[:, istate, :]), ntuple(j->(N+1),dim)..., nelem)
            writemesh(@sprintf("viz/tracer_%04d_nse3d-tracer_%s_rank_%04d_step_%05d",
                               itracer, ArrType, mpirank, nsteps), X...;
                      fields = (("Rho", ρ), ("U", U), ("V", V), ("E", E), ("QT", Qtracers)),
                      realelems = mesh.realelems)
    end

    mpirank == 0 && println("[CPU] computing final energy...")
    stats[2] = L2energysquared(Val(dim), Val(N), Q_temp, vgeo, mesh.realelems)

    stats = sqrt.(MPI.allreduce(stats, MPI.SUM, mpicomm))
    if  mpirank == 0
        @show eng0 = stats[1]
        @show engf = stats[2]
        @show Δeng = engf - eng0
    end
end
# }}}

# {{{
function calculate_dry_pressure(z, theta)
    DFloat          = eltype(z)
    γ::DFloat       = _γ
    p0::DFloat      = _p0
    R_gas::DFloat   = _R_gas
    c_p::DFloat     = _c_p
    c_v::DFloat     = _c_v
    gravity::DFloat = _gravity
    
    exner = 1.0 - gravity/(c_p*theta)*z;
    rho   = p0/(R_gas*theta)*(exner)^(c_v/R_gas);
    p     = p0*exner^(c_v/R_gas);

    return (p, rho)
end
# }}}

# {{{ main
function main()
    DFloat = Float64

    # MPI.Init()
    MPI.Initialized() || MPI.Init()
    MPI.finalize_atexit()

    mpicomm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)

    # FIXME: query via hostname
    @hascuda device!(mpirank % length(devices()))

    #Initial Conditions
    
    function ic(dim, x...)
        
        # FIXME: Type generic?
        
        DFloat          = eltype(x)
        γ::DFloat       = _γ
        p0::DFloat      = _p0
        R_gas::DFloat   = _R_gas
        c_p::DFloat     = _c_p
        c_v::DFloat     = _c_v
        gravity::DFloat = _gravity
        
        Qinit           = Array{DFloat}(undef, _nstate+3)
        icase = _icase
        qt, ql, qi = 0.0, 0.0, 0.0

        if(icase == 1) # ---------------------------------------------------------------
            u0      = 0
            #r       = sqrt((x[1]-500)^2 + (x[dim]-350)^2 ) # 2D Thermal Bubble 
            r       = sqrt((x[1]-500)^2+(x[2]-350)^2+(x[dim]-350)^2)
            rc      = 250.0
            θ_ref   = 300.0
            θ_c     = 0.5
            Δθ      = 0.0
            
            if r <= rc
                Δθ  = 0.5 * θ_c * (1.0 + cos(π * r/rc))
            end
            
            θ     = θ_ref + Δθ
            π_k     = 1.0 - gravity/(c_p*θ)*x[dim]
            c       = c_v/R_gas
            ρ_k     = p0/(R_gas*θ)*(π_k)^c
            ρ       = ρ_k
            T      = π_k*θ

            (p,~) =calculate_dry_pressure(x[dim],θ)

            U       = u0
            V       = 0.0
            W       = 0.0
            E       = θ
            Qinit[1] = U
            Qinit[2] = V
            Qinit[3] = W
            Qinit[4] = ρ
            Qinit[5] = θ
            Qinit[_nstate+1] = T
            Qinit[_nstate+2] = θ_ref
            Qinit[_nstate+3] = p

            return Qinit
        
        elseif (icase == 1001) # --------------------------------------------------------
            
            u0      = 0
            r       = sqrt((x[1]-500)^2 + (x[2]-350)^2 + (x[dim]-350)^2 ) 
            rc      = 250.0

            θ_ref   = 300.0
            θ_c     = 0.5
            Δθ      = 0.0

            rtracer = sqrt((x[1]-500)^2 + (x[dim]-350)^2 ) 
            rctracer= 250.0
            qt_ref  = 0.0
            qt_c    = 1.0
            Δqt     = 0.0
            
            if r <= rc
                Δθ  = 0.5 * θ_c  * (1.0 + cos(π * r/rc))
                Δqt = 0.5 * qt_c * (1.0 + cos(π * rtracer/rctracer))
            end
            

            θ_k     = θ_ref + Δθ
            π_k     = 1.0 - gravity/(c_p*θ_k)*x[dim]
            c       = c_v/R_gas
            ρ_k     = p0/(R_gas*θ_k)*(π_k)^c
            qt_k    = qt_ref + Δqt
            ρ       = ρ_k
            U       = u0
            V       = 0.0
            W       = 0.0
            T       = π_k * θ_k
            E_int   = MoistThermodynamics.internal_energy(T, 0.0, 0.0, 0.0)
            E       = E_int + ρ * x[dim]
            q_t     = qt_k
            T       = MoistThermodynamics.air_temperature(E_int, q_t, 0.0, 0.0)
            P       = p0 * (ρ * R_gas * θ_k/p0)^(c_p/c_v)
            
            # Preserve order of variables in Q 
            Qinit[1]= U
            Qinit[2]= V
            Qinit[3]= W
            Qinit[4]= ρ
            Qinit[5]= E
            Qinit[6]= qt
            Qinit[7]= 0.0
            Qinit[8]= 0.0

            Qinit[_nstate + 1] = T
            Qinit[_nstate + 2] = θ_ref
            Qinit[_nstate + 3] = P
            
            return Qinit
        
        elseif(icase == 1002) # ----------------------------------------------------------

            error("Case 1002 not programmed yet")
        
        elseif(icase == 1003)

            u0     = 0.0
            r      = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
            rc     = 250.0

            #Thermal
            θ_ref  = 300.0
            θ_c    =   0.5
            Δθ     =   0.0
            
            #Passive
            rt1  = sqrt((x[1]-500)^2 + (x[dim]-350)^2 )
            rct1 = 250.0
            
            rt2  = sqrt((x[1]-200)^2 + (x[dim]-500)^2 )
            rct2 = 100.0
            
            rt3  = sqrt((x[1]-300)^2 + (x[dim]-400)^2 )
            rct3 = 150.0
            
            qt_ref  =   0.0196 # or 0.0 # # # ASR ASR ASR ASR 
            qt_c    =   1.0
            
            Δqt1    =   0.0
            Δqt2    =   0.0
            Δqt3    =   0.0
            
            if r <= rc
                Δθ  = 0.5 * θ_c  * (1.0 + cos(π * r/rc))
            end 
            if rt1 <= rct1
                Δqt1 = 0.5 * qt_c * (1.0 + cos(π * rt1/rct1))
            end
            if rt2 <= rct2
                Δqt2 = 0.5 * qt_c * (1.0 + cos(π * rt2/rct2))
            end
            if rt3 <= rct3
                Δqt3 = 0.5 * qt_c * (1.0 + cos(π * rt3/rct3))
            end
            
            θ   = θ_ref + Δθ
            π_k = 1.0 - gravity/(c_p*θ)*x[dim]
            c   = c_v/R_gas
            ρ   = p0/(R_gas*θ)*(π_k)^c

            U    = u0
            V    = 0.0
            W    = 0.0
            qt1 = qt_ref + Δqt1
            qt2 = qt_ref + Δqt2
            qt3 = qt_ref + Δqt3
            
            
            Qinit[1] = U
            Qinit[2] = V
            Qinit[3] = W
            Qinit[4] = ρ
            Qinit[5] = θ
            Qinit[6] = qt1
            Qinit[7] = qt2
            Qinit[8] = qt3
            
            return Qinit

        elseif(icase == 1010) # ----------------------------------------------------------
            #
            # Moist bubble: Pressel at al. 2015 JAMES
            #
            u0  = 0.0
            v0  = 0.0
            w0  = 0.0
            rc  = 250.0
            r   = sqrt((x[1]-500)^2 + (x[2]-350)^2 + (x[dim]-350)^2) # 3D bubble description
            
            #Thermal
            θ_ref  = 320.0
            θ_c    = 2.0
            Δθ     = 0.0
            
            #Moisture
            qt_ref      = 0.0196 #kg/kg
            qt       = qt_ref
            ql      = 0.0
            qi      = 0.0
            R_gas    = MoistThermodynamics.gas_constant_air(qt, ql, qi)
            
            Δθ = 0.0
            if r <= 1.0
                Δθ  = θ_c * cos(0.5 * π * r)*cos(0.5 * π * r)
            end
            
            θ   = θ_ref + Δθ
            π_k = 1.0 - gravity/(c_p * θ) * x[dim]
            c   = c_v/R_gas
            ρ   = p0/(R_gas*θ)*(π_k)^c
            p   = p0 * (ρ*R_gas*θ/p0)^(c_p/c_v)
            T   = π_k*θ
            
            # Saturation adjustment
            T_trial     = 290.0
            E_int       = MoistThermodynamics.internal_energy_sat.(T,ρ, qt);
            T           = MoistThermodynamics.saturation_adjustment.(E_int,ρ, qt);
            θ           = T/π_k
            ρ           = p0/(R_gas*θ)*(π_k)^c
            
            #Obtain ql, qi from T, ρ, qt
            ql = zeros(size(T)); qi = zeros(size(T))
            MoistThermodynamics.phase_partitioning_eq!(ql,qi,T,ρ,qt);
           
            #Velo 
            U    = u0
            V    = v0
            W    = w0
            
            #ρtotal = ρ_dry*(1 + qt)
            ρt = ρ * (1.0 + qt)
            
            Qinit[1] = U
            Qinit[2] = V
            Qinit[3] = W
            Qinit[4] = ρt
            Qinit[5] = θ #E
            Qinit[6] = qt 
            Qinit[7] = 0.0
            Qinit[8] = 0.0
            Qinit[_nstate+1] = T 
            Qinit[_nstate+2] = θ_ref
            Qinit[_nstate+3] = p
            
            return Qinit
            
        else 

            error("Undefined initial condition: Please assign a valid _icase in \'main\' ")
        
        end

    end

    time_final = DFloat(700.0)
    iplot = 400
    Ne = 10
    N  = 4
    visc = 1.5
    dim = _nsd
    hardware="cpu"
    @show (N,Ne,visc,iplot,time_final,hardware,mpisize)
    mesh3D = brickmesh((range(_xmin; length=Ne+1, stop=_xmax),
                        range(_ymin; length=2, stop=_ymax),
                        range(_zmin; length=Ne+1, stop=_zmax)),
                       (true, true, false),
                       part=mpirank+1, numparts=mpisize)

    if hardware == "cpu"
        mpirank == 0 && println("Running 3d (CPU)...")
        nse(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh3D, time_final, iplot, visc;
              ArrType=Array, tout = 10)
        mpirank == 0 && println()
    elseif hardware == "gpu"
        @hascuda begin
            mpirank == 0 && println("Running 3d (GPU)...")
            nse(Val(dim), Val(N), mpicomm, (x...)->ic(dim, x...), mesh3D, time_final, iplot, visc;
                  ArrType=CuArray, tout = 10)
            mpirank == 0 && println()
        end
    end
    nothing
end
# }}}

main()
