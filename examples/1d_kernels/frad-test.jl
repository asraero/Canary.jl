#--------------------------------Markdown Language Header-----------------------
# # 1D Radiation Forcing
#--------------------------------Markdown Language Header-----------------------



include(joinpath(@__DIR__,"vtk.jl"))
include(joinpath(@__DIR__,"../../src/Canary.jl"))
using MPI
using ..Canary
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

# {{{ constants
# note the order of the fields below is also assumed in the code.
const _nstate = 3
const _U, _h, _b = 1:_nstate
const stateid = (U = _U, h = _h, b = _b)

const _nvgeo = 4
const _ξx, _MJ, _MJI, _x = 1:_nvgeo

const _nsgeo = 3
const _nx, _sMJ, _vMJI = 1:_nsgeo
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

    (ξx, MJ, MJI, x) = ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
    J = similar(x)
    (nx, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
    sJ = similar(sMJ)
    
    X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
    creategrid!(X..., mesh.elemtocoord, ξ)

    # Compute the metric terms
    computemetric!(x, J, ξx, sJ, nx, D)
    @show(sJ)
    M = kron(1, ntuple(j->ω, dim)...)
    MJ .= M .* J
    MJI .= 1 ./ MJ
    vMJI .= MJI[vmapM]
    sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
    sMJ .= sM .* sJ
    (vgeo, sgeo, J)

end
# }}}
     
# Simple unit test to demo 1-d integration operator using Canary quadrature
# functions.  
function vertint!(::Val{dim}, ::Val{N}, Q, vgeo, J, D, elems, ω) where {dim, N}
    DFloat = eltype(Q)
    Nq = size(D, 1)
    nelem = size(Q)[end]
    vgeo = reshape(vgeo, Nq, _nvgeo, nelem)
    Q = reshape(Q, Nq, 3, nelem)
    Q_int0 = zeros(DFloat, Nq, nelem)
    Q_cumulative = 0
    L2_error     = 0 
    (ξ,ω) = lglpoints(DFloat, N)
    D2 = spectralderivative(ξ)
	  @inbounds for e in elems
			  Q_int0[e] = 0
			  for j = 1:Nq
				  x = vgeo[j, _x, e]
          # f is the test function (arbitrary here)
          # true f is a function of the state Q[i,j,s,e]
          f = sin(x) + cos(x).^2  
          J2 = D2 * ξ
          ωJ = ω .* J
          Q_int0[j, e] = ωJ[j] * f 
			  end
        Q_cumulative =  sum(Q_int0)
	  end
	  @show(Q_cumulative)
end

function driver(::Val{dim}, ::Val{N}, mpicomm, ic, mesh, tend,
             advection, visc; meshwarp=(x...)->identity(x), tout = 60, ArrType=Array,
             plotstep=0) where {dim, N}

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
    (vmapM, vmapP) = mappings(N, mesh.elemtoelem, mesh.elemtoface, mesh.elemtoordr)
    D = spectralderivative(ξ)
    (vgeo, sgeo, J) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM)
    (nface, nelem) = size(mesh.elemtoelem)
        
    Q = zeros(DFloat, (N+1)^dim, _nstate, nelem)

    @inbounds for e = 1:nelem, i = 1:(N+1)^dim
        x = vgeo[i, _x, e]
        r = ic(x)
        Q[i, 1, e] = 1#cospi(x/2)
        Q[i, 2, e] = 0 
        Q[i, 3, e] = 0 
    end
	
    vertint!(Val(dim), Val(N), Q, vgeo, J, D, mesh.realelems, ω) 
end

function main()
    
    DFloat = Float64
    dim = 1
    N = 100
    Ne= 1
    visc=0.01
    iplot=10
    icase=10
    gravity= 10.0
    time_final=DFloat(0.1)
    hardware="cpu"
    @show (N,Ne,visc,iplot,icase,time_final,hardware)
    
    MPI.Initialized() || MPI.Init()
    MPI.finalize_atexit()
    mpicomm = MPI.COMM_WORLD
    mpirank = MPI.Comm_rank(mpicomm)
    mpisize = MPI.Comm_size(mpicomm)
    
    function ic(x...)
        r = x[1] # Linear test function with quadratic integral
        return r
    end
    # Aperiodic boundary conditions
    periodic = (false,)
    # No advection terms
    advection = false
    # Generate mesh
    mesh = brickmesh((range(DFloat(0); length=Ne +1, stop=pi),),
                     periodic; part= mpirank + 1, numparts=mpisize)
    # Print run message
    mpirank == 0 && println("Running (CPU)...")
    # Return/store state vector Q obtained from expression in driver()
    Q = driver(Val(dim), Val(N), mpicomm, ic, mesh, time_final,advection, visc;
        ArrType=Array, tout = 10, plotstep = iplot)
    # Check mpirank and print successful completion message
    mpirank == 0 && println("Completed (CPU)")
    
    return Q
end

main()
