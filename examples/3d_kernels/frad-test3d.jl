#--------------------------------Markdown Language Header-----------------------
# # 1D Radiation Forcing
#--------------------------------Markdown Language Header-----------------------



include(joinpath(@__DIR__,"vtk.jl"))
include(joinpath(@__DIR__,"../../src/Canary.jl"))
using MPI
using ..Canary
using LinearAlgebra
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
    
    #Compute the metric terms
    computemetric!(x, y, z, J, ξx, ηx, ζx, ξy, ηy, ζy, ξz, ηz, ζz, sJ,
                   nx, ny, nz, D)
    M = kron(1, ntuple(j->ω, dim)...)
    MJ .= M .* J
    MJI .= 1 ./ MJ
    vMJI .= MJI[vmapM]
    sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
    sMJ .= sM .* sJ
    (vgeo, sgeo, J, MJ, M, sMJ)

end
# }}}
      
function vertint_flux!(::Val{dim}, ::Val{N}, Q, vgeo, sgeo, elems, vmapM, vmapP, J, D)where {dim, N}
```
Single column integral of arbitrary function 
Currently ∫f(x) dx in the DG is calculated using the weighted basis functions. 
Care: in one dimension the weights are simply given by the MJ metric calculated in
computemetric.jl (note the dependency on Canary.jl which will disappear once we fold the code
into CliMA)

Q in the argument can be replaced with any field for which the integral is required
```
  Q_int = 0 
  Np = (N+1)^dim
  Nfp = (N+1)^(dim-1)
  nface = 2*dim
  f = 1
  @inbounds for e in elems
            for n = 1:Nfp
              sMJ = sgeo[_sMJ, n, f, e]
              
              idM, idP = vmapM[n, f, e], 
                         vmapP[n, f, e]
              eM, eP = e, 
                       ((idP - 1) ÷ Np) + 1
              
              vidM, vidP = ((idM - 1) % Np) + 1,  
                           ((idP - 1) % Np) + 1
                           Q_int += sMJ * Q[vidM, 1, eM]
            end
        @show(Q_int)
   end
end

function driver(::Val{dim}, ::Val{N}, mpicomm, mesh, tend,
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
    (vgeo, sgeo, J, MJ, sMJ) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM)
    (nface, nelem) = size(mesh.elemtoelem)
        
    Q = zeros(DFloat, (N+1)^dim, _nstate, nelem)

    @inbounds for e = 1:nelem, i = 1:(N+1)^dim, 
        x = vgeo[i,_x, e]
        y = vgeo[i,_y, e]
        Q[i,1, e] = 1#cospi(y/2)
        Q[i,2, e] = x
        Q[i,3, e] = 0
        Q[i,4, e] = sin(x)
    end
  vertint_flux!(Val(dim), Val(N), Q, vgeo, sgeo, mesh.realelems, vmapM, vmapP, J, D)
end

function main()
    
    DFloat = Float64
    dim = 2
    N=2
    Ne= (1,3)
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
    
    # Aperiodic boundary conditions
    periodic = (true,false)
    # No advection terms
    advection = false
    # Generate mesh
    mesh = brickmesh((range(DFloat(-1); length=Ne[1]+1, stop=1),
                      range(DFloat(-1); length=Ne[2]+1, stop=1)), 
		                  periodic; 
                      part=mpirank + 1, 
                      numparts=mpisize)

    # Print run message
    mpirank == 0 && println("Running (CPU)...")
    # Return/store state vector Q obtained from expression in driver()
    Q = driver(Val(dim), Val(N), mpicomm, mesh, time_final,advection, visc;
        ArrType=Array, tout = 10, plotstep = iplot)
    # Check mpirank and print successful completion message
    mpirank == 0 && println("Completed (CPU)")
    
    return Q
end

main()
