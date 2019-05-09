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

const _nstate = 1
const _nvgeo = 9
const _ξx, _ηx, _ξy, _ηy, _MJ, _MJI, _ωJ, _x, _y = 1:_nvgeo
const vgeoid = (ξx = _ξx, ηx = _ηx,
                ξy = _ξy, ηy = _ηy,
                MJ = _MJ, MJI = _MJI, ωJ = _ωJ, 
                x = _x,   y = _y)
const _nsgeo = 4
const _nx, _ny, _sMJ, _vMJI = 1:_nsgeo
const sgeoid = (nx = _nx, ny = _ny, sMJ = _sMJ, vMJI = _vMJI)

# {{{ compute geometry
function computegeometry(::Val{dim}, mesh, D, ξ, ω, meshwarp, vmapM) where dim
    # Compute metric terms
    Nq = size(D, 1)
    DFloat = eltype(D)
    (nface, nelem) = size(mesh.elemtoelem)
    crd = creategrid(Val(dim), mesh.elemtocoord, ξ)
    vgeo = zeros(DFloat, Nq^dim, _nvgeo, nelem)
    sgeo = zeros(DFloat, _nsgeo, Nq^(dim-1), nface, nelem)
    (ξx,ηx, ξy,ηy, MJ, MJI, ωJ, x, y) = ntuple(j->(@view vgeo[:, j, :]), _nvgeo)
    J = similar(x)
    (nx, ny, sMJ, vMJI) = ntuple(j->(@view sgeo[ j, :, :, :]), _nsgeo)
    sJ = similar(sMJ)
    X = ntuple(j->(@view vgeo[:, _x+j-1, :]), dim)
    creategrid!(X..., mesh.elemtocoord, ξ)
    # Compute the metric terms
    computemetric!(x, y, J, ξx, ηx, ξy, ηy, sJ, nx, ny, D)
    @show(sJ)
    @show(size(sJ))
    
    M = kron(1, ntuple(j->ω, dim)...)
    MJ .= M .* J
    MJI .= 1 ./ MJ
    vMJI .= MJI[vmapM]
    sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
    sMJ .= sM .* sJ
    (vgeo, sgeo, J, MJ, M, sMJ, sJ)

end
# }}}
      
# Simple unit test to demo 2-d integration operator using Canary quadrature
# functions.  
function vertint!(::Val{dim}, ::Val{N}, ::Val{Ne_x}, ::Val{Ne_y}, vgeo, J, sJ, D, elems, ω, sMJ) where {dim, N, Ne_x, Ne_y}
    
  DFloat = eltype(vgeo)
  Nq = N + 1

  vgeo = reshape(vgeo, Nq, Nq, _nvgeo, Ne_x, Ne_y)
  QI = zeros(DFloat, Nq, Nq, Ne_x, Ne_y)
  
  (ξ,ω) = lglpoints(DFloat, N)
  D = spectralderivative(ξ)
  # calculate for debugging / checks)  
  @inbounds for ex = 1:Ne_x, ey = 1:Ne_y 
    # along vertical elems on a stack
    @inbounds for i = 1:Nq # along horizontal nodes
      # By setting the loop extent to 1 we go along one single vertical node stack
      y = vgeo[i, :, _y, ex, ey]
      J = D * y
      @inbounds for j = 1:Nq # along vertical nodes
        # f is the test function 
        x = vgeo[i, j, _x, ex, ey]
        f = sin(y[j]) + cos(y[j])^2 
        QI[i, j, ex, ey] = f * ω[j] * J[j] 
      end
      QI0, QI1, QI2 = 0, 0, 0
      @inbounds for ey = 1:Ne_y, j = 1:Nq
        y = vgeo[i, j, _y, ex, ey]
        QI0 += QI[i, j, ex, ey] 
        if y <= 1  
          # Subset of vertical domain integral
          # Only compute for ∫f dy from [0, 1]
          QI1 += QI[i, j, ex, ey] 
        end
      end
      QI2 = QI0 - QI1
      @show(QI0, QI1, QI2)
    end
  end

end


function driver(::Val{dim}, ::Val{N}, ::Val{Ne_x}, ::Val{Ne_y}, mpicomm, mesh, tend,
             advection, visc; meshwarp=(x...)->identity(x), tout = 60, ArrType=Array,
             plotstep=0) where {dim, N, Ne_x, Ne_y}

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
    (vgeo, sgeo, J, MJ, sMJ, sJ) = computegeometry(Val(dim), mesh, D, ξ, ω, meshwarp, vmapM)
    (nface, nelem) = size(mesh.elemtoelem)
    Q = zeros(DFloat, (N+1)^dim, _nstate, nelem)

    @inbounds for e = 1:nelem, i = 1:(N+1)^dim, 
        x = vgeo[i,_x, e]
        y = vgeo[i,_y, e]
        Q[i,1, e] = 1 #cospi(y/2)
    end
    vertint!(Val(dim), Val(N), Val(Ne_x), Val(Ne_y), vgeo, J, sJ, D, mesh.realelems, ω, sMJ)
end

function main()
    
    DFloat = Float64
    dim = 2
    N = 10
    #Ne= (1,1) 
    #Ne= (1,2) 
    #Ne= (2,1)
    Ne= (2,2) 
    Ne_x = Ne[1]
    Ne_y = Ne[2]
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
    periodic = (false,false)
    # No advection terms
    advection = false
    # Generate mesh
    x0, x1 = 0 , π
    y0, y1 = 0 , π
    mesh = brickmesh((range(DFloat(x0); length=Ne[1]+1, stop=x1),
                      range(DFloat(y0); length=Ne[2]+1, stop=y1)), 
		                  periodic; 
                      part=mpirank + 1, 
                      numparts=mpisize)

    # Print run message
    mpirank == 0 && println("Running (CPU)...")
    # Return/store state vector Q obtained from expression in driver()
    Q = driver(Val(dim), Val(N), Val(Ne_x), Val(Ne_y), mpicomm, mesh, time_final,advection, visc;
        ArrType=Array, tout = 10, plotstep = iplot)
    # Check mpirank and print successful completion message
    mpirank == 0 && println("Completed (CPU)")
    
    return Q
end

main()
