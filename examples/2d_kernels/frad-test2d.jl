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

const _nstate = 4
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
    M = kron(1, ntuple(j->ω, dim)...)
    MJ .= M .* J
    MJI .= 1 ./ MJ
    vMJI .= MJI[vmapM]
    sM = dim > 1 ? kron(1, ntuple(j->ω, dim-1)...) : one(DFloat)
    sMJ .= sM .* sJ
    (vgeo, sgeo, J, MJ, M, sMJ)

end
# }}}
      
function vertint_flux!(::Val{dim}, ::Val{N}, ::Val{horsize}, Q, vgeo, sgeo, elems, vmapM, vmapP, J, D, mesh) where {dim, N, horsize} 
  Q_int = 0 
  Np = (N+1)^dim
  Nfp = (N+1)^(dim-1)
  nface = 2*dim
  f = 1
  Q_store = zeros(eltype(Q), Nfp)
  ibot = 0 
  botelems = zeros(eltype(horsize), horsize)
  @inbounds for e in elems
    if (e == mesh.elemtoelem[3,e])
        ibot += 1
        botelems[ibot] = e
    end
  end
  vcol = 0 

  Ne_vert = Int64(length(elems) / horsize)
  vert_col = zeros(eltype(botelems), horsize, Ne_vert)
  ibot = 0 
  @inbounds for ebot in botelems
    ibot += 1
    # Assuming non-periodic conditions for the top, bottom
    # We use the list of bottom elements to then find the 
    # elements `stacked` vertically
    local_e = ebot
    elemind = 1 
    vert_col[ibot, elemind] = ebot
    while (local_e != mesh.elemtoelem[4,local_e] ) 
      elemind += 1
      vert_col[ibot, elemind] = mesh.elemtoelem[4,local_e] 
      @show(ibot, elemind)
      local_e = mesh.elemtoelem[4, local_e]
    end
  end
  
  
  

  @inbounds for ibot in botelems
    elem_list = vert_col[ibot,:]
    Q_int = 0
    # Note that this assumes a structured grid 
    # Parallel sides (vertical / horizontal) so that the surface metrics can 
    # be assumed constant across all element nodes
         @inbounds for e in elem_list
          faceid = mesh.elemtoelem[4,e]
            for n = 1:Nfp
              sMJ = sgeo[_sMJ, n, f, e]
              idM = vmapM[n, f, e]
              eM = e 
              vidM = ((idM - 1) % Np) + 1 
              Q_int += sMJ * Q[vidM, 1, eM]
            end
            @show(Q_int)
         end
    end
end

  

function driver(::Val{dim}, ::Val{N}, ::Val{horsize}, mpicomm, mesh, tend,
             advection, visc; meshwarp=(x...)->identity(x), tout = 60, ArrType=Array,
             plotstep=0) where {dim, N, horsize}

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
        Q[i,1, e] = 1 #cospi(y/2)
    end
    vertint_flux!(Val(dim), Val(N), Val(horsize), Q, vgeo, sgeo, mesh.realelems, vmapM, vmapP, J, D, mesh)
    #vertint2_flux!(Val(dim), Val(N), Val(horsize), Q, vgeo, sgeo, mesh.realelems, vmapM, vmapP, J, D)
end

function main()
    
    DFloat = Float64
    dim = 2
    N = 2
    Ne= (5,1)
    horsize = Ne[1]
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
    mesh = brickmesh((range(DFloat(-1); length=Ne[1]+1, stop=1),
                      range(DFloat(-1); length=Ne[2]+1, stop=1)), 
		                  periodic; 
                      part=mpirank + 1, 
                      numparts=mpisize)

    # Print run message
    mpirank == 0 && println("Running (CPU)...")
    # Return/store state vector Q obtained from expression in driver()
    Q = driver(Val(dim), Val(N), Val(horsize), mpicomm, mesh, time_final,advection, visc;
        ArrType=Array, tout = 10, plotstep = iplot)
    # Check mpirank and print successful completion message
    mpirank == 0 && println("Completed (CPU)")
    
    return Q
end

main()
