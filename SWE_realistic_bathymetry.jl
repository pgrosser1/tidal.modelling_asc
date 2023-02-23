using SparseArrays
using Statistics
using SpecialFunctions
using LinearAlgebra
using DataDeps
using JLD2

using Oceananigans
using Oceananigans.Architectures: architecture, device_event, device
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators
using Oceananigans.Grids: new_data
using Oceananigans.Solvers: solve!,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver

using IterativeSolvers

using KernelAbstractions: @kernel, @index

using GLMakie
Makie.inline!(true)

λ = 0.1
g = 9.8
f = 0.5
H = 100 # maximum depth
ω = 5

include("one_degree_inputs.jl")
# include("create_bathymetry.jl")

include("utilities_to_create_matrix.jl")

include("SWE_matrix_components.jl")

# Now let's construct a grid and play around
arch = CPU()
Nx = 120
Ny = 50
Nz = 1

file = jldopen("data/bathymetry_three_degree.jld2")
bathymetry = file["bathymetry"]
close(file)

z_faces = z_49_levels_10_to_400_meter_spacing

underlying_grid = LatitudeLongitudeGrid(arch,
                                        size = (Nx, Ny, Nz),
                                        longitude = (-180, 180),
                                        latitude = (-75, 75),
                                        z = (-H, 0),
                                        halo = (5, 5, 5),
                                        topology = (Periodic, Periodic, Bounded))
# v (y) can be periodic if you make the northern-most and southern-most points 0 (so then periodic is appropriate)

grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bathymetry))

using Oceananigans.Grids: inactive_cell, inactive_node, peripheral_node

[!inactive_cell(i, j, k, grid) for i=1:Nx, j=1:Ny, k=1:Nz]
[!inactive_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz]
[peripheral_node(i, j, k, grid, Face(), Center(), Center()) for i=1:Nx+1, j=1:Ny, k=1:Nz]



loc = (Face, Center, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = OpenBoundaryCondition(0),
                                              east = OpenBoundaryCondition(0))
u = Field(loc, grid)

loc = (Center, Face, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = OpenBoundaryCondition(0),
                                              east = OpenBoundaryCondition(0))
v = Field(loc, grid)

η = CenterField(grid)

# Construct the matrix to inspect
Auu = initialize_matrix(arch, u, u, compute_Auu!)
Auv = initialize_matrix(arch, u, v, compute_Auv!)
Auη = initialize_matrix(arch, u, η, compute_Auη!)
Avu = initialize_matrix(arch, v, u, compute_Avu!)
Avv = initialize_matrix(arch, v, v, compute_Avv!)
Avη = initialize_matrix(arch, v, η, compute_Avη!)
Aηu = initialize_matrix(arch, η, u, compute_Aηu!)
Aηv = initialize_matrix(arch, η, v, compute_Aηv!)
Aηη = initialize_matrix(arch, η, η, compute_Aηη!)

# Add an i omega matrix to Auu, Avv, Aetaeta
Auu_iom = Auu .+ Matrix(im*ω*I, (Nx*Ny,Nx*Ny))
Avv_iom = Avv .+ Matrix(im*ω*I, (Nx*Ny,Nx*Ny))
Aηη_iom = Aηη .+ Matrix(im*ω*I, (Nx*Ny,Nx*Ny))

A = [ Auu_iom   Auv     Auη;
        Avv   Avv_iom   Avη;
        Aηu     Aηv   Aηη_iom]

Ainverse = I / Matrix(A)

b_test = ones(Complex{Float64}, Nx*Ny*3,)

x_truth = Ainverse * b_test

x = zeros(Complex{Float64}, Nx*Ny*3,)

IterativeSolvers.idrs!(x, A, b_test)

@show x

@show x ≈ x_truth
