using SparseArrays
using Statistics
using SpecialFunctions
using LinearAlgebra

using Oceananigans
using Oceananigans.Architectures: architecture, device_event, device
using Oceananigans.Utils: launch!
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Operators: ∇²ᶜᶜᶜ
using Oceananigans.Grids: new_data
using Oceananigans.Solvers: solve!,
                            PreconditionedConjugateGradientSolver,
                            MultigridSolver,
                            finalize_solver!

using KernelAbstractions: @kernel, @index

import Oceananigans.Solvers: initialize_matrix
import Base: similar

using GLMakie
Makie.inline!(true)



# my attempt to enforce boundary conditions to pass along
function Base.similar(f::Field, grid=f.grid)
    loc = location(f)
    return Field(loc,
                 grid,
                 new_data(eltype(parent(f)), grid, loc, f.indices),
                 deepcopy(f.boundary_conditions),
                 f.indices,
                 f.operand,
                 deepcopy(f.status))
end

# this won't be necessary when https://github.com/CliMA/Oceananigans.jl/pull/2885 is merged
function initialize_matrix(::CPU, template_field, linear_operator!, args...; boundary_conditions=nothing)

    loc = location(template_field)
    Nx, Ny, Nz = size(template_field)
    grid = template_field.grid

    A = spzeros(eltype(grid), Nx*Ny*Nz, Nx*Ny*Nz)
    make_column(f) = reshape(interior(f), Nx*Ny*Nz)

    if boundary_conditions === nothing
        boundary_conditions = FieldBoundaryConditions(grid, loc, template_field.indices)
    end

    eᵢⱼₖ = Field(loc, grid; boundary_conditions)
    ∇²eᵢⱼₖ = similar(template_field)

    for k = 1:Nz, j in 1:Ny, i in 1:Nx
        parent(eᵢⱼₖ) .= 0
        parent(∇²eᵢⱼₖ) .= 0
        eᵢⱼₖ[i, j, k] = 1
        fill_halo_regions!(eᵢⱼₖ)
        linear_operator!(∇²eᵢⱼₖ, eᵢⱼₖ, args...)

        A[:, Ny*Nx*(k-1) + Nx*(j-1) + i] .= make_column(∇²eᵢⱼₖ)
    end
    
    return A
end

# the function we give to the solvers
function compute_∇²!(∇²φ, φ)
    grid = φ.grid
    arch = architecture(grid)

    # parent(φ) .-= mean(φ)
    fill_halo_regions!(φ)
    event = launch!(arch, grid, :xyz, ∇²!, ∇²φ, grid, φ, dependencies=device_event(arch))
    wait(device(arch), event)
    return nothing
end

@kernel function ∇²!(∇²f, grid, f)
    i, j, k = @index(Global, NTuple)
    @inbounds ∇²f[i, j, k] = ∇²ᶜᶜᶜ(i, j, k, grid, f)
end


# Now let's construct a grid and play around
arch = CPU()

#=
grid = RectilinearGrid(arch,
                       size = (100, 1, 1),
                       x = (-1, 1),
                       y = (0, 1),
                       z = (0, 1),
                       halo = (1, 1, 1),
                       topology = (Bounded, Periodic, Periodic))
=#

grid = RectilinearGrid(arch,
                       size = 300,
                       x = (-1, 1),
                       topology = (Bounded, Flat, Flat))

loc = (Center, Center, Center)
boundary_conditions = FieldBoundaryConditions(grid, loc,
                                              west = ValueBoundaryCondition(0),
                                              east = ValueBoundaryCondition(0))



σ = 8

# a symetric solution with zero mean for φ(-1) = φ(+1)=0
rhs(x, y, z) = x * exp(-σ^2 * x^2)
φ(x, y, z) = √π * (x * erf(σ) - erf(σ * x)) / (4σ^3)

# an assymetric solution for φ(-1) = φ(+1)=0
rhs(x, y, z) = (x - 1/4) * exp(- σ^2 * (x - 1/4)^2)
φ(x, y, z) = √π * ((1 + x) * erf(3σ/4) + (x - 1)* erf(5σ/4) + 2 * erf(σ/4 - x * σ)) / (8σ^3)

# a symetric solution with zero mean for φ'(-1) = φ'(+1)=0
# rhs(x, y, z) = x * exp(-σ^2 * x^2)
# φ(x, y, z) = x * exp(-σ^2) / (2σ^2) - √π * erf(x * σ) / (4σ^3)


# Solve ∇²φ = r
φ_truth = CenterField(grid; boundary_conditions)

# Initialize zero-mean "truth" solution
set!(φ_truth, φ)
fill_halo_regions!(φ_truth)


# Calculate Laplacian of "truth"
r = CenterField(grid)
set!(r, rhs)
parent(r) .-= mean(r) # not sure we need this
fill_halo_regions!(r)


A = initialize_matrix(CPU(), φ_truth, compute_∇²!; boundary_conditions)
# @show eigvals(collect(A))


φ_mg = CenterField(grid; boundary_conditions)

# Now solve numerically via MG or CG solvers

@info "Constructing an Algebraic Multigrid solver..."
@time mgs = MultigridSolver(compute_∇²!, template_field=φ_mg)

@info "Solving with the Algebraic Multigrid solver..."
@time solve!(φ_mg, mgs, r)
fill_halo_regions!(φ_mg)

# Solve Poisson equation
φ_cg = CenterField(grid; boundary_conditions)

@info "Constructing a Preconditioned Congugate Gradient solver..."
@time cg_solver = PreconditionedConjugateGradientSolver(compute_∇²!, template_field=φ_cg, reltol=eps(eltype(grid)))

@info "Solving with the Preconditioned Congugate Gradient solver..."
@time solve!(φ_cg, cg_solver, r)
fill_halo_regions!(φ_cg)


# Compute the ∇²φ to see how good it matches the right-hand-side
∇²φ_cg = CenterField(grid)
compute_∇²!(∇²φ_cg, φ_cg)
fill_halo_regions!(∇²φ_cg)

∇²φ_mg = CenterField(grid)
compute_∇²!(∇²φ_mg, φ_mg)
fill_halo_regions!(∇²φ_mg)


# Now plot

x, y, z = nodes(r)

fig = Figure()
ax1 = Axis(fig[1, 1], xlabel="x", ylabel="∇²φ")
lines!(ax1, x, interior(r, :, 1, 1), linewidth=6, label="truth")
lines!(ax1, x, interior(∇²φ_mg, :, 1, 1), linewidth=3, label="MG")
lines!(ax1, x, interior(∇²φ_cg, :, 1, 1), linewidth=3, linestyle=:dash, label="CG")
axislegend(ax1)

ax2 = Axis(fig[2, 1], xlabel="x", ylabel="φ")
lines!(ax2, x, interior(φ_truth, :, 1, 1), linewidth=6, label="truth")
lines!(ax2, x, interior(φ_mg, :, 1, 1), linewidth=3, label="MG")
lines!(ax2, x, interior(φ_cg, :, 1, 1), linewidth=3, linestyle=:dash, label="CG")
axislegend(ax2)

max_r = maximum(abs.(r))
ylims!(ax1, (-1.2*max_r, 1.2max_r))
current_figure()