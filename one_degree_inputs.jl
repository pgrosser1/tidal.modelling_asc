branch_url = "https://github.com/CliMA/OceananigansArtifacts.jl/raw/glw/near-global-data"

dir = "lat_lon_bathymetry_and_fluxes"
bathymetry_name                  = "bathymetry_lat_lon_360_150.jld2"
bathymetry_url = joinpath(branch_url, dir, bathymetry_name)

dep = DataDep("near_global_one_degree",
              "Bathymetry for near-global one degree simulations",
              [bathymetry_url])

DataDeps.register(dep)

bathymetry_path = datadep"near_global_one_degree/bathymetry_lat_lon_360_150.jld2"

bathymetry_file = jldopen(bathymetry_path)
bathymetry = bathymetry_file["bathymetry"]
close(bathymetry_file)

# Vertical grid with 49 levels.
# Stretched from 10 meters spacing at surface
# to 400 meter at the bottom.
z_49_levels_10_to_400_meter_spacing = [
    -5244.5,
    -4834.0,
    -4446.5,
    -4082.0,
    -3740.5,
    -3422.0,
    -3126.5,
    -2854.0,
    -2604.5,
    -2378.0,
    -2174.45,
    -1993.62,
    -1834.68,
    -1695.59,
    -1572.76,
    -1461.43,
    -1356.87,
    -1255.54,
    -1155.53,
    -1056.28,
     -958.03,
     -861.45,
     -767.49,
     -677.31,
     -592.16,
     -513.26,
     -441.68,
     -378.18,
     -323.18,
     -276.68,
     -238.26,
     -207.16,
     -182.31,
     -162.49,
     -146.45,
     -133.04,
     -121.27,
     -110.47,
     -100.15,
      -90.04,
      -80.01,
      -70.0,
      -60.0,
      -50.0,
      -40.0,
      -30.0,
      -20.0,
      -10.0,
        0.0
]
