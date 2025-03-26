using Plots, StatsPlots
using CSV
using DataFrames

# Load the data. Columns are operation, size, implementation, time
df = DataFrame(CSV.File("benchmark/out/spmv_results.csv"))

df[!, :time] /= 1e9  # convert ns to s

# Plot the data
p = @df df plot(
    :size,
    :time,
    group = :implementation,
    markershape = [:utriangle :x :circle :square],
    xlabel = "Size",
    ylabel = "Time (s)",
    title = "Sparse Matrix-Vector Multiplication",
    legend = :topleft,
    yscale = :log10,
    xscale = :log10,
    yticks = [10^i for i = -6.0:0.0],
    ylim = (1e-5, 1e+0),


    # log Y scale 


)
display(p)

# Plot the speedup relative to the SuiteSparseGraphBLAS implementation

# Get the SuiteSparseGraphBLAS times
ssgb_times = df[df.implementation.=="SuiteSparseGraphBLAS", :time]
gpu_times = df[df.implementation.=="GPUGraphs", :time]

# Calculate the speedup
speedup = ssgb_times ./ gpu_times

# Plot the speedup
speedup_plot = plot(
    unique(df.size),
    speedup,
    xlabel = "Size",
    ylabel = "Speedup",
    title = "Speedup of GPUGraphs relative to SuiteSparseGraphBLAS",
    legend = false,
    markershape = [:utriangle :x :circle :square],
)
display(speedup_plot)
# Save the plots
savefig(p, "benchmark/out/spmv_results.png")
savefig(speedup_plot, "benchmark/out/spmv_speedup.png")
