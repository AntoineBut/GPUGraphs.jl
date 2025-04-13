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
    yticks = [10^i for i = -6.0:1.0],
    ylim = (1e-5, 1e+1),


    # log Y scale 


)
display(p)

# Plot the speedup relative to the SuiteSparseGraphBLAS implementation

# Get the SuiteSparseGraphBLAS times
ssgb_times = df[df.implementation.=="SuiteSparseGraphBLAS", :time]
csr_gpu_times = df[df.implementation.=="GPUGraphsCSR", :time]
ell_gpu_times = df[df.implementation.=="GPUGraphsELL", :time]


# Calculate the speedup
speedup_csr = ssgb_times ./ csr_gpu_times
speedup_ell = ssgb_times ./ ell_gpu_times

# Plot the speedup
speedup_plot = plot(
    unique(df.size),
    [speedup_csr, speedup_ell],
    label = ["CSR" "ELL"],
    xlabel = "Size",
    ylabel = "Speedup",
    title = "Speedup of GPUGraphs relative to SSBG",
    legend = true,
    xscale = :log2,
    xticks = [2^i for i = 1:30],
    markershape = [:utriangle :x :circle :square],
)
display(speedup_plot)


df2 = DataFrame(CSV.File("benchmark/out/spmv_results_data.csv"))
df2[!, :time] /= 1e9  # convert ns to s

# For each dataset, normalize the time by the time of the SuiteSparseGraphBLAS implementation
gb_times = df2[df2.implementation.=="SuiteSparseGraphBLAS", :time]
gb_times_column = repeat(gb_times, inner = 3)
# Normalize the time by the SuiteSparseGraphBLAS time
df2[!, :time] = df2[!, :time] ./ gb_times_column


# Plot the data as 3 bar plots (one for each dataset), with the x-axis as the implementation, and the y-axis as the time
p2 = @df df2 bar(
    1:nrow(df2),
    :time,
    group = :implementation,
    bar_width = 1,
    ylabel = "Time (relative to SuiteSparseGraphBLAS)",
    xlabel = "Implementation",
    xticks = (2:3:10, unique(df2.dataset)),
    legend = :topleft,
    title = "Sparse Matrix-Vector Multiplication",
)
# Add the speedup
speedups = 1 ./ df2[!, :time]
speedups[end] = 1.0
annotate!(2, 0.2, text("$(round(speedups[2], digits = 2))x \n", :black, 8, :center))
annotate!(3, 0.2, text("$(round(speedups[3], digits = 2))x \n", :black, 8, :center))
annotate!(5, 0.2, text("$(round(speedups[5], digits = 2))x \n", :black, 8, :center))
annotate!(6, 0.2, text("$(round(speedups[6], digits = 2))x \n", :black, 8, :center))
annotate!(8, 0.2, text("$(round(speedups[8], digits = 2))x \n", :black, 8, :center))

display(p2)
# Save the plots
savefig(p, "benchmark/out/plot_spmv_results.png")
savefig(speedup_plot, "benchmark/out/plot_spmv_speedup.png")
savefig(p2, "benchmark/out/plot_spmv_results_data.png")
