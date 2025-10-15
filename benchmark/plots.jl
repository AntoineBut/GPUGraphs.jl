using Plots, StatsPlots
using CSV
using DataFrames

# Load the data. Columns are operation, size, implementation, time
df = DataFrame(CSV.File("benchmark/out/spmv_results.csv"))
df_bfs = DataFrame(CSV.File("benchmark/out/bfs_results.csv"))

df[!, :time] /= 1e9  # convert ns to s
df_bfs[!, :time] /= 1e9  # convert ns to s

# Plot the data
p = @df df plot(
    :size,
    :time,
    group = :implementation,
    markershape = [:utriangle :x :circle :square],
    xlabel = "Size",
    ylabel = "Time (s)",
    title = "Square Matrix SpMV times",
    legend = :topleft,
    yscale = :log10,
    xscale = :log10,
    yticks = [10^i for i = -6.0:1.0],
    ylim = (1e-5, 1e+1),


    # log Y scale 
)
display(p)
#"""
# Plot the BFS data
p_bfs = @df df_bfs plot(
    :size,
    :time,
    group = :implementation,
    markershape = [:utriangle :x :circle :square],
    xlabel = "Size",
    ylabel = "Time (s)",
    title = "BFS times",
    legend = :topleft,
    yscale = :log10,
    xscale = :log10,
    yticks = [10^i for i = -6.0:1.0],
    ylim = (1e-5, 1e+1),
)
display(p_bfs)
savefig(p_bfs, "benchmark/out/plot_bfs_results.png")
#"""
# Plot the speedup relative to the SuiteSparseGraphBLAS implementation

# Get the times
ssgb_times = df[df.implementation.=="SuiteSparseGraphBLAS", :time]
csr_gpu_times = df[df.implementation.=="GPUGraphsCSR", :time]
ell_gpu_times = df[df.implementation.=="GPUGraphsELL", :time]
cusparse_csr_times = df[df.implementation.=="CUSPARSE-CSR", :time]
cusparse_csc_times = df[df.implementation.=="CUSPARSE-CSC", :time]



# Calculate the speedup
speedup_csr = ssgb_times ./ csr_gpu_times
speedup_ell = ssgb_times ./ ell_gpu_times
speedup_cusparse_csr = ssgb_times ./ cusparse_csr_times
speedup_cusparse_csc = ssgb_times ./ cusparse_csc_times

# Plot the speedup
speedup_plot = plot(
    unique(df.size),
    [speedup_csr, speedup_ell, speedup_cusparse_csr, speedup_cusparse_csc],
    label = ["CSR" "ELL" "CuSparse-CSR" "CuSparse-CSC"],
    xlabel = "Size",
    ylabel = "Speedup",
    title = "Speedup of SpMV relative to JuliaSparse \n Square Matrix",
    legend = true,
    xscale = :log2,
    xticks = [2^i for i = 1:30],
    markershape = [:utriangle :x :circle :square],
)
display(speedup_plot)
savefig(speedup_plot, "benchmark/out/plot_spmv_speedup.png")
#"""
# Get the times
graphsjl_times = df_bfs[df_bfs.implementation .== "Graphs.jl", :time]

ssgb_times = df_bfs[df_bfs.implementation.=="SuiteSparseGraphBLAS", :time]
csr_gpu_times = df_bfs[df_bfs.implementation.=="GPUGraphsCSR", :time]
ell_gpu_times = df_bfs[df_bfs.implementation.=="GPUGraphsELL", :time]

# Calculate the speedup
speedup_ssgb = graphsjl_times ./ ssgb_times
speedup_csr_bfs = graphsjl_times ./ csr_gpu_times
speedup_sell_bfs = graphsjl_times ./ sell_gpu_times

# Plot the speedup
speedup_plot_bfs = plot(
    unique(df_bfs.size),
    [speedup_csr_bfs, speedup_sell_bfs, speedup_ssgb],
    label = ["CSR" "SELL" "SuiteSparseGraphBLAS"],
    xlabel = "Size",
    ylabel = "Speedup",
    title = "Speedup of BFS relative to Graphs.jl",
    legend = true,
    xscale = :log2,
    xticks = [2^i for i = 1:30],
    markershape = [:utriangle :x :circle :square],
)
display(speedup_plot_bfs)
savefig(speedup_plot_bfs, "benchmark/out/plot_bfs_speedup.png")
#"""

df2 = DataFrame(CSV.File("benchmark/out/spmv_results_data.csv"))
df2[!, :time] /= 1e9  # convert ns to s

df2_bfs = DataFrame(CSV.File("benchmark/out/bfs_results_data.csv"))
df2_bfs[!, :time] /= 1e9  # convert ns to s

# For each dataset, normalize the time by the time of the SuiteSparseGraphBLAS implementation
gb_times = df2[df2.implementation.=="CUSPARSE-CSR", :time]
gb_times_column = repeat(gb_times, inner = 4)
# Normalize the time by the SuiteSparseGraphBLAS time
df2[!, :time] = julia_times_column ./ df2[!, :time]


# Plot the data as 3 bar plots (one for each dataset), with the x-axis as the implementation, and the y-axis as the time
p2 = @df df2 bar(
    1:nrow(df2),
    :time,
    group = :implementation,
    bar_width = 1,
    ylabel = "Speedup relative to JuliaSparse",
    xlabel = "Implementation",
    xticks = (2:4:10, unique(df2.dataset)),
    legend = :topleft,
    title = "Sparse Matrix-Vector Multiplication",
)
# Add the speedup
speedups = df2[!, :time]
if false 
annotate!(1, 0, text("$(round(speedups[1], digits = 2))x \n", :black, 8, :center))
annotate!(2, 0, text("$(round(speedups[2], digits = 2))x \n", :black, 8, :center))
annotate!(3, 0, text("$(round(speedups[3], digits = 2))x \n", :black, 8, :center))
annotate!(4, 0, text("$(round(speedups[4], digits = 2))x \n", :black, 8, :center))
annotate!(5, 0, text("$(round(speedups[5], digits = 2))x \n", :black, 8, :center))
annotate!(6, 0, text("$(round(speedups[6], digits = 2))x \n", :black, 8, :center))
annotate!(7, 0, text("$(round(speedups[7], digits = 2))x \n", :black, 8, :center))
annotate!(8, 0, text("$(round(speedups[8], digits = 2))x \n", :black, 8, :center))
annotate!(9, 0, text("$(round(speedups[9], digits = 2))x \n", :black, 8, :center))
end
display(p2)

#"""
# Plot the BFS data
# For each dataset, normalize the time by the time of the SuiteSparseGraphBLAS implementation
graphsjl_times_bfs = df2_bfs[df2_bfs.implementation .== "Graphs.jl", :time]
graphsjl_times_column_bfs = repeat(graphsjl_times_bfs, inner = 4)
# Normalize the time by the SuiteSparseGraphBLAS time
df2_bfs[!, :time] = df2_bfs[!, :time] ./ graphsjl_times_column_bfs
# Plot the data as 4 bar plots (one for each dataset), with the x-axis as the implementation, and the y-axis as the time
p2_bfs = @df df2_bfs bar(
    1:nrow(df2_bfs),
    :time,
    group = :implementation,
    bar_width = 1,
    ylabel = "Time (relative to Graphs.jl)",
    xlabel = "Implementation",
    xticks = (2:4:13, unique(df2_bfs.dataset)),
    legend = :topleft,
    title = "BFS",
    ylim = (0, 1),

)
# Add the speedup
speedups_bfs = 1 ./ df2_bfs[!, :time]
annotate!(2, 0.2, text("$(round(speedups_bfs[2], digits = 2))x \n", :black, 8, :center))
annotate!(3, 0.2, text("$(round(speedups_bfs[3], digits = 2))x \n", :black, 8, :center))
annotate!(4, 0.2, text("$(round(speedups_bfs[4], digits = 2))x \n", :black, 8, :center))
annotate!(5, 0.2, text("$(round(speedups_bfs[5], digits = 2))x \n", :black, 8, :center))
annotate!(6, 0.2, text("$(round(speedups_bfs[6], digits = 2))x \n", :black, 8, :center))
annotate!(7, 0.2, text("$(round(speedups_bfs[7], digits = 2))x \n", :black, 8, :center))
annotate!(8, 0.2, text("$(round(speedups_bfs[8], digits = 2))x \n", :black, 8, :center))
annotate!(9, 0.2, text("$(round(speedups_bfs[9], digits = 2))x \n", :black, 8, :center))
annotate!(10, 0.2, text("$(round(speedups_bfs[10], digits = 2))x \n", :black, 8, :center))
annotate!(11, 0.2, text("$(round(speedups_bfs[11], digits = 2))x \n", :black, 8, :center))
annotate!(12, 0.2, text("$(round(speedups_bfs[12], digits = 2))x \n", :black, 8, :center))
display(p2_bfs)
#"""
# Save the plots
savefig(p, "benchmark/out/plot_spmv_results.png")
savefig(speedup_plot, "benchmark/out/plot_spmv_speedup.png")
savefig(p2, "benchmark/out/plot_spmv_results_data.png")
