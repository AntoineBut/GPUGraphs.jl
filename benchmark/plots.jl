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
    ylim = (0, 7),
    xscale = :log2,
    xticks = [2^i for i = 1:30],
    yticks = [0, 1, 2, 3, 4, 5, 6],
    markershape = [:utriangle :x :circle :square],
)
display(speedup_plot)
# Save the plots
savefig(p, "benchmark/out/spmv_results.png")
savefig(speedup_plot, "benchmark/out/spmv_speedup.png")


df2 = DataFrame(CSV.File("benchmark/out/spmv_results_comOrkut.csv"))
df2[!, :time] /= 1e9  # convert ns to s

time_csr = df2[df2.implementation.=="GPUGraphsCSR", :time][end]
time_ssgb = df2[df2.implementation.=="SuiteSparseGraphBLAS", :time][end]

# Plot the data

p2 = bar(
    1:1,
    [time_ssgb],
    label = "SuiteSparseGraphBLAS",
    bar_width = 0.5,
    xlabel = "Implementation",
    ylabel = "Time (s)",
    title = "spmv! on com-Orkut (Social Network)",
    legend = :topleft,
)
bar!(
    2:2,
    [time_csr],
    label = "GPUGraphsCSR",
    bar_width = 0.5,
    xlabel = "Implementation",
    ylabel = "Time (s)",
    legend = :topleft,
)
# Add the speedup
speedup = time_ssgb / time_csr
annotate!(2, 3, text("Speedup: $(round(speedup, digits = 2))x", :black, 8, :left))
display(p2)



df3 = DataFrame(CSV.File("benchmark/out/spmv_results_osm.csv"))
df3[!, :time] /= 1e9  # convert ns to s

time_csr = df3[df3.implementation.=="GPUGraphsCSR", :time][end]
time_ell = df3[df3.implementation.=="GPUGraphsELL", :time][end]
time_ssgb = df3[df3.implementation.=="SuiteSparseGraphBLAS", :time][end]

# Plot the data

p3 = bar(
    1:1,
    [time_ssgb],
    label = "SuiteSparseGraphBLAS",
    bar_width = 0.5,
    xlabel = "Implementation",
    ylabel = "Time (s)",
    title = "spmv! on Italy-OSM (Road Network)",
    legend = :topleft,
)
bar!(
    2:2,
    [time_csr],
    label = "GPUGraphsCSR",
    bar_width = 0.5,
    xlabel = "Implementation",
    ylabel = "Time (s)",
    legend = :topleft,
)
bar!(
    3:3,
    [time_ell],
    label = "GPUGraphsELL",
    bar_width = 0.5,
    xlabel = "Implementation",
    ylabel = "Time (s)",
    legend = :topleft,
)
# Add the speedup
speedup_csr = time_ssgb / time_csr
speedup_ell = time_ssgb / time_ell
y_coord1 = time_csr * 1.1
y_coord2 = time_ell * 1.1
annotate!(
    2,
    y_coord1,
    text("Speedup with CSR: $(round(speedup_csr, digits = 2))x \n", :black, 8, :center),
)
annotate!(
    3,
    y_coord2,
    text("Speedup with ELL: $(round(speedup_ell, digits = 2))x \n", :black, 8, :center),
)
display(p3)
