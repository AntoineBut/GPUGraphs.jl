using Metal, KernelAbstractions

LIMIT = 2^32
ELTYPE = Int8
backend = Metal.MetalBackend()

cpu = rand(ELTYPE, LIMIT - 1);
gpu = allocate(backend, eltype(cpu), length(cpu));
copyto!(gpu, cpu);
cpu_2 = collect(gpu);
println(cpu == cpu_2);

cpu = rand(ELTYPE, LIMIT);
gpu = allocate(backend, eltype(cpu), length(cpu));
copyto!(gpu, cpu);
cpu_2 = collect(gpu);
println(cpu == cpu_2);
