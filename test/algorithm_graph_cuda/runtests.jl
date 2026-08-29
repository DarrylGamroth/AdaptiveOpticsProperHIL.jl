using CUDA

CUDA.functional() || error("CUDA is not functional on this host")
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "..", "algorithm_graph_gpu_common.jl"))

@testset "AOS-native Proper propagation node on CUDA" begin
    run_proper_graph_gpu_tests(CUDA.CuArray, CUDA.synchronize)
end
