using CUDA

CUDA.functional() || error("CUDA is not functional on this host")
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "..", "calculon_gpu_common.jl"))

@testset "Calculon Proper propagation on CUDA" begin
    run_calculon_proper_gpu_tests(CUDA.CuArray, CUDA.synchronize)
end
