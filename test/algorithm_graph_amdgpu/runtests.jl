using AMDGPU

AMDGPU.functional() || error("AMDGPU is not functional on this host")
AMDGPU.allowscalar(false)

include(joinpath(@__DIR__, "..", "algorithm_graph_gpu_common.jl"))

@testset "AOS-native Proper propagation node on AMDGPU" begin
    run_proper_graph_gpu_tests(AMDGPU.ROCArray, AMDGPU.synchronize)
end
