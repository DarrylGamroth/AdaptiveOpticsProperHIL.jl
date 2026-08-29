using AMDGPU

AMDGPU.functional() || error("AMDGPU is not functional on this host")
AMDGPU.allowscalar(false)

include(joinpath(@__DIR__, "..", "calculon_gpu_common.jl"))

@testset "Calculon Proper propagation on AMDGPU" begin
    run_calculon_proper_gpu_tests(AMDGPU.ROCArray, AMDGPU.synchronize)
end
