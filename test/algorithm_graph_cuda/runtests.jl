using CUDA

CUDA.functional() || error("CUDA is not functional on this host")
CUDA.allowscalar(false)

include(joinpath(@__DIR__, "..", "algorithm_graph_gpu_common.jl"))

struct CUDAGraphOpticalFixture{A}
    input::A
    output::A
end

@inline function propagate_cuda_fixture!(fixture::CUDAGraphOpticalFixture)
    @. fixture.output = muladd(fixture.input, fixture.input, 1f0)
    return fixture.output
end

@inline cuda_fixture_intensity(fixture::CUDAGraphOpticalFixture) = fixture.output

@testset "prepared optical CUDA Graph capture" begin
    fixture = CUDAGraphOpticalFixture(
        CUDA.zeros(Float32, 8, 8),
        CUDA.zeros(Float32, 8, 8),
    )
    direct = PreparedOpticalIntensityModel(
        fixture,
        propagate_cuda_fixture!,
        cuda_fixture_intensity,
        15f-6,
    )
    @test_throws ArgumentError capture_cuda_optical_model(direct; warmup=-1)

    captured = capture_cuda_optical_model(direct; warmup=2)
    fill!(fixture.input, 2f0)
    @test propagate_optics!(captured) === fixture.output
    CUDA.synchronize()
    @test Array(fixture.output) == fill(5f0, 8, 8)

    # Captured kernels retain buffer addresses, but observe new contents.
    fill!(fixture.input, 3f0)
    propagate_optics!(captured)
    CUDA.synchronize()
    @test Array(fixture.output) == fill(10f0, 8, 8)

    propagate_optics!(captured)
    CUDA.synchronize()
    @test (@allocated propagate_optics!(captured)) == 0
    CUDA.synchronize()
end

@testset "AOS-native Proper propagation node on CUDA" begin
    run_proper_graph_gpu_tests(CUDA.CuArray, CUDA.synchronize)
end
