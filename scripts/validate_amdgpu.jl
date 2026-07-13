using AMDGPU
using AdaptiveOpticsProperHIL
using AdaptiveOpticsSim
using Proper

AMDGPU.allowscalar(false)
backend_array = AMDGPU.ROCArray
ctx = Proper.RunContext(backend_array{Float32,2})
results = validate_conventions(T=Float32, n=32, backend_array=backend_array, context=ctx, atol=5e-6)
println("amdgpu_opd_convention_validation")
for name in sort(collect(keys(results)); by=String)
    println("  ", name, " max_abs=", results[name].max_abs, " rms=", results[name].rms)
end

hil = build_hil_context(backend=AdaptiveOpticsSim.AMDGPUBackend(), resolution=64, n_act=8, T=Float32)
psf, sampling = hil_step!(hil)
AMDGPU.synchronize()
println("amdgpu_hil_smoke")
println("  psf_size=", size(psf))
println("  sampling=", sampling)
println("  command_length=", length(hil.command_buffer))
