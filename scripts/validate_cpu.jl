using AdaptiveOpticsProperHIL

results = validate_conventions(T=Float64, n=32, atol=1e-12)
println("cpu_opd_convention_validation")
for name in sort(collect(keys(results)); by=String)
    println("  ", name, " max_abs=", results[name].max_abs, " rms=", results[name].rms)
end

ctx = build_hil_context(resolution=64, n_act=8, T=Float32)
psf, sampling = hil_step!(ctx)
println("cpu_hil_smoke")
println("  psf_size=", size(psf))
println("  sampling=", sampling)
println("  command_length=", length(ctx.command_buffer))
