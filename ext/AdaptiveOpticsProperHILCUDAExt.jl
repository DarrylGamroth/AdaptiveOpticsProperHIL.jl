module AdaptiveOpticsProperHILCUDAExt

using AdaptiveOpticsProperHIL
using CUDA

struct CUDAGraphPropagation{G,E,I}
    graph::G
    executable::E
    intensity::I
end

@inline function (propagation::CUDAGraphPropagation)(prepared)
    CUDA.launch(propagation.executable)
    return propagation.intensity(prepared)
end

function AdaptiveOpticsProperHIL.capture_cuda_optical_model(
    model::AdaptiveOpticsProperHIL.PreparedOpticalIntensityModel;
    warmup::Integer=10,
)
    warmup >= 0 || throw(ArgumentError("warmup must be nonnegative"))
    for _ in 1:warmup
        AdaptiveOpticsProperHIL.propagate_optics!(model)
    end
    CUDA.synchronize()

    graph = CUDA.capture() do
        AdaptiveOpticsProperHIL.propagate_optics!(model)
    end
    executable = CUDA.instantiate(graph)
    CUDA.launch(executable)
    CUDA.synchronize()
    propagation = CUDAGraphPropagation(
        graph,
        executable,
        model.intensity,
    )
    return AdaptiveOpticsProperHIL.PreparedOpticalIntensityModel(
        model.prepared,
        propagation,
        model.intensity,
        model.sampling_m,
    )
end

end
