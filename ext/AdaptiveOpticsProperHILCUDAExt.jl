module AdaptiveOpticsProperHILCUDAExt

using AdaptiveOpticsProperHIL
using CUDA
using Proper

struct CUDAGraphPropagation{E,I}
    execution::E
    intensity::I
end

@inline function (propagation::CUDAGraphPropagation)(prepared)
    Proper.launch_cuda_graph!(propagation.execution)
    return propagation.intensity(prepared)
end

function AdaptiveOpticsProperHIL.capture_cuda_optical_model(
    model::AdaptiveOpticsProperHIL.PreparedOpticalIntensityModel;
    warmup::Integer=10,
)
    execution = Proper.prepare_cuda_graph(
        model.propagate!,
        model.prepared;
        warmup,
    )
    propagation = CUDAGraphPropagation(
        execution,
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
