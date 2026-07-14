const _DEFAULT_LATENCY_HIGHEST_TRACKABLE_NS = Int64(3_600_000_000_000)
const _DEFAULT_LATENCY_SIGNIFICANT_FIGURES = 3
const _P99_MIN_SAMPLES = 1_000
const _P99_9_MIN_SAMPLES = 10_000
const _HDRHISTOGRAM_SOURCE_URL = "https://github.com/DarrylGamroth/HdrHistogram.jl.git"
const _HDRHISTOGRAM_SOURCE_REVISION = "cd42987e0739c06d29d20b4341f088b85e7a620c"

struct LatencyHistogramParams
    lowest_discernible_ns::Int64
    highest_trackable_ns::Int64
    significant_figures::Int
end

function LatencyHistogramParams(;
    lowest_discernible_ns::Integer=1,
    highest_trackable_ns::Integer=_DEFAULT_LATENCY_HIGHEST_TRACKABLE_NS,
    significant_figures::Integer=_DEFAULT_LATENCY_SIGNIFICANT_FIGURES)
    lowest = Int64(lowest_discernible_ns)
    highest = Int64(highest_trackable_ns)
    figures = Int(significant_figures)
    lowest >= 1 || throw(ArgumentError("lowest_discernible_ns must be >= 1"))
    lowest <= typemax(Int64) ÷ 2 ||
        throw(ArgumentError("lowest_discernible_ns is too large"))
    highest >= 2 * lowest ||
        throw(ArgumentError("highest_trackable_ns must be >= 2 * lowest_discernible_ns"))
    0 <= figures <= 5 ||
        throw(ArgumentError("significant_figures must be between 0 and 5"))
    return LatencyHistogramParams(lowest, highest, figures)
end

mutable struct LatencyHistogramState{H}
    const params::LatencyHistogramParams
    const histogram::H
end

function LatencyHistogramState(params::LatencyHistogramParams=LatencyHistogramParams())
    histogram = HdrHistogram.Histogram(
        Int64,
        params.lowest_discernible_ns,
        params.highest_trackable_ns,
        params.significant_figures,
    )
    return LatencyHistogramState(params, histogram)
end

@inline function record_latency!(state::LatencyHistogramState, latency_ns::Integer)
    latency_ns >= 0 || throw(ArgumentError("latency_ns must be >= 0"))
    latency_ns <= state.params.highest_trackable_ns ||
        throw(ArgumentError("latency_ns exceeds the fixed histogram range"))
    HdrHistogram.record_value!(state.histogram, Int64(latency_ns))
    return state
end

function merge_latency!(target::LatencyHistogramState, source::LatencyHistogramState)
    target.params == source.params ||
        throw(ArgumentError("latency histogram layouts must match before merging"))
    HdrHistogram.add(target.histogram, source.histogram)
    return target
end

function latency_summary(state::LatencyHistogramState)
    histogram = state.histogram
    samples = Int(HdrHistogram.total_count(histogram))
    samples > 0 || throw(ArgumentError("latency histogram cannot be empty"))
    samples >= _P99_MIN_SAMPLES ||
        throw(ArgumentError("at least $(_P99_MIN_SAMPLES) samples are required to report p99"))
    p99_9 = samples >= _P99_9_MIN_SAMPLES ?
        HdrHistogram.value_at_percentile(histogram, 99.9) : nothing
    return (
        samples=samples,
        min_ns=min(histogram),
        mean_ns=HdrHistogram.mean(histogram),
        stddev_ns=HdrHistogram.stddev(histogram),
        p50_ns=HdrHistogram.value_at_percentile(histogram, 50.0),
        p90_ns=HdrHistogram.value_at_percentile(histogram, 90.0),
        p95_ns=HdrHistogram.value_at_percentile(histogram, 95.0),
        p99_ns=HdrHistogram.value_at_percentile(histogram, 99.0),
        p99_9_ns=p99_9,
        max_ns=max(histogram),
    )
end

function latency_histogram_artifact(state::LatencyHistogramState)
    encoded = HdrHistogram.encode_into_compressed_byte_buffer(state.histogram)
    return (
        encoding="HdrHistogram compressed V2 binary encoded as base64",
        implementation="HdrHistogram.jl",
        implementation_version=string(Base.pkgversion(HdrHistogram)),
        source_url=_HDRHISTOGRAM_SOURCE_URL,
        source_revision=_HDRHISTOGRAM_SOURCE_REVISION,
        value_unit="nanoseconds",
        lowest_discernible_ns=state.params.lowest_discernible_ns,
        highest_trackable_ns=state.params.highest_trackable_ns,
        significant_figures=state.params.significant_figures,
        samples=Int(HdrHistogram.total_count(state.histogram)),
        payload_base64=Base64.base64encode(encoded),
    )
end
