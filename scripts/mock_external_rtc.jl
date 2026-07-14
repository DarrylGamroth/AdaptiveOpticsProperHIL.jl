using AdaptiveOpticsProperHIL
using Sockets

host = ip"127.0.0.1"
port = parse(Int, get(ENV, "AOS_RTC_PORT", "50000"))
slope_count = parse(Int, get(ENV, "AOS_RTC_SLOPES", "512"))
command_count = parse(Int, get(ENV, "AOS_RTC_COMMANDS", "66"))
precision = lowercase(get(ENV, "AOS_RTC_PRECISION", "float32"))
T = precision == "float32" ? Float32 :
    precision == "float64" ? Float64 :
    throw(ArgumentError("AOS_RTC_PRECISION must be float32 or float64"))
gain = parse(Float64, get(ENV, "AOS_RTC_GAIN", "0.1"))
max_cycles = parse(Int, get(ENV, "AOS_RTC_CYCLES", string(typemax(Int))))

server = listen(host, port)
println("mock RTC listening on ", host, ":", port,
    " slopes=", slope_count,
    " commands=", command_count,
    " precision=", T)
client = accept(server)
println("mock RTC accepted ", getpeername(client))
try
    serve_mock_rtc!(client;
        slope_count=slope_count,
        command_count=command_count,
        T=T,
        gain=gain,
        max_cycles=max_cycles,
    )
finally
    close(client)
    close(server)
end
