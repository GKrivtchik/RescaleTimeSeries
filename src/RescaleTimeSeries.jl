module RescaleTimeSeries

export rescale

using FFTW

"""
Normalized real FFT.
Normalization to authorize additive scaling of FFT.
"""
function norm_fft(v::AbstractVector)
    return rfft(v) / length(v)
end

"""
Denormalized real IFFT.
"""
function denorm_ifft(v::AbstractVector)
    ifft_size = 2*(length(v)-1)
    return irfft(v, ifft_size) * ifft_size
end

"""
Return frequency pairs: freqs in f1 to be associated with all f2 frequencies.
"""
function freq_pairs(ref_expanded::AbstractVector, ref_collapsed::AbstractVector) #length(f1) > length(f2)
    f1 = rfftfreq(length(ref_expanded))
    f2 = rfftfreq(length(ref_collapsed))
    corr = Vector{Int64}(undef,length(f2))
    for i in 1:length(f2)
        corr[i] = findmin(abs.(f1 .- f2[i]))[2]
    end
    return corr
end

"""
Get n biggest values and associated frequency.
"""
function get_components(scaling::AbstractVector, ref_expanded::AbstractVector, ref_collapsed::AbstractVector; n=nothing)
    # Default value for n such that all components are used
    max_length = length(scaling)
    if isnothing(n)
        n = max_length
    end
    # Test on maximum possible n value
    if n > max_length
        error("n must be <= $max_length")
    end
    #
    pairs = freq_pairs(ref_expanded, ref_collapsed)
    # working on copies (which will be mutated)
    v = copy(scaling)
    # get n greatest components of norm
    norm = abs.(v)
    comp = Dict{Int64,ComplexF64}()
    for i in 1:n
        indmax = findmax(norm)[2]
        deleteat!(norm,indmax)
        comp[popat!(pairs,indmax)] = popat!(v,indmax)
    end
    return comp
end

@doc """
    rescale(ref_expanded::AbstractVector, ref_collapsed::AbstractVector, target_collapsed::AbstractVector; order = nothing)
Adapt "collapsed" target time series to a higher temporal resolution ("expanded") using template "expanded" and "collapsed" time series.
`ref_collapsed` and `target_collapsed` must have the same number of steps.
The result is a time series with the same resolution as `ref_expanded`

| Argument           | Description                                      | 
|--------------------|:-------------------------------------------------| 
| `ref_expanded`     | Template vector (high resolution)                | 
| `ref_collapsed`    | Template vector (low resolution)                 | 
| `target_collapsed` | Target vector (low resolution)                   | 
| `order`            | Number of Fourier components transferred         | 
"""
function rescale(ref_expanded::AbstractVector, ref_collapsed::AbstractVector, target_collapsed::AbstractVector; order = nothing)
    if isnothing(order)
        order = length(ref_collapsed)/2+1
    end
    # fft of input time series
    fft_ref_expanded = norm_fft(ref_expanded)
    fft_ref_collapsed = norm_fft(ref_collapsed)
    fft_target_collapsed = norm_fft(target_collapsed)
    # scaling evaluated on collapsed time series
    scaling = fft_target_collapsed - fft_ref_collapsed
    # scaling components retained for the transformation
    scaling_comp = get_components(scaling, ref_expanded, ref_collapsed; n=order)
    # initialization of scaled at reference
    fft_scaled = copy(fft_ref_expanded)
    # apply scaling components on corresponding frequencies of reference expanded time series
    for (k,v) in scaling_comp
        fft_scaled[k] = fft_ref_expanded[k] + v
    end
    # inverse fft to get scaled time series
    scaled = denorm_ifft(fft_scaled)
    return scaled
end

end # module
