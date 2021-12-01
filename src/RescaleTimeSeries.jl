module RescaleTimeSeries

export rescale

using OrderedCollections
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
Return index of closest frequency.
"""
function index_closest_freq(value::Number, freqs::AbstractVector)
    findmin(abs.(freqs .- value))[2]
end

"""
Sort an unsorted Dict by magnitude of values.
"""
function sort_magnitude(dict::AbstractDict{K,V}) where {K,V}
    sorted = OrderedDict{K,V}(sort(collect(dict); by = x->abs(x[2]), rev=true))
    return sorted
end

"""
Return scaling for base -> target.
Scaling is a Dict of frequencies => scaling factor. Frequencies belong to base.
Dict is sorted by descending magnitude of scaling factor.
"""
function get_scaling(base::AbstractVector, target::AbstractVector)
    fft_base = norm_fft(base)
    freq_base = rfftfreq(length(base))
    fft_target = norm_fft(target)
    freq_target = rfftfreq(length(target))
    unsorted_scaling = Dict{Number,Number}()
    l = length(freq_base)
    for i in 1:l
        unsorted_scaling[freq_base[i]] = fft_target[index_closest_freq(freq_base[i], freq_target)] - fft_base[i]
    end
    sorted_scaling = sort_magnitude(unsorted_scaling)
    return sorted_scaling
end

"""
Apply scaling to base time series.
The order can be up to the number of FFT frequencies.
"""
function apply_scaling(base::AbstractVector, scaling::AbstractDict; order = nothing)
    freq_base = rfftfreq(length(base))
    # If no order is defined, applying all scaling components
    if isnothing(order)
        order = length(scaling)
    end
    fft_scaled = norm_fft(base)
    components = collect(scaling)
    i = 1
    while !isempty(components) && i <= order
        freq, factor = popfirst!(components)
        fft_scaled[index_closest_freq(freq, freq_base)] += factor
        i = i+1
    end
    scaled = denorm_ifft(fft_scaled)
    return scaled
end

@doc """
    rescale(target_collapsed, ref_expanded, ref_collapsed; order = nothing)
Rescale a target collapsed time series.

| Argument           | Description                                      | 
|--------------------|:-------------------------------------------------| 
| `target_collapsed` | Target vector (low resolution)                   | 
| `ref_expanded`     | Template vector (high resolution)                | 
| `ref_collapsed`    | Template vector (low resolution)                 | 
| `order`            | Number of Fourier components transferred         | 

"""
function rescale(target_collapsed, ref_expanded, ref_collapsed; order = nothing)
    scaling = get_scaling(ref_collapsed, target_collapsed)
    scaled = apply_scaling(ref_expanded, scaling; order=order)
    return scaled
end

end # module
