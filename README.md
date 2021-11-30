# RescaleTimeSeries

This package exports the function `rescale`. The behavior of `rescale` is the following:
  * Find a transformation (based on FFT) transforming a `ref_collapsed` time series into a `target_collapsed` time series;
  * Apply this transformation to `ref_expanded` time series.

For help: 
```julia
  help?> rescale
```
