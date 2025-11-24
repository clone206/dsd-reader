# dsd-reader
A library for reading DSD audio data. Allows for reading from standard in ("stdin"), 
DSD container files (e.g. DSF or DFF), and raw DSD files, which are assumed to contain 
no metadata. For reading stdin or raw DSD files, the library relies on certain input 
parameters to interpret the format of the DSD data.

Provides an iterator over the frames of the DSD data, which is basically a vector 
of channels in planar format, with a `block_size` slice for each channel in least 
significant bit first order. Channels are ordered by number (ch1,ch2,...). 
This planar format was chosen due to the prevalence of DSF files and the efficiency 
with which it can be iterated over and processed in certain scenarios, 
however it should be trivial for the implementer to convert to interleaved format if needed.
