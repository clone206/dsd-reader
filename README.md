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

For an example of a binary that uses this library, see [dsd2dxd](https://github.com/clone206/dsd2dxd).

## Examples

Opening and reading a DFF file:
```rust
use std::path::PathBuf;
use dsd_reader::DsdReader;

let in_path = PathBuf::from("my/music.dff");
// Constructor for use with container files. DSF works the same
let dsd_reader = DsdReader::from_container(in_path.clone()).unwrap();
let channels_num = dsd_reader.channels_num() as usize;
let dsd_iter = dsd_reader.dsd_iter().unwrap();

for (read_size, chan_bufs) in dsd_iter {
    eprintln!("read_size: usize is {} bytes.", read_size);
    for chan in 0..channels_num {
        my_process_channel(chan, &chan_bufs[chan]);
    }
}

fn my_process_channel(chan: usize, chan_bytes: &[u8]) {
    eprintln!("Processing channel {} with {} bytes. Not guaranteed to have filled buffers.", chan + 1, chan_bytes.len());
    // do stuff
}
```

Reading from stdin:
```rust
use dsd_reader::{DsdReader, Endianness, FmtType, DsdRate};

let dsd_reader = DsdReader::new(
    None, // in_path: None triggers stdin reading
    FmtType::Interleaved,
    Endianness::MsbFirst,
    DsdRate::DSD64,
    4096, // A safe choice of block size for all DSD inputs
    2 // Stereo
).unwrap();
let channels_num = dsd_reader.channels_num() as usize;
let dsd_iter = dsd_reader.dsd_iter().unwrap();

for (read_size, chan_bufs) in dsd_iter {
    eprintln!("read_size: usize is {} bytes.", read_size);
    for chan in 0..channels_num {
        my_process_channel(chan, &chan_bufs[chan]);
    }
}

fn my_process_channel(chan: usize, chan_bytes: &[u8]) {
    eprintln!("Processing channel {} with {} bytes. Not guaranteed to have filled buffers.", chan + 1, chan_bytes.len());
    // do stuff
}
```

Reading from raw dsd file (no metadata contained within):
```rust
use dsd_reader::{DsdReader, Endianness, FmtType, DsdRate};
use std::path::PathBuf;

let in_path = PathBuf::from("my/raw_audio.dsd");

let dsd_reader = DsdReader::new(
    Some(in_path.clone()),
    FmtType::Planar,
    Endianness::LsbFirst,
    DsdRate::DSD128,
    4096, // A safe choice of block size for all DSD inputs
    1 // Mono
).unwrap();
let channels_num = dsd_reader.channels_num() as usize;
let dsd_iter = dsd_reader.dsd_iter().unwrap();

for (read_size, chan_bufs) in dsd_iter {
    eprintln!(
        "read_size: usize is {} bytes. Not guaranteed to have filled buffers.", 
        read_size
    );
    my_process_channel(0, &chan_bufs[0]);
}

fn my_process_channel(chan: usize, chan_bytes: &[u8]) {
    eprintln!("Processing channel {} with {} bytes.", chan + 1, chan_bytes.len());
    // do stuff
}
```
