# dsd-reader
A library for reading DSD audio data. DSD is a high-resolution digital audio format which
encodes audio as a 1 bit stream at high sample rates using delta sigma modulation.

Allows for reading from standard in ("stdin"),
DSD container files (e.g. DSF or DFF), and raw DSD files, which are assumed to contain
no metadata. For reading stdin or raw DSD files, the library relies on certain input
parameters to interpret the format of the DSD data.

Provides iterators over the frames of the DSD data. `dsd_iter()` returns a vector
of channels in planar format, with a `block_size` slice for each channel in least
significant bit first order. Channels are ordered by number (ch1,ch2,...).
This planar format was chosen due to the prevalence of DSF
files and the efficiency with which it can be iterated over and processed
in certain scenarios. For more control over the output of planar data, there is 
also a `planar_iter(out_lsbf, out_block_size)` which allows you to specify 
the bit endianness and block size of the output.

There is also an interleaved iterator available, which can be set to output
either least significant bit first or most significant bit first.
The output is a vector containing a single slice with 1 byte per channel,
ordered by channel number, with this pattern repeating over each full frame.

For an example of a binary that uses this library, see [dsd2dxd](https://github.com/clone206/dsd2dxd).

## DFF Notes
For .dff files, this library only supports ID3 tags that appear at the end of the file, not those found in the property chunk. DST is not supported. Currently only supports mono and stereo audio.

## Adding a Container Format
To add support for a new input file type, first implement the shared `dsd-source` traits in the crate that owns the format:

1. Implement `DsdSource` for the format's file type. Its `info()` method must report the native channel count, endianness, layout, block size, sample rate, audio length, data offset, and optional tag (all as `Option`s except audio length/data offset, which are always knowable). Its `reader()` method must return a boxed, sendable reader positioned at the beginning of the audio data. Its `file_len()` method must report the underlying file's actual on-disk size (typically `self.file.metadata()?.len()`).
2. Implement `DsdSourceExtensions` for the same type and list its lowercase file extensions in `EXTENSIONS`.
3. Publish the format crate, then add its crates.io dependency to `dsd-reader/Cargo.toml`.
4. Update `src/dsd_file.rs`: add a `DsdFileFormat` variant, include it in `is_container()`, add its extension-detection arm to `From<&PathBuf> for DsdFileFormat`, add an `open_<format>()` helper that opens the file and returns `Box<dyn DsdSource>`, and add a match arm for it in `open_source()` (falling back to `open_raw()` on a container open/parse error, same as the existing DSF/DFF arms).
5. Regenerate `Cargo.lock` and add tests or fixtures for opening the new format and iterating its audio data.

The existing `DsdReader` and `DsdIter` code should not need format-specific branches. They read every field they need straight off the returned `Box<dyn DsdSource>` via the trait's own getter methods (`channels()`, `endianness()`, `layout()`, etc.), and use the reported layout and endianness to reshape data into the requested output format. Keep format parsing and native audio-stream handling in the format crate, and keep dispatch registration in `src/dsd_file.rs`.

## Examples

### Opening and reading a DFF file
```rust
use std::path::PathBuf;
use dsd_reader::DsdReader;

let in_path = PathBuf::from("my/music.dff");
// Constructor for use with container files. DSF works the same
let dsd_reader = DsdReader::from_container(in_path.clone()).unwrap();
let channels_num = dsd_reader.channels_num();
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

### Reading from stdin
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
let channels_num = dsd_reader.channels_num();
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

### Reading from raw dsd file (no metadata contained within)
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
let channels_num = dsd_reader.channels_num();
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
