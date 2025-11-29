//! A library for reading DSD audio data. Allows for reading from standard in ("stdin"), 
//! DSD container files (e.g. DSF or DFF), and raw DSD files, which are assumed to contain 
//! no metadata. For reading stdin or raw DSD files, the library relies on certain input 
//! parameters to interpret the format of the DSD data.
//! 
//! Provides an iterator over the frames of the DSD data, which is basically a vector 
//! of channels in planar format, with a `block_size` slice for each channel in least 
//! significant bit first order. Channels are ordered by number (ch1,ch2,...). 
//! This planar format was chosen due to the prevalence of DSF files and the efficiency 
//! with which it can be iterated over and processed in certain scenarios, 
//! however it should be trivial for the implementor to convert to interleaved format if needed.
//!
//! ## Examples
//! 
//! Opening and reading a DFF file:
//! 
//!```no_run
//! use std::path::PathBuf;
//! use dsd_reader::DsdReader;
//! 
//! let in_path = PathBuf::from("my/music.dff");
//! // Constructor for use with container files. DSF works the same
//! let dsd_reader = DsdReader::from_container(in_path.clone()).unwrap();
//! let channels_num = dsd_reader.channels_num();
//! let dsd_iter = dsd_reader.dsd_iter().unwrap();
//! 
//! for (read_size, chan_bufs) in dsd_iter {
//!     eprintln!("read_size: usize is {} bytes.", read_size);
//!     for chan in 0..channels_num {
//!         my_process_channel(chan, &chan_bufs[chan]);
//!     }
//! }
//! 
//! fn my_process_channel(chan: usize, chan_bytes: &[u8]) {
//!     eprintln!("Processing channel {} with {} bytes. Not guaranteed to have filled buffers.", chan + 1, chan_bytes.len());
//!     // do stuff
//! }
//! ```
//! 
//! Reading from stdin:
//!```no_run
//! use dsd_reader::{DsdReader, Endianness, FmtType, DsdRate};
//! 
//! let dsd_reader = DsdReader::new(
//!     None, // in_path: None triggers stdin reading
//!     FmtType::Interleaved,
//!     Endianness::MsbFirst,
//!     DsdRate::DSD64,
//!     4096, // A safe choice of block size for all DSD inputs
//!     2 // Stereo
//! ).unwrap();
//! let channels_num = dsd_reader.channels_num();
//! let dsd_iter = dsd_reader.dsd_iter().unwrap();
//! 
//! for (read_size, chan_bufs) in dsd_iter {
//!     eprintln!("read_size: usize is {} bytes.", read_size);
//!     for chan in 0..channels_num {
//!         my_process_channel(chan, &chan_bufs[chan]);
//!     }
//! }
//! 
//! fn my_process_channel(chan: usize, chan_bytes: &[u8]) {
//!     eprintln!("Processing channel {} with {} bytes. Not guaranteed to have filled buffers.", chan + 1, chan_bytes.len());
//!     // do stuff
//! }
//! ```
//! 
//! Reading from raw dsd file (no metadata contained within):
//!```no_run
//! use dsd_reader::{DsdReader, Endianness, FmtType, DsdRate};
//! use std::path::PathBuf;
//! 
//! let in_path = PathBuf::from("my/raw_audio.dsd");
//! 
//! let dsd_reader = DsdReader::new(
//!     Some(in_path.clone()),
//!     FmtType::Planar,
//!     Endianness::LsbFirst,
//!     DsdRate::DSD128,
//!     4096, // A safe choice of block size for all DSD inputs
//!     1 // Mono
//! ).unwrap();
//! let channels_num = dsd_reader.channels_num();
//! let dsd_iter = dsd_reader.dsd_iter().unwrap();
//! 
//! for (read_size, chan_bufs) in dsd_iter {
//!     eprintln!(
//!         "read_size: usize is {} bytes. Not guaranteed to have filled buffers.", 
//!         read_size
//!     );
//!     my_process_channel(0, &chan_bufs[0]);
//! }
//! 
//! fn my_process_channel(chan: usize, chan_bytes: &[u8]) {
//!     eprintln!("Processing channel {} with {} bytes.", chan + 1, chan_bytes.len());
//!     // do stuff
//! }
//! ```


pub mod dsd_file;
use crate::dsd_file::{
    DFF_BLOCK_SIZE, DSD_64_RATE, DSF_BLOCK_SIZE, DsdFile, DsdFileFormat, FormatExtensions,
};
use log::{debug, error, info, warn};
use std::convert::{TryFrom, TryInto};
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::io::{self, BufReader, ErrorKind, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const RETRIES: usize = 1; // Max retries for progress send

/// DSD bit endianness
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Endianness {
    LsbFirst,
    MsbFirst,
}

/// DSD channel format
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum FmtType {
    /// Block per channel
    Planar,
    /// Byte per channel
    Interleaved,
}

/// DSD rate multiplier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DsdRate {
    #[default]
    DSD64 = 1,
    DSD128 = 2,
    DSD256 = 4,
}

impl TryFrom<u32> for DsdRate {
    type Error = &'static str;
    fn try_from(v: u32) -> Result<Self, Self::Error> {
        match v {
            1 => Ok(DsdRate::DSD64),
            2 => Ok(DsdRate::DSD128),
            4 => Ok(DsdRate::DSD256),
            _ => Err("Invalid DSD rate multiplier (expected 1,2,4)"),
        }
    }
}

struct InputPathAttrs {
    std_in: bool,
    file_format: Option<DsdFileFormat>,
    parent_path: Option<PathBuf>,
    file_name: OsString,
}

/// DSD input context for reading DSD audio data from various sources.
pub struct DsdReader {
    dsd_rate: DsdRate,
    channels_num: usize,
    std_in: bool,
    tag: Option<id3::Tag>,
    file_name: OsString,
    audio_length: u64,
    block_size: u32,
    parent_path: Option<PathBuf>,
    in_path: Option<PathBuf>,
    lsbit_first: bool,
    interleaved: bool,
    audio_pos: u64,
    file: Option<File>,
    file_format: Option<DsdFileFormat>,
}
impl Default for DsdReader {
    fn default() -> Self {
        DsdReader {
            dsd_rate: DsdRate::default(),
            channels_num: 2,
            std_in: true,
            tag: None,
            file_name: OsString::from("stdin"),
            audio_length: 0,
            block_size: DSF_BLOCK_SIZE,
            parent_path: None,
            in_path: None,
            lsbit_first: false,
            interleaved: true,
            audio_pos: 0,
            file: None,
            file_format: None,
        }
    }
}

impl DsdReader {
    pub fn dsd_rate(&self) -> i32 {
        self.dsd_rate as i32
    }
    pub fn channels_num(&self) -> usize {
        self.channels_num
    }
    pub fn std_in(&self) -> bool {
        self.std_in
    }
    pub fn tag(&self) -> &Option<id3::Tag> {
        &self.tag
    }
    pub fn file_name(&self) -> &OsString {
        &self.file_name
    }
    pub fn audio_length(&self) -> u64 {
        self.audio_length
    }
    pub fn block_size(&self) -> u32 {
        self.block_size
    }
    pub fn parent_path(&self) -> &Option<PathBuf> {
        &self.parent_path
    }
    pub fn in_path(&self) -> &Option<PathBuf> {
        &self.in_path
    }

    /// Construct `DsdReader` from stdin or a file path. Note that for container files,
    /// like DSF or DFF, you may prefer to use `from_container(in_path)` instead. If `in_path` is
    /// `None`, stdin is assumed. If `in_path` is a file containing raw DSD data, the other
    /// parameters are used to interpret the data. If `in_path` is a container file, the
    /// container metadata will override the other parameters.
    /// # Arguments
    /// * `in_path` - Optional path to input file. If `None`, stdin is used.
    /// * `format` - DSD channel format (planar or interleaved)
    /// * `endianness` - DSD bit endianness (LSB-first or MSB-first)
    /// * `dsd_rate` - DSD rate multiplier
    /// * `block_size` - Block size of each read in bytes per channel
    /// * `channels` - Number of audio channels
    pub fn new(
        in_path: Option<PathBuf>,
        format: FmtType,
        endianness: Endianness,
        dsd_rate: DsdRate,
        block_size: u32,
        channels: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let lsbit_first = match endianness {
            Endianness::LsbFirst => true,
            Endianness::MsbFirst => false,
        };

        let interleaved = match format {
            FmtType::Planar => false,
            FmtType::Interleaved => true,
        };

        let path_attrs = Self::get_path_attrs(&in_path)?;

        // Only enforce CLI dsd_rate for stdin or raw inputs
        if path_attrs.std_in && ![1, 2, 4].contains(&(dsd_rate as u32)) {
            return Err("Unsupported DSD input rate.".into());
        }

        let mut ctx = Self {
            lsbit_first,
            interleaved,
            std_in: path_attrs.std_in,
            dsd_rate,
            in_path,
            parent_path: path_attrs.parent_path,
            channels_num: channels,
            block_size: block_size,
            audio_length: 0,
            audio_pos: 0,
            file: None,
            tag: None,
            file_name: path_attrs.file_name,
            file_format: path_attrs.file_format,
        };

        debug!("Set block_size={}", ctx.block_size);

        ctx.init()?;

        Ok(ctx)
    }

    /// Construct DsdReader from container file input (e.g. DSF, DFF)
    pub fn from_container(in_path: PathBuf) -> Result<Self, Box<dyn Error>> {
        if in_path.is_dir() {
            return Err("Input path cannot be a directory".into());
        }
        let file_format = DsdFileFormat::from(&in_path);
        if !file_format.is_container() {
            return Err("Input file is not a recognized container format".into());
        }
        let parent_path = Some(in_path.parent().unwrap_or(Path::new("")).to_path_buf());
        let file_name = in_path
            .file_name()
            .unwrap_or_else(|| "stdin".as_ref())
            .to_os_string();

        let mut ctx = Self::default();
        ctx.in_path = Some(in_path);
        ctx.std_in = false;
        ctx.parent_path = parent_path;
        ctx.file_name = file_name;
        ctx.file_format = Some(file_format);

        debug!("Set block_size={}", ctx.block_size);

        ctx.update_from_file(file_format)?;

        Ok(ctx)
    }

    /// Construct and return instance of DSD sample iterator for reading DSD data frames.
    pub fn dsd_iter(&self) -> Result<DsdIter, Box<dyn Error>> {
        DsdIter::new(self)
    }

    fn get_path_attrs(path: &Option<PathBuf>) -> Result<InputPathAttrs, Box<dyn Error>> {
        if let Some(p) = path {
            if p.is_dir() {
                return Err("Input path cannot be a directory".into());
            }
            let file_format = Some(DsdFileFormat::from(p));
            let parent_path = Some(p.parent().unwrap_or(Path::new("")).to_path_buf());
            let file_name = p
                .file_name()
                .unwrap_or_else(|| "stdin".as_ref())
                .to_os_string();

            Ok(InputPathAttrs {
                std_in: false,
                file_format,
                parent_path,
                file_name,
            })
        } else {
            Ok(InputPathAttrs {
                std_in: true,
                file_format: None,
                parent_path: None,
                file_name: OsString::from("stdin"),
            })
        }
    }

    fn init(&mut self) -> Result<(), Box<dyn Error>> {
        if self.std_in {
            self.debug_stdin();
        } else if let Some(format) = self.file_format {
            self.update_from_file(format)?;
        } else {
            return Err("No valid input specified".into());
        }

        Ok(())
    }

    fn debug_stdin(&mut self) {
        debug!("Reading from stdin");
        debug!(
            "Using CLI parameters: {} channels, LSB first: {}, Interleaved: {}",
            self.channels_num,
            if self.lsbit_first == true {
                "true"
            } else {
                "false"
            },
            self.interleaved
        );
    }

    fn update_from_file(
        &mut self,
        dsd_file_format: DsdFileFormat,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let Some(path) = &self.in_path else {
            return Err("No readable input file".into());
        };

        info!("Opening input file: {}", self.file_name.to_string_lossy());
        debug!(
            "Parent path: {}",
            self.parent_path.as_ref().unwrap().display()
        );
        match DsdFile::new(path, dsd_file_format) {
            Ok(my_dsd) => {
                // Pull raw fields
                let file_len = my_dsd.file().metadata()?.len();
                debug!("File size: {} bytes", file_len);

                self.file = Some(my_dsd.file().try_clone()?);
                self.tag = my_dsd.tag().cloned();

                self.audio_pos = my_dsd.audio_pos();
                // Clamp audio_length to what the file can actually contain
                let max_len: u64 = (file_len - self.audio_pos).max(0);
                self.audio_length = if my_dsd.audio_length() > 0 && my_dsd.audio_length() <= max_len
                {
                    my_dsd.audio_length()
                } else {
                    max_len
                };

                // Channels from container (fallback to CLI on nonsense)
                if let Some(chans_num) = my_dsd.channel_count() {
                    self.channels_num = chans_num;
                }

                // Bit order from container
                if let Some(lsb) = my_dsd.is_lsb() {
                    self.lsbit_first = lsb;
                }

                // Interleaving from container (DSF = block-interleaved → treat as planar per frame)
                match my_dsd.container_format() {
                    DsdFileFormat::Dsdiff => self.interleaved = true,
                    DsdFileFormat::Dsf => self.interleaved = false,
                    DsdFileFormat::Raw => {}
                }

                // Block size from container.
                // For dff, which always has a block size per channel of 1,
                // we accept the user-supplied or default block size, which really
                // just governs how many bytes we read at a time.
                // For DSF, we treat the block size as representing the
                // block size per channel and override any user-supplied
                // or default values for block size.
                if let Some(block_size) = my_dsd.block_size()
                    && block_size > DFF_BLOCK_SIZE
                {
                    self.block_size = block_size;
                    debug!("Set block_size={}", self.block_size,);
                }

                // DSD rate from container sample_rate if valid (2.8224MHz → 1, 5.6448MHz → 2)
                if let Some(sample_rate) = my_dsd.sample_rate() {
                    if sample_rate % DSD_64_RATE == 0 {
                        self.dsd_rate = (sample_rate / DSD_64_RATE).try_into()?;
                    } else {
                        // Fallback: keep CLI value (avoid triggering “Invalid DSD rate”)
                        info!(
                            "Container sample_rate {} not standard; keeping CLI dsd_rate={}",
                            sample_rate, self.dsd_rate as i32
                        );
                    }
                }

                debug!("Audio length in bytes: {}", self.audio_length);
                debug!(
                    "Container: sr={}Hz channels={} interleaved={}",
                    self.dsd_rate as u32 * DSD_64_RATE,
                    self.channels_num,
                    self.interleaved,
                );
            }
            Err(e) if dsd_file_format != DsdFileFormat::Raw => {
                info!("Container open failed with error: {}", e);
                info!("Treating input as raw DSD (no container)");
                self.update_from_file(DsdFileFormat::Raw)?;
            }
            Err(e) => {
                return Err(e);
            }
        }
        Ok(())
    }
}

/// DSD data iterator that yields frames of DSD data. Provided by [DsdReader].
pub struct DsdIter {
    std_in: bool,
    bytes_remaining: u64,
    channels_num: usize,
    channel_buffers: Vec<Box<[u8]>>,
    block_size: u32,
    reader: Box<dyn Read + Send>,
    frame_size: u32,
    interleaved: bool,
    lsbit_first: bool,
    dsd_data: Vec<u8>,
    file: Option<File>,
    audio_pos: u64,
    retries: usize,
}
impl DsdIter {
    pub fn new(dsd_input: &DsdReader) -> Result<Self, Box<dyn Error>> {
        let mut ctx = DsdIter {
            std_in: dsd_input.std_in,
            bytes_remaining: if dsd_input.std_in {
                dsd_input.block_size as u64 * dsd_input.channels_num as u64
            } else {
                dsd_input.audio_length
            },
            channels_num: dsd_input.channels_num,
            channel_buffers: Vec::new(),
            block_size: 0,
            reader: Box::new(io::empty()),
            frame_size: 0,
            interleaved: dsd_input.interleaved,
            lsbit_first: dsd_input.lsbit_first,
            dsd_data: vec![0; dsd_input.block_size as usize * dsd_input.channels_num],
            file: if let Some(file) = &dsd_input.file {
                Some(file.try_clone()?)
            } else {
                None
            },
            audio_pos: dsd_input.audio_pos,
            retries: 0,
        };
        ctx.set_block_size(dsd_input.block_size);
        ctx.set_reader()?;
        Ok(ctx)
    }

    /// Read one frame of DSD data into the channel buffers and return read size.
    #[inline(always)]
    pub fn load_frame(&mut self) -> Result<usize, Box<dyn Error>> {
        //stdin always reads frame_size
        let partial_frame = if self.bytes_remaining < self.frame_size as u64 && !self.std_in {
            true
        } else {
            false
        };

        if self.interleaved {
            if partial_frame {
                self.set_block_size((self.bytes_remaining / self.channels_num as u64) as u32);
            }
            self.reader
                .read_exact(&mut self.dsd_data[..self.frame_size as usize])?;
            // Copy interleaved data into channel buffers
            for chan in 0..self.channels_num {
                let chan_bytes = self.get_chan_bytes_interl(chan, self.frame_size as usize);
                self.channel_buffers[chan].copy_from_slice(&chan_bytes);
            }
            Ok(self.frame_size as usize)
        } else {
            // Planar DSF: Each channel block is fixed size (block_size) and the last
            // frame may contain zero-padded tail inside each channel block.
            let mut total_valid = 0usize;
            let remaining = if partial_frame {
                self.reset_buffers();
                self.bytes_remaining
            } else {
                self.frame_size as u64
            };
            let valid_for_chan = (remaining / self.channels_num as u64) as usize;
            let padding = self.block_size as usize - valid_for_chan;

            for chan in 0..self.channels_num {
                let chan_buf = &mut self.channel_buffers[chan];

                self.reader.read_exact(&mut chan_buf[..valid_for_chan])?;
                total_valid += valid_for_chan;

                if padding > 0 {
                    // If block is padded, discard padding from file so next channel starts aligned.
                    let byte_reader = self.reader.as_mut();
                    for _ in 0..padding {
                        byte_reader.bytes().next();
                    }
                }
            }
            Ok(total_valid)
        }
    }

    /// Take frame of interleaved DSD bytes from internal buffer and return one
    /// channel with the endianness we need.
    #[inline(always)]
    fn get_chan_bytes_interl(&self, chan: usize, read_size: usize) -> Vec<u8> {
        let chan_size = read_size / self.channels_num;
        let mut chan_bytes: Vec<u8> = Vec::with_capacity(chan_size);

        for i in 0..chan_size {
            let byte_index = chan + i * self.channels_num;
            if byte_index >= self.dsd_data.len() {
                break;
            }
            let b = self.dsd_data[byte_index];
            chan_bytes.push(if self.lsbit_first {
                b
            } else {
                bit_reverse_u8(b)
            });
        }
        chan_bytes
    }

    fn set_block_size(&mut self, block_size: u32) {
        self.block_size = block_size;
        self.frame_size = self.block_size * self.channels_num as u32;

        self.channel_buffers = (0..self.channels_num)
            .map(|_| vec![0x69u8; self.block_size as usize].into_boxed_slice())
            .collect();

        debug!("Set block_size={}", self.block_size,);
    }

    fn reset_buffers(&mut self) {
        for chan in 0..self.channels_num {
            let chan_buf = &mut self.channel_buffers[chan];
            for byte in chan_buf.iter_mut() {
                *byte = 0x69;
            }
        }
    }

    fn set_reader(&mut self) -> Result<(), Box<dyn Error>> {
        if self.std_in {
            // Use Stdin (not StdinLock) so the reader is 'static + Send for threaded use
            self.reader = Box::new(BufReader::with_capacity(
                self.frame_size as usize * 8,
                io::stdin(),
            ));
            return Ok(());
        }
        // Obtain an owned File by cloning the handle from InputContext, then seek if needed.
        let mut file = self
            .file
            .as_ref()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Missing input file handle"))?
            .try_clone()?;

        if self.audio_pos > 0 {
            file.seek(SeekFrom::Start(self.audio_pos as u64))?;
            debug!(
                "Seeked to audio start position: {}",
                file.stream_position()?
            );
        }
        self.reader = Box::new(BufReader::with_capacity(self.frame_size as usize * 8, file));
        Ok(())
    }

    /// Returns true if we have reached the end of file for non-stdin inputs
    #[inline(always)]
    pub fn is_eof(&self) -> bool {
        !self.std_in && self.bytes_remaining <= 0
    }
}

impl Iterator for DsdIter {
    /// Item is (frame_read_size, Vec<channel_data_bytes>), where each channel_data_bytes is a boxed slice of bytes for a channel. The Vec of channels is in order (channel 1, channel 2, ...).
    type Item = (usize, Vec<Box<[u8]>>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.is_eof() {
            return None;
        }
        match self.load_frame() {
            Ok(read_size) => {
                self.retries = 0;
                if !self.std_in {
                    self.bytes_remaining -= read_size as u64;
                }
                return Some((read_size, self.channel_buffers.clone()));
            }
            Err(e) => {
                if let Some(io_err) = e.downcast_ref::<io::Error>() {
                    match io_err.kind() {
                        ErrorKind::Interrupted => {
                            if self.retries < RETRIES {
                                warn!("I/O interrupted, retrying read.");
                                self.retries += 1;
                                return self.next();
                            } else {
                                error!("Max retries reached for interrupted I/O.");
                                return None;
                            }
                        }
                        _ => {
                            return None;
                        }
                    }
                }
                error!("Error reading DSD frame: {}", e);
                return None;
            }
        }
    }
}

/// Bit-reverse a byte, to convert from MSB-first to LSB-first or vice versa.
#[inline]
pub fn bit_reverse_u8(mut b: u8) -> u8 {
    // Reverse bits in a byte (branchless, 3 shuffle steps)
    b = (b & 0xF0) >> 4 | (b & 0x0F) << 4;
    b = (b & 0xCC) >> 2 | (b & 0x33) << 2;
    b = (b & 0xAA) >> 1 | (b & 0x55) << 1;
    b
}
