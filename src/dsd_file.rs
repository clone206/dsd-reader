use dsd_source::{DsdSource, DsdSourceExtensions, Endianness, FmtType};
use id3::Tag;
use log::warn;
use std::{
    fs::File,
    path::{Path, PathBuf},
};

// Strongly typed container format
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DsdFileFormat {
    Dsdiff,
    Dsf,
    Raw,
}

pub trait FormatExtensions {
    fn is_container(&self) -> bool;
}
impl FormatExtensions for DsdFileFormat {
    fn is_container(&self) -> bool {
        match self {
            DsdFileFormat::Dsf | DsdFileFormat::Dsdiff => true,
            DsdFileFormat::Raw => false,
        }
    }
}

impl From<&PathBuf> for DsdFileFormat {
    fn from(path: &PathBuf) -> Self {
        let Some(ext) = path.extension() else {
            return DsdFileFormat::Raw;
        };
        let ext = ext.to_ascii_lowercase();
        let ext = ext.to_string_lossy();
        // Each format crate owns its own list of recognized extensions
        // (via `DsdSourceExtensions`), so adding a new container format
        // only requires adding a match arm here, not editing string
        // literals disconnected from the crate that actually defines them.
        if dsf_meta::DsfFile::EXTENSIONS.contains(&ext.as_ref()) {
            DsdFileFormat::Dsf
        } else if dff_meta::DffFile::EXTENSIONS.contains(&ext.as_ref()) {
            DsdFileFormat::Dsdiff
        } else {
            DsdFileFormat::Raw
        }
    }
}

pub use dff_meta::DFF_BLOCK_SIZE;
pub use dsd_source::DSD_64_RATE;
pub use dsf_meta::DSF_BLOCK_SIZE;

/// A DSD input opened from a file path: either a [`DsdSource`]-backed
/// container (DSF, DFF), or a raw file with no container metadata.
pub struct DsdFile {
    audio_length: u64,
    /// Byte offset where audio data begins; used only to sanity-check
    /// container-reported `audio_length` against the file's actual size.
    /// Always 0 for raw input.
    data_offset: u64,
    channel_count: Option<usize>,
    is_lsb: Option<bool>,
    layout: Option<FmtType>,
    block_size: Option<u32>,
    sample_rate: Option<u32>,
    container_format: DsdFileFormat,
    tag: Option<Tag>,
    source: Option<Box<dyn DsdSource>>,
    file: Option<File>,
}

impl DsdFile {
    pub fn audio_length(&self) -> u64 {
        self.audio_length
    }
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }
    pub fn tag(&self) -> Option<&Tag> {
        self.tag.as_ref()
    }
    pub fn channel_count(&self) -> Option<usize> {
        self.channel_count
    }
    pub fn is_lsb(&self) -> Option<bool> {
        self.is_lsb
    }
    /// Native channel layout (planar vs. interleaved), format-agnostic.
    pub fn layout(&self) -> Option<FmtType> {
        self.layout
    }
    pub fn block_size(&self) -> Option<u32> {
        self.block_size
    }
    pub fn sample_rate(&self) -> Option<u32> {
        self.sample_rate
    }
    pub fn container_format(&self) -> DsdFileFormat {
        self.container_format
    }
    /// Consumes `self`, returning the container's [`DsdSource`] (if any) and
    /// the raw file handle (present only for [`DsdFileFormat::Raw`] input).
    pub fn into_parts(self) -> (Option<Box<dyn DsdSource>>, Option<File>) {
        (self.source, self.file)
    }

    pub fn new(
        path: &PathBuf,
        file_format: DsdFileFormat,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if file_format == DsdFileFormat::Dsf {
            use dsf_meta::DsfFile;
            let file_path = Path::new(&path);
            let dsf_file = DsfFile::open(file_path)?;
            if let Some(e) = dsf_file.tag_read_err() {
                warn!(
                    "Attempted read of ID3 tag failed. Partial read attempted: {}",
                    e
                );
            }
            let info = dsf_file
                .info()
                .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            Ok(Self {
                sample_rate: Some(info.sample_rate),
                container_format: DsdFileFormat::Dsf,
                channel_count: Some(info.channels),
                is_lsb: Some(info.endianness == Endianness::LsbFirst),
                layout: Some(info.layout),
                block_size: Some(info.block_size),
                audio_length: info.audio_length,
                data_offset: info.data_offset,
                tag: info.tag,
                source: Some(Box::new(dsf_file)),
                file: None,
            })
        } else if file_format == DsdFileFormat::Dsdiff {
            use dff_meta::DffFile;
            use dff_meta::model::*;
            let file_path = Path::new(&path);
            let dff_file = match DffFile::open(file_path) {
                Ok(dff) => dff,
                Err(Error::Id3Error(e, dff_file)) => {
                    warn!(
                        "Attempted read of ID3 tag failed. Partial read attempted: {}",
                        e
                    );
                    dff_file
                }
                Err(e) => {
                    return Err(e.into());
                }
            };
            let info = dff_file
                .info()
                .map_err(|e| -> Box<dyn std::error::Error> { e })?;
            Ok(Self {
                sample_rate: Some(info.sample_rate),
                container_format: DsdFileFormat::Dsdiff,
                channel_count: Some(info.channels),
                is_lsb: Some(info.endianness == Endianness::LsbFirst),
                layout: Some(info.layout),
                block_size: Some(info.block_size),
                audio_length: info.audio_length,
                data_offset: info.data_offset,
                tag: info.tag,
                source: Some(Box::new(dff_file)),
                file: None,
            })
        } else if file_format == DsdFileFormat::Raw {
            let Ok(meta) = std::fs::metadata(path) else {
                return Err("Failed to read input file metadata".into());
            };
            Ok(Self {
                sample_rate: None,
                container_format: DsdFileFormat::Raw,
                channel_count: None,
                is_lsb: None,
                layout: None,
                block_size: None,
                audio_length: meta.len(),
                data_offset: 0,
                tag: None,
                source: None,
                file: Some(File::open(path)?),
            })
        } else {
            Err("Unsupported file type; only dsf, dff, and raw dsd files are supported"
                .into())
        }
    }
}
