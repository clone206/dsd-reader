use dsd_source::{DsdSource, DsdSourceError, DsdSourceExtensions, DsdSourceInfo};
use log::warn;
use std::{
    fs::File,
    io::{Read, Seek, SeekFrom},
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
pub use dsf_meta::DSF_BLOCK_SIZE;

/// A raw DSD file with no container metadata: a [`DsdSource`] whose
/// [`DsdSourceInfo`] has every format-specific field as `None`, and whose
/// audio data starts at byte 0.
struct RawSource {
    file: File,
    audio_length: u64,
}

impl DsdSource for RawSource {
    fn info(&self) -> Result<DsdSourceInfo, DsdSourceError> {
        Ok(DsdSourceInfo {
            channels: None,
            endianness: None,
            layout: None,
            block_size: None,
            sample_rate: None,
            audio_length: self.audio_length,
            data_offset: 0,
            tag: None,
        })
    }

    fn reader(&self) -> Result<Box<dyn Read + Send>, DsdSourceError> {
        let mut file = self.file.try_clone()?;
        file.seek(SeekFrom::Start(0))?;
        Ok(Box::new(file))
    }
}

/// Opens `path` as a [`DsdSource`], dispatching on `file_format`. Callers get
/// back a single trait object regardless of whether the input is a
/// container (DSF, DFF) or a raw file with no header — all format-specific
/// fields are queried uniformly via [`DsdSource`]'s getter methods.
pub fn open_source(
    path: &PathBuf,
    file_format: DsdFileFormat,
) -> Result<Box<dyn DsdSource>, Box<dyn std::error::Error>> {
    if file_format == DsdFileFormat::Dsf {
        use dsf_meta::DsfFile;
        let dsf_file = DsfFile::open(Path::new(&path))?;
        if let Some(e) = dsf_file.tag_read_err() {
            warn!(
                "Attempted read of ID3 tag failed. Partial read attempted: {}",
                e
            );
        }
        // Validate metadata parsed cleanly; on failure, let the caller fall
        // back to treating this as a raw file.
        dsf_file
            .info()
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        Ok(Box::new(dsf_file))
    } else if file_format == DsdFileFormat::Dsdiff {
        use dff_meta::DffFile;
        use dff_meta::model::*;
        let dff_file = match DffFile::open(Path::new(&path)) {
            Ok(dff) => dff,
            Err(Error::Id3Error(e, dff_file)) => {
                warn!(
                    "Attempted read of ID3 tag failed. Partial read attempted: {}",
                    e
                );
                dff_file
            }
            Err(e) => return Err(e.into()),
        };
        dff_file
            .info()
            .map_err(|e| -> Box<dyn std::error::Error> { e })?;
        Ok(Box::new(dff_file))
    } else if file_format == DsdFileFormat::Raw {
        let Ok(meta) = std::fs::metadata(path) else {
            return Err("Failed to read input file metadata".into());
        };
        Ok(Box::new(RawSource {
            file: File::open(path)?,
            audio_length: meta.len(),
        }))
    } else {
        Err("Unsupported file type; only dsf, dff, and raw dsd files are supported".into())
    }
}

