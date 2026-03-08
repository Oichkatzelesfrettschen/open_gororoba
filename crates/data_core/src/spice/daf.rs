//! Double Precision Array File (DAF) parser.
//!
//! SPICE .bsp (SPK) files are stored in DAF format. A DAF consists of a 1024-byte
//! File Record, followed by Comment Records, followed by an array of 1024-byte
//! Summary Records that point to Name Records and the actual Double Precision (f64) arrays.

use memmap2::Mmap;
use std::{
    fs::File,
    io::{Error, ErrorKind, Result},
    path::Path,
};

/// A zero-copy reader for DAF files (like JPL .bsp ephemerides).
pub struct DafReader {
    mmap: Mmap,
    nd: usize,
    ni: usize,
    is_le: bool,
}

impl DafReader {
    /// Opens and memory-maps a DAF file.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 1024 {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "File too short to be DAF",
            ));
        }

        // Validate the DAF magic string "DAF/SPK" or "NAIF/DAF" (bytes 0..8)
        let magic = &mmap[0..8];
        let magic_str = std::str::from_utf8(magic).unwrap_or("");
        if !magic_str.starts_with("DAF/") && !magic_str.starts_with("NAIF/") {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Invalid DAF magic: {}", magic_str),
            ));
        }

        // Endianness detection
        let arch = std::str::from_utf8(&mmap[88..96]).unwrap_or("");
        let mut is_le = if arch.contains("LTL-IEEE") {
            true
        } else if arch.contains("BIG-IEEE") {
            false
        } else {
            // Heuristic for older files: check if ND/NI are reasonable in LE
            let nd_le = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
            let ni_le = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
            if nd_le > 100 || ni_le > 100 {
                // Probable BE if LE values are huge
                false
            } else {
                true
            }
        };

        // Final check: FWARD pointer at 76..80
        let fward_le = u32::from_le_bytes(mmap[76..80].try_into().unwrap());
        if fward_le as u64 * 1024 > mmap.len() as u64 {
            is_le = false;
        }

        let nd = if is_le {
            u32::from_le_bytes(mmap[8..12].try_into().unwrap()) as usize
        } else {
            u32::from_be_bytes(mmap[8..12].try_into().unwrap()) as usize
        };

        let ni = if is_le {
            u32::from_le_bytes(mmap[12..16].try_into().unwrap()) as usize
        } else {
            u32::from_be_bytes(mmap[12..16].try_into().unwrap()) as usize
        };

        Ok(Self {
            mmap,
            nd,
            ni,
            is_le,
        })
    }

    pub fn nd(&self) -> usize {
        self.nd
    }
    pub fn ni(&self) -> usize {
        self.ni
    }
    pub fn is_little_endian(&self) -> bool {
        self.is_le
    }

    /// Iterates over all summary arrays in the DAF file.
    pub fn arrays(&self) -> DafArrayIter<'_> {
        let le = self.is_le;
        let fward = if le {
            u32::from_le_bytes(self.mmap[76..80].try_into().unwrap()) as usize
        } else {
            u32::from_be_bytes(self.mmap[76..80].try_into().unwrap()) as usize
        };

        DafArrayIter {
            reader: self,
            current_record_block: fward,
            current_array_index: 0,
            arrays_in_block: 0,
        }
    }

    /// Gets a slice of f64 values given a start and end word (1-based indices in the DAF format).
    pub fn read_f64_slice(&self, start_word: usize, end_word: usize) -> Result<&[f64]> {
        let start_byte = (start_word - 1) * 8;
        let end_byte = end_word * 8;

        if end_byte > self.mmap.len() || start_byte >= end_byte {
            return Err(Error::new(
                ErrorKind::InvalidData,
                format!("Array bounds out of range: {}..{}", start_byte, end_byte),
            ));
        }

        let byte_slice = &self.mmap[start_byte..end_byte];
        let ptr = byte_slice.as_ptr() as *const f64;
        let len = byte_slice.len() / 8;

        Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
    }
}

#[derive(Debug, Clone)]
pub struct DafArray {
    pub doubles: Vec<f64>,
    pub integers: Vec<i32>,
}

pub struct DafArrayIter<'a> {
    reader: &'a DafReader,
    current_record_block: usize,
    current_array_index: usize,
    arrays_in_block: usize,
}

impl<'a> Iterator for DafArrayIter<'a> {
    type Item = DafArray;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_record_block == 0 {
            return None;
        }

        let block_start = (self.current_record_block - 1) * 1024;
        let le = self.reader.is_le;

        if self.current_array_index == 0 {
            self.arrays_in_block = self.read_f64_as_usize(block_start + 16, le);
            if self.arrays_in_block == 0 {
                self.current_record_block = self.read_f64_as_usize(block_start, le);
                if self.current_record_block == 0 {
                    return None;
                }
                return self.next();
            }
        }

        if self.current_array_index >= self.arrays_in_block {
            self.current_record_block = self.read_f64_as_usize(block_start, le);
            self.current_array_index = 0;
            return self.next();
        }

        let summary_size = (self.reader.nd + self.reader.ni.div_ceil(2)) * 8;
        let sum_start = block_start + 24 + self.current_array_index * summary_size;

        let mut doubles = Vec::with_capacity(self.reader.nd);
        for i in 0..self.reader.nd {
            let b = &self.reader.mmap[(sum_start + i * 8)..(sum_start + i * 8 + 8)];
            doubles.push(if le {
                f64::from_le_bytes(b.try_into().unwrap())
            } else {
                f64::from_be_bytes(b.try_into().unwrap())
            });
        }

        let int_start = sum_start + self.reader.nd * 8;
        let mut integers = Vec::with_capacity(self.reader.ni);
        for i in 0..self.reader.ni {
            let b = &self.reader.mmap[(int_start + i * 4)..(int_start + i * 4 + 4)];
            integers.push(if le {
                i32::from_le_bytes(b.try_into().unwrap())
            } else {
                i32::from_be_bytes(b.try_into().unwrap())
            });
        }

        self.current_array_index += 1;
        Some(DafArray { doubles, integers })
    }
}

impl<'a> DafArrayIter<'a> {
    fn read_f64_as_usize(&self, offset: usize, le: bool) -> usize {
        let b = &self.reader.mmap[offset..(offset + 8)];
        let val_f64 = if le {
            f64::from_le_bytes(b.try_into().unwrap())
        } else {
            f64::from_be_bytes(b.try_into().unwrap())
        };
        val_f64 as usize
    }
}
