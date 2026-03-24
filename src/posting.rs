use bytemuck::{Pod, Zeroable};
use serde::{Deserialize, Serialize};

use crate::pq::{PqCode, PQ_BYTES};

pub type ChunkId = u32;

/// A single posting-list entry: chunk identifier + its PQ-compressed embedding.
/// Laid out as a flat 36-byte struct for cache-friendly scanning.
#[derive(Clone, Copy, Serialize, Deserialize)]
#[repr(C)]
pub struct VtplEntry {
    pub chunk_id: ChunkId,
    pub pq_code: PqCode,
}

unsafe impl Zeroable for VtplEntry {}
unsafe impl Pod for VtplEntry {}

impl VtplEntry {
    pub fn new(chunk_id: ChunkId, pq_code: PqCode) -> Self {
        Self { chunk_id, pq_code }
    }
}

/// A posting list for a single n-gram: a sorted vec of VtplEntry (sorted by chunk_id).
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct PostingList {
    pub entries: Vec<VtplEntry>,
}

impl PostingList {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    pub fn push(&mut self, entry: VtplEntry) {
        self.entries.push(entry);
    }

    /// Sort entries by chunk_id for merge-intersection.
    pub fn sort(&mut self) {
        self.entries.sort_unstable_by_key(|e| e.chunk_id);
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Byte overhead of storing PQ codes vs. a plain chunk-id-only posting list.
    pub fn pq_overhead_bytes(&self) -> usize {
        self.entries.len() * PQ_BYTES
    }
}
