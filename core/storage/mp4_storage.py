"""
MP4 Storage System for MCP v5
Implements vector storage using MP4 container format with custom boxes
"""

import json
import mmap
import hashlib
import struct
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class VirtualChunk:
    """
    Virtual chunk that references text in source MD files without duplication
    Only stores metadata and vector offset, not the actual text
    """
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    vector_offset: int
    vector_size: int
    section: str = ""
    summary: str = ""
    text_hash: str = ""
    
    def get_text(self) -> str:
        """Read actual text from source MD file on-demand"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                text = ''.join(lines[self.start_line:self.end_line + 1])
                return text.strip()
        except (IOError, IndexError) as e:
            logger.error(f"Error reading chunk text from {self.file_path}: {e}")
            return ""
    
    def compute_hash(self) -> str:
        """Compute hash of the text for integrity checking"""
        text = self.get_text()
        return hashlib.sha256(text.encode()).hexdigest()
    
    def validate_integrity(self) -> bool:
        """Verify chunk integrity against source file"""
        if not Path(self.file_path).exists():
            return False
        
        current_hash = self.compute_hash()
        if self.text_hash and current_hash != self.text_hash:
            logger.warning(f"Chunk {self.chunk_id} hash mismatch - file may have been modified")
            return False
        
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VirtualChunk':
        """Create from dictionary"""
        return cls(**data)


class MP4Storage:
    """
    MP4-based storage for vectors and metadata
    Uses custom boxes within ISO BMFF (MP4) container
    """
    
    def __init__(self, mp4_path: str):
        self.mp4_path = Path(mp4_path)
        self.mp4_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.chunks: List[VirtualChunk] = []
        self.metadata: Dict = {}
        self.mmap_handle = None
        self.mmap_data = None
        
        logger.info(f"Initialized MP4Storage at {self.mp4_path}")
    
    def create_snapshot(self, chunks: List[VirtualChunk], vectors_blob: bytes, 
                       hnsw_blob: bytes, metadata: Dict) -> str:
        """
        Create new MP4 snapshot with vectors and index
        
        Args:
            chunks: List of VirtualChunk objects
            vectors_blob: Binary blob of all vectors
            hnsw_blob: Serialized HNSW index
            metadata: Additional metadata
        
        Returns:
            Snapshot hash
        """
        logger.info(f"Creating MP4 snapshot with {len(chunks)} chunks")
        
        # Prepare index data
        index_data = {
            'version': metadata.get('version', '5.0.0'),
            'snapshot_hash': '',  # Will be computed
            'embedding_model': metadata.get('embedding_model', 'unknown'),
            'chunks': [chunk.to_dict() for chunk in chunks],
            'created_at': metadata.get('created_at', ''),
            'total_vectors': len(chunks),
            'vector_dimension': metadata.get('vector_dimension', 384)
        }
        
        # Compute snapshot hash
        content_for_hash = json.dumps(index_data, sort_keys=True) + str(len(vectors_blob))
        snapshot_hash = hashlib.sha256(content_for_hash.encode()).hexdigest()
        index_data['snapshot_hash'] = snapshot_hash
        
        # Write MP4 structure
        self._write_mp4_structure(index_data, vectors_blob, hnsw_blob)
        
        self.chunks = chunks
        self.metadata = index_data
        
        logger.info(f"Snapshot created: {snapshot_hash}")
        return snapshot_hash
    
    def _write_mp4_structure(self, index_data: Dict, vectors_blob: bytes, hnsw_blob: bytes):
        """
        Write MP4 file structure with custom boxes
        
        Structure:
            ftyp - File type box
            moov - Movie box containing metadata
                udta - User data box
                    mcpi - Custom box for index JSON
            mdat - Media data box containing vectors and HNSW index
        """
        with open(self.mp4_path, 'wb') as f:
            # Write ftyp box (file type)
            ftyp = self._create_ftyp_box()
            f.write(ftyp)
            
            # Write moov box with index JSON in udta
            index_json = json.dumps(index_data, indent=2).encode('utf-8')
            moov = self._create_moov_box(index_json)
            f.write(moov)
            
            # Write mdat box with vectors and HNSW
            mdat = self._create_mdat_box(vectors_blob, hnsw_blob)
            f.write(mdat)
        
        logger.info(f"MP4 structure written to {self.mp4_path}")
    
    def _create_ftyp_box(self) -> bytes:
        """Create ftyp (file type) box"""
        major_brand = b'mcpv'  # Custom brand for MCP v5
        minor_version = struct.pack('>I', 1)
        compatible_brands = b'mcpvisom'
        
        data = major_brand + minor_version + compatible_brands
        size = len(data) + 8
        
        return struct.pack('>I', size) + b'ftyp' + data
    
    def _create_moov_box(self, index_json: bytes) -> bytes:
        """Create moov box with user data containing index"""
        # Create udta box
        udta_data = self._create_udta_box(index_json)
        udta_size = len(udta_data) + 8
        udta = struct.pack('>I', udta_size) + b'udta' + udta_data
        
        # Wrap in moov
        moov_size = len(udta) + 8
        return struct.pack('>I', moov_size) + b'moov' + udta
    
    def _create_udta_box(self, index_json: bytes) -> bytes:
        """Create udta box with custom mcpi box for index"""
        mcpi_size = len(index_json) + 8
        mcpi = struct.pack('>I', mcpi_size) + b'mcpi' + index_json
        return mcpi
    
    def _create_mdat_box(self, vectors_blob: bytes, hnsw_blob: bytes) -> bytes:
        """Create mdat box containing vectors and HNSW index"""
        # Combine vectors and HNSW with separator
        separator = struct.pack('>Q', len(vectors_blob))  # 8 bytes for vector blob size
        data = separator + vectors_blob + hnsw_blob
        
        size = len(data) + 8
        return struct.pack('>I', size) + b'mdat' + data
    
    def load_snapshot(self) -> bool:
        """
        Load existing MP4 snapshot
        
        Returns:
            True if loaded successfully
        """
        if not self.mp4_path.exists():
            logger.warning(f"MP4 file not found: {self.mp4_path}")
            return False
        
        try:
            index_data = self._read_index_from_mp4()
            if not index_data:
                return False
            
            self.metadata = index_data
            self.chunks = [VirtualChunk.from_dict(c) for c in index_data['chunks']]
            
            # Open mmap for vector access
            self._open_mmap()
            
            logger.info(f"Loaded snapshot with {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error loading snapshot: {e}")
            return False
    
    def _read_index_from_mp4(self) -> Optional[Dict]:
        """Read index JSON from MP4 udta box"""
        try:
            with open(self.mp4_path, 'rb') as f:
                data = f.read()
            
            # Find mcpi box (simplified parsing)
            mcpi_pos = data.find(b'mcpi')
            if mcpi_pos == -1:
                logger.error("mcpi box not found in MP4")
                return None
            
            # Read size of mcpi box (4 bytes before 'mcpi')
            size_pos = mcpi_pos - 4
            mcpi_size = struct.unpack('>I', data[size_pos:mcpi_pos])[0]
            
            # Extract JSON data (after 'mcpi' marker)
            json_start = mcpi_pos + 4
            json_end = size_pos + mcpi_size
            json_data = data[json_start:json_end]
            
            return json.loads(json_data.decode('utf-8'))
            
        except Exception as e:
            logger.error(f"Error reading index: {e}")
            return None
    
    def _open_mmap(self):
        """Open memory map for efficient vector access"""
        try:
            self.mmap_handle = open(self.mp4_path, 'rb')
            self.mmap_data = mmap.mmap(self.mmap_handle.fileno(), 0, access=mmap.ACCESS_READ)
            logger.info("Memory map opened for vector access")
        except Exception as e:
            logger.error(f"Error opening mmap: {e}")
    
    def get_vector_blob_offset(self) -> Tuple[int, int]:
        """
        Get offset and size of vector blob in mdat
        
        Returns:
            (offset, size) tuple
        """
        try:
            with open(self.mp4_path, 'rb') as f:
                data = f.read()
            
            # Find mdat box
            mdat_pos = data.find(b'mdat')
            if mdat_pos == -1:
                return (0, 0)
            
            # Read mdat size
            size_pos = mdat_pos - 4
            mdat_size = struct.unpack('>I', data[size_pos:mdat_pos])[0]
            
            # Vector blob starts after 8-byte separator
            data_start = mdat_pos + 4 + 8
            
            # Read separator to get vector blob size
            separator = struct.unpack('>Q', data[mdat_pos + 4:mdat_pos + 12])[0]
            
            return (data_start, separator)
            
        except Exception as e:
            logger.error(f"Error getting vector blob offset: {e}")
            return (0, 0)
    
    def get_hnsw_blob_offset(self) -> Tuple[int, int]:
        """
        Get offset and size of HNSW blob in mdat
        
        Returns:
            (offset, size) tuple
        """
        try:
            vec_offset, vec_size = self.get_vector_blob_offset()
            hnsw_offset = vec_offset + vec_size
            
            # HNSW size is mdat_size - vec_size - overhead
            with open(self.mp4_path, 'rb') as f:
                data = f.read()
            
            mdat_pos = data.find(b'mdat')
            size_pos = mdat_pos - 4
            mdat_size = struct.unpack('>I', data[size_pos:mdat_pos])[0]
            
            hnsw_size = mdat_size - vec_size - 16  # 8 for box header, 8 for separator
            
            return (hnsw_offset, hnsw_size)
            
        except Exception as e:
            logger.error(f"Error getting HNSW blob offset: {e}")
            return (0, 0)
    
    def close(self):
        """Close memory map and file handles"""
        if self.mmap_data:
            self.mmap_data.close()
        if self.mmap_handle:
            self.mmap_handle.close()
        logger.info("MP4 storage closed")
    
    def __del__(self):
        """Ensure resources are cleaned up"""
        self.close()
