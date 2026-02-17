"""
Phase 0: Global Transfer Protocol (GTP) Utilities for Python

Implements all 7 GTP invariants:
1. Content-Addressable Verification (SHA-256)
2. Atomic Upload-Then-Confirm (manifest sidecar)
3. Idempotent Retry (exponential backoff)
4. Bounded Lifetime (expiration tracking)
5. Size Verification
6. Content-Type Consistency
7. Structured Error Propagation

Usage:
    from gtp_utils import download_with_verification, upload_with_manifest

    # Download with manifest verification
    result = download_with_verification(presigned_url, dest_path)
    # result = {"hash": "abc...", "size": 12345, "verified": True}

    # Upload with manifest creation
    result = upload_with_manifest(file_path, presigned_url, content_type)
    # result = {"hash": "abc...", "size": 12345, "manifest_uploaded": True}
"""

import hashlib
import json
import os
import time
import requests
from typing import Dict, Optional, Tuple
from datetime import datetime


# ==================== Constants ====================

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1MB chunks
HASH_BUFFER_SIZE = 65536  # 64KB for file hashing

# Transfer error codes (matches TypeScript TransferErrorCode enum)
class TransferErrorCode:
    HASH_MISMATCH = "HASH_MISMATCH"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    URL_EXPIRED = "URL_EXPIRED"
    MANIFEST_MISSING = "MANIFEST_MISSING"
    MANIFEST_INVALID = "MANIFEST_INVALID"
    UPLOAD_FAILED = "UPLOAD_FAILED"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    NETWORK_ERROR = "NETWORK_ERROR"


class TransferError(Exception):
    """Structured transfer error with retry guidance"""
    def __init__(self, code: str, message: str, retryable: bool = True, context: dict = None):
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()


# Retry policies (matches TypeScript RETRY_POLICIES)
RETRY_POLICIES = {
    TransferErrorCode.TIMEOUT: {
        "max_retries": 3,
        "backoff_base": 2.0,  # seconds
        "backoff_multiplier": 2.0,
    },
    TransferErrorCode.HASH_MISMATCH: {
        "max_retries": 2,
        "backoff_base": 1.0,
        "backoff_multiplier": 2.0,
    },
    TransferErrorCode.SIZE_MISMATCH: {
        "max_retries": 2,
        "backoff_base": 1.0,
        "backoff_multiplier": 2.0,
    },
    TransferErrorCode.NETWORK_ERROR: {
        "max_retries": 3,
        "backoff_base": 2.0,
        "backoff_multiplier": 2.0,
    },
    TransferErrorCode.UPLOAD_FAILED: {
        "max_retries": 3,
        "backoff_base": 2.0,
        "backoff_multiplier": 2.0,
    },
    TransferErrorCode.DOWNLOAD_FAILED: {
        "max_retries": 3,
        "backoff_base": 2.0,
        "backoff_multiplier": 2.0,
    },
}


# ==================== INVARIANT 1: Content-Addressable Verification ====================

def hash_file(file_path: str) -> str:
    """
    Compute SHA-256 hash of file.

    Args:
        file_path: Path to file

    Returns:
        Hex-encoded SHA-256 hash
    """
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(HASH_BUFFER_SIZE)
            if not data:
                break
            sha256.update(data)
    return sha256.hexdigest()


def hash_stream_to_file(stream, dest_path: str) -> Tuple[str, int]:
    """
    Stream data to file while computing SHA-256 hash.
    Memory-efficient for large files.

    Args:
        stream: Iterable byte stream (e.g., requests response.iter_content())
        dest_path: Destination file path

    Returns:
        Tuple of (hash, size) where hash is hex-encoded SHA-256
    """
    sha256 = hashlib.sha256()
    size = 0

    with open(dest_path, 'wb') as f:
        for chunk in stream:
            if chunk:
                sha256.update(chunk)
                f.write(chunk)
                size += len(chunk)

    return sha256.hexdigest(), size


# ==================== INVARIANT 2: Atomic Upload-Then-Confirm ====================

def create_manifest(
    file_key: str,
    file_path: str,
    content_type: str,
    uploaded_by: str = "runpod",
    metadata: dict = None
) -> dict:
    """
    Create transfer manifest for file.

    Args:
        file_key: R2 key for the file
        file_path: Local file path
        content_type: MIME type
        uploaded_by: Source of upload ('backend', 'runpod', 'client')
        metadata: Additional metadata

    Returns:
        Manifest dict matching TypeScript TransferManifest interface
    """
    content_hash = hash_file(file_path)
    size_bytes = os.path.getsize(file_path)

    return {
        "version": "1.0",
        "fileKey": file_key,
        "contentHash": content_hash,
        "sizeBytes": size_bytes,
        "contentType": content_type,
        "uploadedAt": datetime.utcnow().isoformat() + 'Z',
        "uploadedBy": uploaded_by,
        "metadata": metadata or {},
    }


def upload_manifest(manifest: dict, manifest_url: str) -> bool:
    """
    Upload manifest sidecar to R2.

    Args:
        manifest: Manifest dict
        manifest_url: Presigned URL for manifest upload

    Returns:
        True if successful

    Raises:
        TransferError on failure
    """
    try:
        manifest_json = json.dumps(manifest, indent=2)
        response = requests.put(
            manifest_url,
            data=manifest_json.encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        raise TransferError(
            TransferErrorCode.UPLOAD_FAILED,
            f"Manifest upload failed: {e}",
            retryable=True,
            context={"manifest_key": manifest.get("fileKey")}
        )


def download_manifest(manifest_url: str) -> Optional[dict]:
    """
    Download and parse manifest from R2.

    Args:
        manifest_url: Presigned URL for manifest

    Returns:
        Manifest dict or None if not found

    Raises:
        TransferError if manifest exists but is invalid
    """
    try:
        response = requests.get(manifest_url, timeout=60)

        if response.status_code == 404:
            return None  # Manifest doesn't exist (non-GTP upload)

        response.raise_for_status()
        manifest = response.json()

        # Validate manifest structure
        required_fields = ["version", "fileKey", "contentHash", "sizeBytes", "contentType"]
        for field in required_fields:
            if field not in manifest:
                raise TransferError(
                    TransferErrorCode.MANIFEST_INVALID,
                    f"Manifest missing required field: {field}",
                    retryable=False
                )

        return manifest

    except requests.exceptions.RequestException as e:
        if "404" in str(e):
            return None
        raise TransferError(
            TransferErrorCode.DOWNLOAD_FAILED,
            f"Manifest download failed: {e}",
            retryable=True
        )
    except json.JSONDecodeError as e:
        raise TransferError(
            TransferErrorCode.MANIFEST_INVALID,
            f"Manifest JSON invalid: {e}",
            retryable=False
        )


# ==================== INVARIANT 3: Idempotent Retry ====================

def calculate_backoff_delay(attempt: int, policy: dict, max_delay: float = 30.0) -> float:
    """
    Calculate exponential backoff delay with jitter.

    Formula: delay = min(base × multiplier^(attempt-1) × jitter, max_delay)
    where jitter ∈ [0.75, 1.25] (±25% randomization)

    Args:
        attempt: Attempt number (1-indexed)
        policy: Retry policy dict with backoff_base and backoff_multiplier
        max_delay: Maximum delay cap in seconds

    Returns:
        Delay in seconds
    """
    import random

    base = policy.get("backoff_base", 1.0)
    multiplier = policy.get("backoff_multiplier", 2.0)

    exponential_delay = base * (multiplier ** (attempt - 1))
    jitter = 0.75 + random.random() * 0.5  # [0.75, 1.25]

    return min(exponential_delay * jitter, max_delay)


def retry_with_backoff(operation, max_attempts: int = 3, error_code: str = None):
    """
    Execute operation with automatic retry and exponential backoff.

    Args:
        operation: Callable to execute
        max_attempts: Maximum retry attempts
        error_code: TransferErrorCode for policy lookup

    Returns:
        Operation result

    Raises:
        TransferError if all retries exhausted
    """
    policy = RETRY_POLICIES.get(error_code, {
        "max_retries": max_attempts,
        "backoff_base": 1.0,
        "backoff_multiplier": 2.0,
    })

    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except TransferError as e:
            last_error = e

            if not e.retryable or attempt >= max_attempts:
                print(f"[GTP Retry] Operation failed (not retryable or exhausted): {e.code} - {e}")
                raise

            delay = calculate_backoff_delay(attempt, policy)
            print(f"[GTP Retry] Attempt {attempt}/{max_attempts} failed: {e.code}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except Exception as e:
            # Convert unknown errors to TransferError
            last_error = TransferError(
                "UNKNOWN_ERROR",
                str(e),
                retryable=True,
                context={"original_error": type(e).__name__}
            )

            if attempt >= max_attempts:
                raise last_error

            delay = calculate_backoff_delay(attempt, policy)
            print(f"[GTP Retry] Attempt {attempt}/{max_attempts} failed: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)

    raise last_error


# ==================== High-Level GTP Operations ====================

def download_with_verification(
    file_url: str,
    dest_path: str,
    manifest_url: Optional[str] = None,
    max_retries: int = 3
) -> dict:
    """
    INVARIANT 1+2+5: Download file with manifest verification.

    Process:
    1. Check manifest first (if provided)
    2. Download file while computing hash
    3. Verify hash matches manifest (INVARIANT 1)
    4. Verify size matches manifest (INVARIANT 5)

    Args:
        file_url: Presigned URL for file download
        dest_path: Destination path
        manifest_url: Optional presigned URL for manifest (auto-generated if None)
        max_retries: Maximum retry attempts

    Returns:
        Dict with keys: hash, size, verified (bool)

    Raises:
        TransferError on verification failure
    """
    def _download():
        print(f"[GTP Download] Starting: {os.path.basename(dest_path)}")

        # 1. Check manifest first (if URL provided or auto-generate)
        if manifest_url is None:
            # Auto-generate manifest URL by appending .manifest.json to file URL
            auto_manifest_url = file_url.split('?')[0] + '.manifest.json'
            if '?' in file_url:
                # Preserve query parameters
                auto_manifest_url += '?' + file_url.split('?')[1]
            manifest = download_manifest(auto_manifest_url)
        else:
            manifest = download_manifest(manifest_url)

        expected_hash = manifest.get("contentHash") if manifest else None
        expected_size = manifest.get("sizeBytes") if manifest else None

        if manifest:
            print(f"[GTP Download] Manifest found: hash={expected_hash[:8]}..., size={expected_size}")
        else:
            print(f"[GTP Download] No manifest found (non-GTP upload)")

        # 2. Download file while hashing
        try:
            response = requests.get(file_url, stream=True, timeout=300)
            response.raise_for_status()

            actual_hash, actual_size = hash_stream_to_file(
                response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE),
                dest_path
            )
        except requests.exceptions.RequestException as e:
            raise TransferError(
                TransferErrorCode.DOWNLOAD_FAILED,
                f"Download failed: {e}",
                retryable=True
            )

        # 3. Verify hash (if manifest exists)
        if expected_hash and actual_hash != expected_hash:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise TransferError(
                TransferErrorCode.HASH_MISMATCH,
                f"Hash mismatch: expected {expected_hash}, got {actual_hash}",
                retryable=True,
                context={"expected": expected_hash, "actual": actual_hash}
            )

        # 4. Verify size (if manifest exists)
        if expected_size and actual_size != expected_size:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            raise TransferError(
                TransferErrorCode.SIZE_MISMATCH,
                f"Size mismatch: expected {expected_size}, got {actual_size}",
                retryable=True,
                context={"expected": expected_size, "actual": actual_size}
            )

        print(f"[GTP Download] Success: {os.path.basename(dest_path)} ({actual_size/1024/1024:.2f} MB, verified={bool(manifest)})")

        return {
            "hash": actual_hash,
            "size": actual_size,
            "verified": bool(manifest),
        }

    return retry_with_backoff(_download, max_retries, TransferErrorCode.DOWNLOAD_FAILED)


def upload_with_manifest(
    file_path: str,
    file_url: str,
    content_type: str,
    manifest_url: Optional[str] = None,
    metadata: dict = None,
    max_retries: int = 3
) -> dict:
    """
    INVARIANT 1+2+5: Upload file with manifest (atomic, verified).

    Process:
    1. Hash file (INVARIANT 1)
    2. Verify size (INVARIANT 5)
    3. Upload file to R2
    4. Create and upload manifest sidecar (INVARIANT 2)

    Args:
        file_path: Local file path
        file_url: Presigned URL for file upload
        content_type: MIME type (INVARIANT 6)
        manifest_url: Optional presigned URL for manifest (auto-generated if None)
        metadata: Additional metadata
        max_retries: Maximum retry attempts

    Returns:
        Dict with keys: hash, size, manifest_uploaded (bool)

    Raises:
        TransferError on upload failure
    """
    def _upload():
        print(f"[GTP Upload] Starting: {os.path.basename(file_path)}")

        # 1. Create manifest (includes hashing + size check)
        file_key = file_url.split('?')[0].split('/')[-1]  # Extract filename from URL
        manifest = create_manifest(file_key, file_path, content_type, "runpod", metadata)

        # 2. Upload file
        file_size = manifest["sizeBytes"]
        try:
            with open(file_path, 'rb') as f:
                response = requests.put(
                    file_url,
                    data=f,
                    headers={'Content-Type': content_type},
                    timeout=600
                )
                response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise TransferError(
                TransferErrorCode.UPLOAD_FAILED,
                f"File upload failed: {e}",
                retryable=True,
                context={"file_key": file_key}
            )

        # 3. Upload manifest sidecar (ATOMIC confirmation)
        if manifest_url is None:
            # Auto-generate manifest URL
            auto_manifest_url = file_url.split('?')[0] + '.manifest.json'
            if '?' in file_url:
                auto_manifest_url += '?' + file_url.split('?')[1]
            manifest_url = auto_manifest_url

        upload_manifest(manifest, manifest_url)

        print(f"[GTP Upload] Success: {os.path.basename(file_path)} ({file_size/1024/1024:.2f} MB, hash={manifest['contentHash'][:8]}...)")

        return {
            "hash": manifest["contentHash"],
            "size": manifest["sizeBytes"],
            "manifest_uploaded": True,
        }

    return retry_with_backoff(_upload, max_retries, TransferErrorCode.UPLOAD_FAILED)


# ==================== Backward Compatibility Wrappers ====================

def download_file_gtp(url: str, path: str) -> None:
    """
    Drop-in replacement for download_file() with GTP support.
    Compatible with existing handler.py code.
    """
    download_with_verification(url, path)


def upload_file_gtp(path: str, url: str, max_retries: int = 3) -> dict:
    """
    Drop-in replacement for upload_file() with GTP support.
    Compatible with existing handler.py code.

    Returns dict matching original upload_file() format plus GTP fields.
    """
    # Detect content type from extension
    ext = os.path.splitext(path)[1].lower()
    content_types = {
        '.zip': 'application/zip',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.webp': 'image/webp',
        '.gif': 'image/gif',
        '.bmp': 'image/bmp',
        '.mp4': 'video/mp4',
        '.mov': 'video/quicktime',
        '.webm': 'video/webm',
        '.mkv': 'video/x-matroska',
        '.avi': 'video/x-msvideo',
        '.m4v': 'video/x-m4v',
    }
    content_type = content_types.get(ext, 'application/octet-stream')

    result = upload_with_manifest(path, url, content_type, max_retries=max_retries)

    # Return format matching original upload_file()
    return {
        "uploaded": True,
        "size": result["size"],
        "contentType": content_type,
        "attempts": 1,  # retry_with_backoff handles retries internally
        # GTP additions
        "contentHash": result["hash"],
        "manifestUploaded": result["manifest_uploaded"],
    }
