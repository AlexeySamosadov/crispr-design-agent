"""Audit logging for API requests and responses."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class AuditEntry(BaseModel):
    """Audit log entry model."""

    timestamp: str
    request_id: str
    endpoint: str
    method: str
    user_id: Optional[str] = None
    request_data: Dict[str, Any]
    response_data: Optional[Dict[str, Any]] = None
    status_code: int
    duration_ms: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = {}


class AuditLogger:
    """Audit logger for API requests."""

    def __init__(
        self,
        log_dir: Path = Path("logs/audit"),
        enable_file_logging: bool = True,
        enable_console_logging: bool = False,
        hash_sequences: bool = True,
    ):
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for audit log files
            enable_file_logging: Write audit logs to files
            enable_console_logging: Write audit logs to console
            hash_sequences: Hash sequence data for privacy
        """
        self.log_dir = log_dir
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.hash_sequences = hash_sequences

        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self._setup_file_handler()

    def _setup_file_handler(self):
        """Setup file handler for audit logs."""
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.INFO)
        self.file_handler.setFormatter(logging.Formatter("%(message)s"))

        audit_logger = logging.getLogger("audit")
        audit_logger.addHandler(self.file_handler)
        audit_logger.setLevel(logging.INFO)

    def _hash_sequence(self, sequence: str) -> str:
        """Hash a sequence for privacy."""
        return hashlib.sha256(sequence.encode()).hexdigest()[:16]

    def _sanitize_request_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize request data for logging."""
        sanitized = data.copy()

        if self.hash_sequences and "sequence" in sanitized:
            original = sanitized["sequence"]
            sanitized["sequence_hash"] = self._hash_sequence(original)
            sanitized["sequence_length"] = len(original)
            del sanitized["sequence"]

        if "sequences" in sanitized and isinstance(sanitized["sequences"], list):
            original_sequences = sanitized["sequences"]
            sanitized["sequence_hashes"] = [self._hash_sequence(seq) for seq in original_sequences]
            sanitized["sequence_lengths"] = [len(seq) for seq in original_sequences]
            del sanitized["sequences"]

        return sanitized

    def log_request(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        request_data: Dict[str, Any],
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Log an API request.

        Args:
            request_id: Unique request identifier
            endpoint: API endpoint path
            method: HTTP method
            request_data: Request payload
            user_id: Optional user identifier
            metadata: Optional additional metadata

        Returns:
            Request start time (for duration calculation)
        """
        return time.time()

    def log_response(
        self,
        request_id: str,
        endpoint: str,
        method: str,
        request_data: Dict[str, Any],
        response_data: Optional[Dict[str, Any]],
        status_code: int,
        start_time: float,
        user_id: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Log an API response.

        Args:
            request_id: Unique request identifier
            endpoint: API endpoint path
            method: HTTP method
            request_data: Request payload
            response_data: Response payload
            status_code: HTTP status code
            start_time: Request start time
            user_id: Optional user identifier
            error: Optional error message
            metadata: Optional additional metadata
        """
        duration_ms = (time.time() - start_time) * 1000

        entry = AuditEntry(
            timestamp=datetime.now().isoformat(),
            request_id=request_id,
            endpoint=endpoint,
            method=method,
            user_id=user_id,
            request_data=self._sanitize_request_data(request_data),
            response_data=response_data,
            status_code=status_code,
            duration_ms=duration_ms,
            error=error,
            metadata=metadata or {},
        )

        log_message = entry.json()

        if self.enable_file_logging:
            audit_logger = logging.getLogger("audit")
            audit_logger.info(log_message)

        if self.enable_console_logging:
            logger.info(f"Audit: {log_message}")

    def get_stats(self, since: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Args:
            since: Optional start time for stats

        Returns:
            Dictionary of statistics
        """
        if not self.enable_file_logging:
            return {"error": "File logging not enabled"}

        stats = {
            "total_requests": 0,
            "endpoints": {},
            "status_codes": {},
            "avg_duration_ms": 0.0,
            "errors": 0,
        }

        log_files = sorted(self.log_dir.glob("audit_*.jsonl"))
        total_duration = 0.0

        for log_file in log_files:
            with open(log_file, "r") as f:
                for line in f:
                    try:
                        entry_dict = json.loads(line)
                        entry = AuditEntry(**entry_dict)

                        if since and datetime.fromisoformat(entry.timestamp) < since:
                            continue

                        stats["total_requests"] += 1
                        total_duration += entry.duration_ms

                        endpoint = entry.endpoint
                        stats["endpoints"][endpoint] = stats["endpoints"].get(endpoint, 0) + 1

                        status = str(entry.status_code)
                        stats["status_codes"][status] = stats["status_codes"].get(status, 0) + 1

                        if entry.error:
                            stats["errors"] += 1

                    except Exception as e:
                        logger.warning(f"Failed to parse audit entry: {e}")

        if stats["total_requests"] > 0:
            stats["avg_duration_ms"] = total_duration / stats["total_requests"]

        return stats
