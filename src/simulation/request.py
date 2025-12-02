"""Request data structure for LLM inference simulation."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Request:
    """Represents a single LLM inference request."""

    request_id: int
    arrival_time: float
    prompt_length: int
    expected_output_length: int

    # Timestamps tracked during processing
    queue_entry_time: Optional[float] = None
    processing_start_time: Optional[float] = None
    processing_end_time: Optional[float] = None

    # Actual generated tokens (may differ from expected)
    actual_output_length: Optional[int] = None

    @property
    def queue_wait_time(self) -> Optional[float]:
        """Time spent waiting in queue."""
        if self.queue_entry_time and self.processing_start_time:
            return self.processing_start_time - self.queue_entry_time
        return None

    @property
    def processing_time(self) -> Optional[float]:
        """Time spent in processing (prefill + decode)."""
        if self.processing_start_time and self.processing_end_time:
            return self.processing_end_time - self.processing_start_time
        return None

    @property
    def total_latency(self) -> Optional[float]:
        """Total time from arrival to completion."""
        if self.arrival_time and self.processing_end_time:
            return self.processing_end_time - self.arrival_time
        return None

    @property
    def total_tokens(self) -> int:
        """Total tokens processed (prompt + output)."""
        output_len = self.actual_output_length if self.actual_output_length is not None else self.expected_output_length
        return self.prompt_length + output_len

    def __lt__(self, other: 'Request') -> bool:
        """Default comparison for priority queue (FCFS by arrival time)."""
        return self.arrival_time < other.arrival_time
