"""
Logging utilities for training.
"""
import logging
import sys
import time
from pathlib import Path
from typing import Optional


class TrainingLogger:
    """Training logger using Python's logging module."""

    def __init__(
        self,
        log_dir: Path,
        total_steps: int,
        exp_name: str = "",
        redirect_stdout: bool = True,
    ):
        """Initialize training logger.

        Args:
            log_dir: Directory to save log files.
            total_steps: Total training steps.
            exp_name: Experiment name.
            redirect_stdout: If True, also redirect stdout/stderr to console.log.
        """
        self.log_dir = log_dir
        self.log_file = log_dir / "train.log"
        self.total_steps = total_steps
        self._stdout_file = None
        self._stderr_file = None
        self._original_stdout = None
        self._original_stderr = None

        # Setup Python logger
        self.logger = logging.getLogger(exp_name or "train")
        self.logger.setLevel(logging.INFO)
        self.logger.handlers = []  # Clear existing handlers

        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                                       datefmt="%Y-%m-%d %H:%M:%S")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # Redirect stdout/stderr to console.log (captures tqdm, print, etc.)
        if redirect_stdout:
            self._setup_stdout_redirect()

        # Log start
        self.logger.info(f"Starting training: {exp_name}, total_steps={total_steps}")

        # Tracking state
        self.start_time = time.time()
        self.last_log_time = self.start_time
        self.last_log_step = 0

    def log(self, step: int, loss: float, lr: Optional[float] = None):
        """Log training metrics for current step."""
        current_time = time.time()

        # Compute speed (iterations per second)
        steps_since_last = step - self.last_log_step
        time_since_last = current_time - self.last_log_time
        if time_since_last > 0 and steps_since_last > 0:
            iter_per_sec = steps_since_last / time_since_last
        else:
            iter_per_sec = 0.0

        # Log message
        self.logger.info(
            f"step {step}, loss {loss}, lr {lr}, iter/s {iter_per_sec:.2f}"
        )

        self.last_log_time = current_time
        self.last_log_step = step

    def log_metric(self, name: str, value: float, step: int):
        """Log a custom metric."""
        self.logger.info(f"step {step}, {name} {value:.4f}")

    def _setup_stdout_redirect(self):
        """Redirect stdout/stderr to console.log while preserving terminal output."""
        console_log = self.log_dir / "console.log"
        self._stdout_file = open(console_log, "w")
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = _TeeWriter(sys.stdout, self._stdout_file)
        sys.stderr = _TeeWriter(sys.stderr, self._stdout_file)

    def finalize(self):
        """Log training completion and clean up."""
        elapsed = time.time() - self.start_time
        self.logger.info(f"Training complete, elapsed {elapsed:.1f}s")

        # Restore stdout/stderr
        if self._original_stdout is not None:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
        if self._stdout_file is not None:
            self._stdout_file.close()


class _TeeWriter:
    """Write to both terminal and file."""

    def __init__(self, terminal, file):
        self.terminal = terminal
        self.file = file

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.file.flush()

    def flush(self):
        self.terminal.flush()
        self.file.flush()

    def isatty(self):
        return self.terminal.isatty()
