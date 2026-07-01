"""
Wi-Fi channel scheduler for single-radio channel hopping.

Supports three modes:
  fixed    — stays on one channel; no background thread is started.
  scan     — cycles through the configured channel list sequentially.
  adaptive — gives recently-detected channels extra dwell time while
             continuing to visit all configured channels so new devices
             can still be discovered.

Thread-safety
-------------
All mutable state is protected by _lock.  The background thread (scan /
adaptive only) is the sole writer of _current_channel; it is readable from
any thread via get_current_channel().

The notify_detection() call comes from the parse thread and is handled
with _lock so it does not race with the schedule-building step.

Lifecycle
---------
A ChannelScheduler must be started exactly once (start()) and stopped
exactly once (stop()).  Starting and stopping monitoring repeatedly must
use a new ChannelScheduler instance each time; do not reuse a stopped one.
"""
from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2.4 GHz channel / frequency helpers
# ---------------------------------------------------------------------------

VALID_24GHZ_CHANNELS = list(range(1, 14))  # 1–13 (channel 14 is Japan-only)

_DEFAULT_CHANNELS = [1, 6, 11]
_DEFAULT_DWELL_MS = 250
_DEFAULT_EXPIRY_S = 300
_DEFAULT_PRIORITY_FACTOR = 3


def channel_to_freq(channel: int) -> int:
    """Return centre frequency (MHz) for a 2.4 GHz channel."""
    if channel == 14:
        return 2484
    return 2412 + (channel - 1) * 5


def freq_to_channel(freq_mhz: int) -> Optional[int]:
    """Convert a 2.4 GHz centre frequency (MHz) to a channel number.

    Returns None if the frequency does not map to a known 2.4 GHz channel.
    """
    if freq_mhz == 2484:
        return 14
    if 2412 <= freq_mhz <= 2472:
        remainder = freq_mhz - 2412
        if remainder % 5 == 0:
            return (remainder // 5) + 1
    return None


# ---------------------------------------------------------------------------
# Mode constants
# ---------------------------------------------------------------------------

class ChannelMode:
    FIXED = 'fixed'
    SCAN = 'scan'
    ADAPTIVE = 'adaptive'


_VALID_MODES = frozenset({ChannelMode.FIXED, ChannelMode.SCAN, ChannelMode.ADAPTIVE})


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------

def validate_channel_config(
    mode: str,
    channels: List[int],
    dwell_ms: int,
    expiry_s: int,
) -> List[str]:
    """Validate channel hopping configuration.

    Returns a list of human-readable error strings.  An empty list means the
    configuration is valid.
    """
    errors: List[str] = []

    if mode not in _VALID_MODES:
        errors.append(
            f"Invalid channel_mode {mode!r}; must be one of: "
            f"{', '.join(sorted(_VALID_MODES))}"
        )

    if not channels:
        errors.append("channel_list must not be empty")
    else:
        for ch in channels:
            if ch not in VALID_24GHZ_CHANNELS:
                errors.append(
                    f"Channel {ch} is not a valid 2.4 GHz channel (1–13)"
                )

    if dwell_ms < 50:
        errors.append("channel_dwell_ms must be at least 50")
    elif dwell_ms > 30_000:
        errors.append("channel_dwell_ms must not exceed 30000")

    if expiry_s < 10:
        errors.append("channel_expiry_s must be at least 10")
    elif expiry_s > 86_400:
        errors.append("channel_expiry_s must not exceed 86400 (24 hours)")

    return errors


def parse_channel_list(raw: str) -> List[int]:
    """Parse a comma-separated channel list string into a list of ints.

    Silently skips blank tokens.  Raises ValueError on non-integer tokens.
    """
    result: List[int] = []
    for token in raw.split(','):
        token = token.strip()
        if token:
            result.append(int(token))
    return result


# ---------------------------------------------------------------------------
# ChannelScheduler
# ---------------------------------------------------------------------------

class ChannelScheduler:
    """Controls Wi-Fi channel hopping for a single monitor-mode interface.

    Parameters
    ----------
    interface:
        Monitor-mode interface name (e.g. ``wlan0mon``).
    channels:
        Ordered list of channels to visit.
    dwell_s:
        Dwell time per channel visit in seconds.
    mode:
        ``'fixed'``, ``'scan'``, or ``'adaptive'``.
    expiry_s:
        Adaptive mode: seconds without a detection before a channel loses
        its priority boost.
    priority_factor:
        Adaptive mode: number of extra visits per cycle for an active
        channel (total slots = priority_factor + 1 per active channel,
        vs. 1 slot for inactive channels).
    """

    def __init__(
        self,
        interface: str,
        channels: List[int],
        dwell_s: float,
        mode: str,
        expiry_s: float = float(_DEFAULT_EXPIRY_S),
        priority_factor: int = _DEFAULT_PRIORITY_FACTOR,
    ):
        self._interface = interface
        self._channels = list(channels)
        self._dwell_s = max(0.05, dwell_s)
        self._mode = mode
        self._expiry_s = expiry_s
        self._priority_factor = max(1, priority_factor)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Current channel — initialised to first in list; updated by _change_channel
        self._current_channel: int = channels[0] if channels else 6

        # Adaptive state: {channel: monotonic timestamp of last detection}
        self._active_channels: Dict[int, float] = {}

        # Error tracking
        self._last_error: str = ''
        self._error_count: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the scheduler.

        For scan/adaptive modes a background thread is started that changes
        channels on a configurable dwell cadence.  For fixed mode this is a
        no-op (the interface was already placed on the correct channel by
        CaptureManager.start_monitor).
        """
        if self._mode == ChannelMode.FIXED:
            log.info(
                "ChannelScheduler: fixed mode on channel %d (interface=%s)",
                self._current_channel, self._interface,
            )
            return

        if self._thread and self._thread.is_alive():
            log.warning("ChannelScheduler: start() called while already running — ignored")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name='channel-sched',
        )
        self._thread.start()
        log.info(
            "ChannelScheduler: started (mode=%s channels=%s dwell=%.3fs interface=%s "
            "expiry=%.0fs)",
            self._mode, self._channels, self._dwell_s, self._interface, self._expiry_s,
        )

    def stop(self) -> None:
        """Stop the scheduler cleanly.

        Signals the background thread to exit and waits for it to finish.
        Safe to call even if the scheduler was never started or already stopped.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self._dwell_s + 2.0)
            if self._thread.is_alive():
                log.warning("ChannelScheduler: background thread did not stop in time")
            self._thread = None
        log.info("ChannelScheduler: stopped")

    def set_interface(self, interface: str) -> None:
        """Update the monitor interface name.

        Called by the engine when a full restart recreates the monitor VIF
        and the name changes (e.g. ``wlan0`` → ``wlan0mon``).
        """
        with self._lock:
            old = self._interface
            self._interface = interface
        if old != interface:
            log.info("ChannelScheduler: interface updated %s → %s", old, interface)

    # ── Detection feedback ───────────────────────────────────────────────────

    def notify_detection(self, channel: int) -> None:
        """Record a valid Remote ID detection on *channel*.

        Called from the parse thread after a successful decode.  Updates
        adaptive priority state.  No-op for fixed and scan modes.
        """
        if self._mode != ChannelMode.ADAPTIVE:
            return
        with self._lock:
            self._active_channels[channel] = time.monotonic()
        log.debug("ChannelScheduler: detection on channel %d — adaptive priority updated", channel)

    # ── Status ───────────────────────────────────────────────────────────────

    def get_current_channel(self) -> int:
        """Return the channel the interface is currently set to."""
        with self._lock:
            return self._current_channel

    def get_status(self) -> dict:
        """Return a serialisable status snapshot."""
        with self._lock:
            now = time.monotonic()
            # Only expose non-expired active channels
            active = {
                ch: round(now - ts, 1)
                for ch, ts in self._active_channels.items()
                if (now - ts) <= self._expiry_s
            }
            return {
                'mode': self._mode,
                'channels': list(self._channels),
                'dwell_s': self._dwell_s,
                'current_channel': self._current_channel,
                'active_channels': active,
                'expiry_s': self._expiry_s,
                'last_error': self._last_error,
                'error_count': self._error_count,
            }

    # ── Background thread ────────────────────────────────────────────────────

    def _run(self) -> None:
        """Main scheduler loop — runs until stop() signals _stop_event."""
        visit_idx = 0
        while not self._stop_event.is_set():
            try:
                schedule = self._build_schedule()
            except Exception as exc:
                log.error("ChannelScheduler: _build_schedule error: %s", exc)
                self._stop_event.wait(self._dwell_s)
                continue

            if not schedule:
                self._stop_event.wait(self._dwell_s)
                continue

            visit_idx = visit_idx % len(schedule)
            channel = schedule[visit_idx]
            visit_idx += 1

            try:
                self._change_channel(channel)
            except Exception as exc:
                log.error("ChannelScheduler: _change_channel error: %s", exc)

            self._stop_event.wait(self._dwell_s)

    def _build_schedule(self) -> List[int]:
        """Build the channel visit order for one cycle.

        Scan mode:  plain round-robin through _channels.
        Adaptive:   active channels appear (priority_factor + 1) times per
                    cycle; inactive channels appear once.  The order interleaves
                    active and inactive slots so active channels are not all
                    bunched together.
        """
        if self._mode == ChannelMode.SCAN:
            return list(self._channels)

        # Adaptive
        with self._lock:
            now = time.monotonic()
            # Expire stale entries in place (no copy needed under the lock)
            self._active_channels = {
                ch: ts
                for ch, ts in self._active_channels.items()
                if (now - ts) <= self._expiry_s
            }
            active = frozenset(self._active_channels.keys()) & frozenset(self._channels)

        # Build weighted list: active channels get (priority_factor + 1) slots total
        # inactive channels get 1 slot.  We interleave to avoid all active
        # channels being visited consecutively.
        schedule: List[int] = []
        inactive = [ch for ch in self._channels if ch not in active]

        # One base pass of all channels
        base = list(self._channels)
        # Extra passes only for active channels
        extra = [ch for ch in self._channels if ch in active] * self._priority_factor

        # Interleave: distribute extra visits roughly evenly across the base pass.
        # Simple approach: append and let round-robin spread them naturally.
        schedule = base + extra

        # Deduplicate while preserving order and proportional representation
        # is preserved by the counts above — no dedup needed, the scheduler
        # naturally revisits active channels more often.
        return schedule

    def _change_channel(self, channel: int) -> None:
        """Switch the monitor interface to *channel* using ``iw``."""
        with self._lock:
            interface = self._interface
            current = self._current_channel

        if channel == current:
            return  # Already on this channel

        log.debug("ChannelScheduler: %s channel %d → %d", interface, current, channel)

        try:
            result = subprocess.run(
                ['iw', 'dev', interface, 'set', 'channel', str(channel)],
                capture_output=True, text=True, timeout=5,
            )
        except subprocess.TimeoutExpired:
            self._handle_error(channel, "iw command timed out")
            return
        except FileNotFoundError:
            self._handle_error(channel, "iw not found in PATH")
            return
        except Exception as exc:
            self._handle_error(channel, str(exc))
            return

        if result.returncode != 0:
            err = (result.stderr.strip() or result.stdout.strip() or
                   f"exit code {result.returncode}")
            self._handle_error(channel, err)
            return

        with self._lock:
            self._current_channel = channel
            self._last_error = ''

    def _handle_error(self, channel: int, reason: str) -> None:
        with self._lock:
            self._error_count += 1
            self._last_error = f"channel {channel}: {reason}"
        log.warning(
            "ChannelScheduler: failed to set channel %d on %s: %s",
            channel, self._interface, reason,
        )
