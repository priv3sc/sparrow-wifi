"""
Tests for channel_scheduler.py.

All tests are hardware-independent — iw is mocked out so no physical
Wi-Fi adapter is required.
"""
import struct
import sys
import os
import threading
import time
import unittest
from unittest.mock import MagicMock, patch, call

# Make the sparrow_droneid package importable without installing it.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from sparrow_droneid.backend.channel_scheduler import (
    ChannelScheduler, ChannelMode,
    channel_to_freq, freq_to_channel,
    validate_channel_config, parse_channel_list,
    VALID_24GHZ_CHANNELS,
    _DEFAULT_CHANNELS, _DEFAULT_DWELL_MS, _DEFAULT_EXPIRY_S,
)


# ---------------------------------------------------------------------------
# Frequency / channel helpers
# ---------------------------------------------------------------------------

class TestFreqChannelHelpers(unittest.TestCase):

    def test_channel_to_freq_standard(self):
        self.assertEqual(channel_to_freq(1), 2412)
        self.assertEqual(channel_to_freq(6), 2437)
        self.assertEqual(channel_to_freq(11), 2462)
        self.assertEqual(channel_to_freq(13), 2472)

    def test_channel_to_freq_ch14(self):
        self.assertEqual(channel_to_freq(14), 2484)

    def test_freq_to_channel_roundtrip(self):
        for ch in range(1, 14):
            freq = channel_to_freq(ch)
            self.assertEqual(freq_to_channel(freq), ch)

    def test_freq_to_channel_none_for_5ghz(self):
        self.assertIsNone(freq_to_channel(5180))
        self.assertIsNone(freq_to_channel(5500))

    def test_freq_to_channel_none_for_mid_band(self):
        # Between two valid channel frequencies
        self.assertIsNone(freq_to_channel(2413))

    def test_freq_to_channel_ch14(self):
        self.assertEqual(freq_to_channel(2484), 14)

    def test_valid_channels_range(self):
        self.assertEqual(VALID_24GHZ_CHANNELS, list(range(1, 14)))


# ---------------------------------------------------------------------------
# validate_channel_config
# ---------------------------------------------------------------------------

class TestValidateChannelConfig(unittest.TestCase):

    def test_valid_fixed(self):
        errs = validate_channel_config('fixed', [6], 250, 300)
        self.assertEqual(errs, [])

    def test_valid_scan(self):
        errs = validate_channel_config('scan', [1, 6, 11], 250, 300)
        self.assertEqual(errs, [])

    def test_valid_adaptive(self):
        errs = validate_channel_config('adaptive', [1, 6, 11], 100, 60)
        self.assertEqual(errs, [])

    def test_invalid_mode(self):
        errs = validate_channel_config('turbo', [6], 250, 300)
        self.assertTrue(any('Invalid channel_mode' in e for e in errs))

    def test_empty_channel_list(self):
        errs = validate_channel_config('scan', [], 250, 300)
        self.assertTrue(any('channel_list must not be empty' in e for e in errs))

    def test_invalid_channel_number(self):
        errs = validate_channel_config('scan', [1, 99], 250, 300)
        self.assertTrue(any('99' in e for e in errs))

    def test_dwell_too_low(self):
        errs = validate_channel_config('scan', [1, 6, 11], 10, 300)
        self.assertTrue(any('dwell_ms' in e for e in errs))

    def test_dwell_too_high(self):
        errs = validate_channel_config('scan', [1, 6, 11], 60_000, 300)
        self.assertTrue(any('dwell_ms' in e for e in errs))

    def test_expiry_too_low(self):
        errs = validate_channel_config('adaptive', [1, 6], 250, 5)
        self.assertTrue(any('expiry_s' in e for e in errs))

    def test_expiry_too_high(self):
        errs = validate_channel_config('adaptive', [1, 6], 250, 100_000)
        self.assertTrue(any('expiry_s' in e for e in errs))

    def test_multiple_errors(self):
        errs = validate_channel_config('bad', [], 5, 1)
        self.assertGreater(len(errs), 1)


# ---------------------------------------------------------------------------
# parse_channel_list
# ---------------------------------------------------------------------------

class TestParseChannelList(unittest.TestCase):

    def test_basic(self):
        self.assertEqual(parse_channel_list('1,6,11'), [1, 6, 11])

    def test_spaces(self):
        self.assertEqual(parse_channel_list('1, 6, 11'), [1, 6, 11])

    def test_single(self):
        self.assertEqual(parse_channel_list('6'), [6])

    def test_trailing_comma(self):
        self.assertEqual(parse_channel_list('1,6,11,'), [1, 6, 11])

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            parse_channel_list('1,abc,11')

    def test_empty_string(self):
        self.assertEqual(parse_channel_list(''), [])


# ---------------------------------------------------------------------------
# ChannelScheduler — fixed mode
# ---------------------------------------------------------------------------

class TestChannelSchedulerFixed(unittest.TestCase):

    def _make(self, mode='fixed', channels=None, dwell_s=0.05):
        return ChannelScheduler(
            interface='wlan0mon',
            channels=channels or [6],
            dwell_s=dwell_s,
            mode=mode,
        )

    def test_initial_channel(self):
        s = self._make(channels=[6])
        self.assertEqual(s.get_current_channel(), 6)

    def test_start_does_not_launch_thread(self):
        s = self._make()
        s.start()
        # No background thread for fixed mode
        self.assertIsNone(s._thread)
        s.stop()

    def test_stop_idempotent(self):
        s = self._make()
        s.start()
        s.stop()
        s.stop()  # second stop must not raise

    def test_get_status_fixed(self):
        s = self._make(channels=[6])
        s.start()
        st = s.get_status()
        self.assertEqual(st['mode'], 'fixed')
        self.assertEqual(st['current_channel'], 6)
        self.assertEqual(st['channels'], [6])
        s.stop()

    def test_notify_detection_noop_in_fixed(self):
        s = self._make(channels=[6])
        s.start()
        s.notify_detection(6)
        self.assertEqual(s._active_channels, {})
        s.stop()

    def test_set_interface(self):
        s = self._make()
        s.set_interface('wlan1mon')
        self.assertEqual(s._interface, 'wlan1mon')


# ---------------------------------------------------------------------------
# ChannelScheduler — scan mode (iw mocked)
# ---------------------------------------------------------------------------

class TestChannelSchedulerScan(unittest.TestCase):

    def _make_scan(self, channels=None, dwell_s=0.05):
        return ChannelScheduler(
            interface='wlan0mon',
            channels=channels or [1, 6, 11],
            dwell_s=dwell_s,
            mode=ChannelMode.SCAN,
        )

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_builds_sequential_schedule(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        s = self._make_scan(channels=[1, 6, 11], dwell_s=0.05)
        schedule = s._build_schedule()
        self.assertEqual(schedule, [1, 6, 11])

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_changes_channel(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        s = self._make_scan(channels=[1, 6], dwell_s=0.05)
        s._change_channel(6)
        self.assertEqual(s.get_current_channel(), 6)
        mock_run.assert_called_once_with(
            ['iw', 'dev', 'wlan0mon', 'set', 'channel', '6'],
            capture_output=True, text=True, timeout=5,
        )

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_same_channel_skips_iw(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        s = self._make_scan(channels=[1, 6], dwell_s=0.05)
        s._change_channel(1)  # already on channel 1 (initial)
        mock_run.assert_not_called()

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_iw_failure_recorded(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stderr='Operation not permitted', stdout='')
        s = self._make_scan(channels=[1, 6], dwell_s=0.05)
        s._change_channel(6)
        self.assertEqual(s._error_count, 1)
        self.assertIn('6', s._last_error)
        self.assertIn('Operation not permitted', s._last_error)
        # Current channel must not have changed
        self.assertEqual(s.get_current_channel(), 1)

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_iw_timeout_recorded(self, mock_run):
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired(['iw'], timeout=5)
        s = self._make_scan(channels=[1, 6], dwell_s=0.05)
        s._change_channel(6)
        self.assertEqual(s._error_count, 1)
        self.assertIn('timed out', s._last_error)

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_iw_not_found_recorded(self, mock_run):
        mock_run.side_effect = FileNotFoundError
        s = self._make_scan(channels=[1, 6], dwell_s=0.05)
        s._change_channel(6)
        self.assertEqual(s._error_count, 1)
        self.assertIn('iw not found', s._last_error)

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_thread_starts_and_hops(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        s = self._make_scan(channels=[1, 6, 11], dwell_s=0.05)
        s.start()
        self.assertIsNotNone(s._thread)
        self.assertTrue(s._thread.is_alive())
        time.sleep(0.35)  # allow a couple of hops
        s.stop()
        self.assertIsNone(s._thread)
        # At least one channel change should have happened
        self.assertGreater(mock_run.call_count, 0)

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_scan_stop_clears_thread(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        s = self._make_scan(channels=[1, 6, 11], dwell_s=0.05)
        s.start()
        s.stop()
        self.assertIsNone(s._thread)

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_notify_detection_noop_in_scan(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        s = self._make_scan()
        s.notify_detection(6)
        self.assertEqual(s._active_channels, {})


# ---------------------------------------------------------------------------
# ChannelScheduler — adaptive mode
# ---------------------------------------------------------------------------

class TestChannelSchedulerAdaptive(unittest.TestCase):

    def _make_adaptive(self, channels=None, dwell_s=0.05, expiry_s=10.0):
        return ChannelScheduler(
            interface='wlan0mon',
            channels=channels or [1, 6, 11],
            dwell_s=dwell_s,
            mode=ChannelMode.ADAPTIVE,
            expiry_s=expiry_s,
            priority_factor=3,
        )

    def test_notify_detection_records_channel(self):
        s = self._make_adaptive()
        s.notify_detection(6)
        self.assertIn(6, s._active_channels)

    def test_notify_detection_multiple_channels(self):
        s = self._make_adaptive()
        s.notify_detection(1)
        s.notify_detection(6)
        self.assertIn(1, s._active_channels)
        self.assertIn(6, s._active_channels)

    def test_adaptive_schedule_active_channel_appears_more(self):
        s = self._make_adaptive(channels=[1, 6, 11])
        s.notify_detection(6)
        schedule = s._build_schedule()
        count_6 = schedule.count(6)
        count_1 = schedule.count(1)
        count_11 = schedule.count(11)
        # Channel 6 is active: should appear more than inactive channels
        self.assertGreater(count_6, count_1)
        self.assertGreater(count_6, count_11)
        # But inactive channels must still appear at least once
        self.assertGreater(count_1, 0)
        self.assertGreater(count_11, 0)

    def test_adaptive_schedule_without_detections_visits_all_once(self):
        s = self._make_adaptive(channels=[1, 6, 11])
        schedule = s._build_schedule()
        # No active channels: each channel appears exactly once
        self.assertEqual(schedule.count(1), 1)
        self.assertEqual(schedule.count(6), 1)
        self.assertEqual(schedule.count(11), 1)

    def test_adaptive_expiry_removes_priority(self):
        s = self._make_adaptive(channels=[1, 6, 11], expiry_s=0.1)
        s.notify_detection(6)
        time.sleep(0.2)
        schedule = s._build_schedule()
        # After expiry, channel 6 should appear exactly once (no priority)
        self.assertEqual(schedule.count(6), 1)
        # Also must not appear in _active_channels after _build_schedule expires it
        self.assertNotIn(6, s._active_channels)

    def test_get_status_active_channels(self):
        s = self._make_adaptive(expiry_s=300)
        s.notify_detection(6)
        st = s.get_status()
        self.assertIn(6, st['active_channels'])

    def test_get_status_expired_channels_hidden(self):
        s = self._make_adaptive(expiry_s=0.1)
        s.notify_detection(6)
        time.sleep(0.2)
        st = s.get_status()
        self.assertNotIn(6, st['active_channels'])

    @patch('sparrow_droneid.backend.channel_scheduler.subprocess.run')
    def test_adaptive_thread_starts(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stderr='', stdout='')
        s = self._make_adaptive(dwell_s=0.05)
        s.start()
        self.assertIsNotNone(s._thread)
        s.stop()


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------

class TestDefaultValues(unittest.TestCase):

    def test_default_channels(self):
        self.assertEqual(_DEFAULT_CHANNELS, [1, 6, 11])

    def test_default_dwell_ms(self):
        self.assertEqual(_DEFAULT_DWELL_MS, 250)

    def test_default_expiry_s(self):
        self.assertEqual(_DEFAULT_EXPIRY_S, 300)

    def test_scheduler_defaults_to_first_channel(self):
        s = ChannelScheduler(
            interface='wlan0mon', channels=[1, 6, 11],
            dwell_s=0.25, mode=ChannelMode.SCAN,
        )
        self.assertEqual(s.get_current_channel(), 1)


# ---------------------------------------------------------------------------
# Radiotap extraction — tested via droneid_engine helpers
# ---------------------------------------------------------------------------

class TestRadiotapChannelExtraction(unittest.TestCase):
    """
    Verify the static helpers that extract channel information from a
    radiotap header.  We build synthetic radiotap bytes that match the
    layout described in the radiotap spec.

    Layout (present mask = 0x00000008 → bit 3 = Channel):
        [0]     version = 0
        [1]     pad     = 0
        [2:4]   len     = 12 (LE) — header is 12 bytes
        [4:8]   present = 0x00000008 (bit 3 set → Channel field present)
        [8:10]  freq    = 2437 (LE) — channel 6
        [10:12] flags   = 0x0080 (LE) — 2.4 GHz OFDM
    """

    def _build_rt_header(self, freq: int, present: int = 0x00000008, length: int = 12) -> bytes:
        hdr = bytearray(length)
        hdr[0] = 0  # version
        hdr[1] = 0  # pad
        struct.pack_into('<H', hdr, 2, length)
        struct.pack_into('<I', hdr, 4, present)
        if present & (1 << 3):
            # Channel field starts at offset 8 (aligned to 2)
            struct.pack_into('<H', hdr, 8, freq)
            struct.pack_into('<H', hdr, 10, 0x0080)
        return bytes(hdr)

    def test_extract_channel_freq_ch6(self):
        from sparrow_droneid.backend.droneid_engine import DroneIDEngine
        rt = self._build_rt_header(freq=2437)
        freq = DroneIDEngine._extract_radiotap_channel_freq(rt)
        self.assertEqual(freq, 2437)

    def test_extract_channel_freq_ch1(self):
        from sparrow_droneid.backend.droneid_engine import DroneIDEngine
        rt = self._build_rt_header(freq=2412)
        freq = DroneIDEngine._extract_radiotap_channel_freq(rt)
        self.assertEqual(freq, 2412)

    def test_extract_channel_freq_absent(self):
        from sparrow_droneid.backend.droneid_engine import DroneIDEngine
        # present mask with no Channel bit
        rt = self._build_rt_header(freq=2437, present=0x00000020)  # bit 5 = RSSI only
        freq = DroneIDEngine._extract_radiotap_channel_freq(rt)
        self.assertEqual(freq, 0)

    def test_extract_channel_freq_too_short(self):
        from sparrow_droneid.backend.droneid_engine import DroneIDEngine
        freq = DroneIDEngine._extract_radiotap_channel_freq(b'\x00\x00\x04\x00')
        self.assertEqual(freq, 0)

    def test_freq_to_channel_mapping(self):
        self.assertEqual(freq_to_channel(2437), 6)
        self.assertEqual(freq_to_channel(2412), 1)
        self.assertEqual(freq_to_channel(2462), 11)


# ---------------------------------------------------------------------------
# Thread safety: concurrent notify_detection + _build_schedule
# ---------------------------------------------------------------------------

class TestConcurrency(unittest.TestCase):

    def test_concurrent_notify_and_build(self):
        """Hammer notify_detection from one thread while _build_schedule
        runs in another; should not raise or deadlock."""
        s = ChannelScheduler(
            interface='wlan0mon', channels=[1, 6, 11],
            dwell_s=0.01, mode=ChannelMode.ADAPTIVE, expiry_s=1.0,
        )
        errors = []

        def notifier():
            for _ in range(200):
                try:
                    s.notify_detection(6)
                    s.notify_detection(1)
                    time.sleep(0.001)
                except Exception as exc:
                    errors.append(exc)

        def builder():
            for _ in range(200):
                try:
                    s._build_schedule()
                    time.sleep(0.001)
                except Exception as exc:
                    errors.append(exc)

        t1 = threading.Thread(target=notifier)
        t2 = threading.Thread(target=builder)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        self.assertEqual(errors, [], f"Concurrent access raised: {errors}")


if __name__ == '__main__':
    unittest.main()
