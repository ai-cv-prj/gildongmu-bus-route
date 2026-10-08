"""Sampling before color conversion must preserve pixels and the replay clock."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np

from bus_route.video import color_filters, latest_frames, native_frames, sample_frames


class SampleEquivalenceTests(unittest.TestCase):
    def ffmpeg(self, *args):
        subprocess.run(
            [imageio_ffmpeg.get_ffmpeg_exe(), '-hide_banner', '-loglevel', 'error', *args],
            check=True, capture_output=True,
        )

    def fixture(self, path, transfer=None, variable_rate=False, codec='ffv1'):
        args = ['-f', 'lavfi', '-i', 'testsrc2=size=96x64:rate=12:duration=1', '-an']
        if variable_rate:
            # Preserve a real gap in source timestamps, rather than merely
            # changing the frame-rate tag of a constant-rate stream.
            args += ['-vf', "select='not(eq(n,2)+eq(n,3)+eq(n,7))'", '-vsync', '0']
        args += ['-c:v', codec, '-threads', '1']
        if transfer:
            args += ['-pix_fmt', 'yuv420p10le', '-color_trc', transfer,
                     '-color_primaries', 'bt2020', '-colorspace', 'bt2020nc']
        else:
            args += ['-pix_fmt', 'yuv420p', '-color_trc', 'bt709',
                     '-color_primaries', 'bt709', '-colorspace', 'bt709']
        self.ffmpeg(*args, str(path))

    def original_samples(self, path, fps):
        """Fixed pre-change filter order, independent of sample_frames."""
        filters, mode = color_filters(path)
        reader = imageio_ffmpeg.read_frames(str(path), output_params=[
            '-vf', f'{filters},fps=fps={fps}:start_time=0', '-threads', '2'])
        try:
            meta = next(reader)
            width, height = meta['size']
            frames = [(index / fps, np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy())
                      for index, raw in enumerate(reader)]
            return {**meta, 'color_mode': mode, 'sample_fps': fps}, frames
        finally:
            reader.close()

    def assert_equivalent(self, path, expected_mode, expected_size=(96, 64)):
        old_meta, old_frames = self.original_samples(path, 5)
        reader = sample_frames(path, 5)
        try:
            new_meta = next(reader)
            new_frames = list(reader)
        finally:
            reader.close()
        self.assertEqual(new_meta, old_meta)
        self.assertEqual(new_meta['color_mode'], expected_mode)
        self.assertEqual(tuple(new_meta['size']), expected_size)
        self.assertGreaterEqual(len(old_frames), 4)
        self.assertEqual(len(new_frames), len(old_frames))
        self.assertEqual([t for t, _ in new_frames], [i / 5 for i in range(len(old_frames))])
        for (old_time, old_rgb), (new_time, new_rgb) in zip(old_frames, new_frames):
            self.assertEqual(old_time, new_time)
            np.testing.assert_array_equal(new_rgb, old_rgb)
        # A changing fixture catches an accidental constant-frame replacement.
        self.assertGreater(len({rgb.tobytes() for _, rgb in new_frames}), 1)

    def test_sdr_samples_keep_every_pixel_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sdr.mkv'
            self.fixture(path)
            self.assert_equivalent(path, 'sdr')

    def test_hlg_and_pq_samples_keep_every_pixel_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            for transfer in ('arib-std-b67', 'smpte2084'):
                with self.subTest(transfer=transfer):
                    path = Path(tmp) / f'{transfer}.mkv'
                    self.fixture(path, transfer=transfer)
                    self.assert_equivalent(path, 'hdr')

    def test_variable_rate_hdr_keeps_sample_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'vfr.mkv'
            self.fixture(path, transfer='arib-std-b67', variable_rate=True)
            native = list(native_frames(path))
            intervals = [round(b['source_pts_s'] - a['source_pts_s'], 2)
                         for a, b in zip(native, native[1:])]
            self.assertGreater(max(intervals), 2 * min(intervals))
            self.assert_equivalent(path, 'hdr')

    def test_rotation_is_preserved_with_sample_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, rotated = root / 'source.mp4', root / 'rotated.mp4'
            self.fixture(source, codec='libx264')
            self.ffmpeg('-display_rotation', '90', '-i', str(source),
                        '-c', 'copy', str(rotated))
            self.assert_equivalent(rotated, 'sdr', expected_size=(64, 96))

    def test_slow_first_rgb_is_not_given_a_fresh_clock_or_timestamp(self):
        # Model decoder readiness without wall-clock sleeps or actual media.
        # Metadata initialization occurs before the existing replay clock;
        # the subsequent 450ms RGB delay must remain visible to the age gate.
        clock = [20.0]
        rgb = np.zeros((2, 2, 3), np.uint8)

        def slow_source(path, fps):
            clock[0] = 100.0
            yield {'size': (2, 2), 'sample_fps': fps}
            clock[0] += .45
            yield 0.0, rgb

        with patch('bus_route.video.sample_frames', side_effect=slow_source), \
                patch('time.perf_counter', side_effect=lambda: clock[0]):
            reader = latest_frames(Path('synthetic'), 5)
            try:
                meta = next(reader)
                self.assertIsNone(meta['replay_origin'])
                frames = list(reader)
            finally:
                reader.close()
            self.assertEqual(meta['replay_origin'], 100.0)
            self.assertEqual(len(frames), 1)
            timestamp, observed = frames[0]
            self.assertEqual(timestamp, 0.0)
            self.assertIs(observed, rgb)
            age_s = clock[0] - meta['replay_origin'] - timestamp
            self.assertAlmostEqual(age_s, .45)
            self.assertGreater(age_s, .3)


if __name__ == '__main__':
    unittest.main()
