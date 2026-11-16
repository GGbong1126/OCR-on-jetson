from pathlib import Path
import sys
import unittest

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jetson'))
import compare_jpeg_rgb as rgb


class Tests(unittest.TestCase):
    def test_metrics_and_chunk_boundary(self):
        x = np.zeros((129, 2, 3), dtype=np.uint8)
        y = x.copy()
        y[-1, 0, 0] = 255
        diff = rgb.metrics(x, y)
        self.assertEqual(diff['max_abs_error'], 255)
        self.assertEqual(diff['mae'], 255 / x.size)
        self.assertEqual(diff['identical_components_fraction'], (x.size - 1) / x.size)
        self.assertEqual(rgb.metrics(x, x)['mae'], 0)
        for bad in (x.astype(np.float32), x[:1], x[:, :, :1]):
            with self.assertRaises(ValueError):
                rgb.metrics(x, bad)

    def test_trace_fd_reuse_and_errors(self):
        trace = '''123 openat(AT_FDCWD, "/dev/nvhost-nvjpg", O_RDWR) = 16</dev/nvhost-nvjpg>
123 ioctl(0x10, 0xc000, 0x1) = 0
123 ioctl(0x10, 0xc001, 0x1) = -1 EINVAL
123 close(16</dev/nvhost-nvjpg>) = 0
123 openat(AT_FDCWD, "/tmp/not_device", O_RDONLY) = 16</tmp/not_device>
123 ioctl(0x10, 0xc002, 0x1) = 0
123 openat(AT_FDCWD, "/dev/missing", O_RDONLY) = -1 ENOENT'''
        self.assertEqual(rgb.device_calls(trace), {'/dev/nvhost-nvjpg': dict(opens=1, ioctl_success=1, ioctl_failure=1)})

    def test_ambiguous_interleaving_not_silently_counted(self):
        with self.assertRaises(ValueError):
            rgb.device_calls('123 ioctl(0x10, 0x1, 0x2 <unfinished ...>')

    def test_raw_submit_command_is_counted_only_for_its_owned_device(self):
        trace='''openat(AT_FDCWD, "/dev/nvhost-nvjpg", O_RDWR) = 6
ioctl(0x6, 0xffffffffc080481a, 0x1) = 0
ioctl(0x6, 0xc080481a, 0x1) = -1 EINVAL
close(6) = 0
openat(AT_FDCWD, "/tmp/synthetic", O_RDWR) = 6
ioctl(0x6, 0xc080481a, 0x1) = 0'''
        self.assertEqual(rgb.device_calls(trace,include_commands=True)['/dev/nvhost-nvjpg']['successful_commands'],{'0xc080481a':1})


if __name__ == '__main__':
    unittest.main()
