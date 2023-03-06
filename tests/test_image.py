import datetime
import pathlib
import time
import unittest

import numpy as np

from pco_image import PCOImage, config, __version__

__this_dir__ = pathlib.Path(__file__).parent


class TestPCOImage(unittest.TestCase):

    def test_version(self):
        self.assertEqual(__version__, '0.3.0')

    def test_tiff(self):
        pco_img = PCOImage.from_tiff(__this_dir__ / 'Cam1_1A_noshift.tiff')
        self.assertEqual(str(pco_img.get_timestamp(True)), '2023-02-15 08:52:01.122900')

        pco_img = PCOImage.from_tiff(__this_dir__ / 'Cam0A.tiff')
        self.assertEqual(str(pco_img.get_timestamp(False)), '2023-02-14 23:05:20.754900')
        self.assertEqual(pco_img.get_index(False), 1)
        self.assertIsInstance(pco_img.get_index(), int)

        pco_img = PCOImage.from_tiff(__this_dir__ / 'Cam0A.tiff')
        self.assertEqual(pco_img.get_index(False), 1)
        self.assertIsInstance(pco_img.get_index(False), int)

        pco_img.write(__this_dir__ / 'out.tiff')

        pco_img = PCOImage.from_tiff(__this_dir__ / 'out.tiff')
        self.assertEqual(pco_img.get_index(False), 1)
        self.assertIsInstance(pco_img.get_index(False), int)

        (__this_dir__ / 'out.tiff').unlink()

    def test_b16(self):
        pco_img = PCOImage.from_b16(__this_dir__ / 'Cam1_0001A.b16')
        self.assertEqual(str(pco_img.get_timestamp(True)), '2023-01-20 18:21:53.096300')
        self.assertEqual(pco_img.get_index(), 1)
        self.assertIsInstance(pco_img.get_index(), int)

        pco_img = PCOImage(__this_dir__ / 'Cam1_0001A.b16')
        self.assertIsInstance(pco_img.filename, pathlib.Path)

        config.ENHANCED_READING = False
        st1 = time.perf_counter_ns()
        ts1 = pco_img.get_timestamp()
        dt1 = time.perf_counter_ns() - st1
        assert ts1 == datetime.datetime(2023, 1, 20, 18, 21, 53, 96300)

        pcoimg2 = PCOImage(__this_dir__ / 'Cam1_0001A.b16')
        config.ENHANCED_READING = True
        st2 = time.perf_counter_ns()
        ts2 = pcoimg2.get_timestamp()
        dt2 = time.perf_counter_ns() - st2
        self.assertTrue(dt2 <= dt1)

        self.assertEqual(ts1, ts2)

        self.assertIsInstance(pco_img.img, np.ndarray)

    def test_other(self):
        pco_img = PCOImage.from_b16(__this_dir__ / 'Cam1_0001A.b16')

        self.assertIsInstance(pco_img.img, np.ndarray)
        self.assertIsInstance(pco_img[:, :], np.ndarray)

        img_orig = pco_img.img[0, 0]
        pco_img.img = pco_img.img / 4
        self.assertEqual(pco_img.img[0, 0], img_orig / 4)

    def test_fails(self):
        pco_img = PCOImage(__this_dir__ / 'Cam0A.tiff', n_pixels=5)
        with self.assertRaises(ValueError):
            pco_img.get_timestamp()

        # in B there is no timestamp:
        pco_img = PCOImage(__this_dir__ / 'Cam0B.tiff')
        with self.assertRaises(ValueError):
            pco_img.get_timestamp()

    def test_mathematical_operations(self):
        pco_img = PCOImage(__this_dir__ / 'Cam0A.tiff', n_pixels=5)
        pco_img2 = PCOImage(__this_dir__ / 'Cam0A.tiff', n_pixels=5)

        # addition
        sub_img = pco_img2 - pco_img
        self.assertIsInstance(sub_img, PCOImage)
        self.assertIsInstance(sub_img.img, np.ndarray)
        np.testing.assert_array_equal(sub_img.img, np.zeros(pco_img.img.shape))

        # add number:
        add_img = pco_img + 1
        self.assertIsInstance(add_img, PCOImage)
        self.assertIsInstance(add_img.img, np.ndarray)
        np.testing.assert_array_equal(add_img.img, pco_img.img + 1)

        # subtraction
        add_img = pco_img2 - pco_img
        self.assertIsInstance(add_img, PCOImage)
        self.assertIsInstance(add_img.img, np.ndarray)
        np.testing.assert_array_equal(add_img.img, 0 * pco_img.img)

        # save as tiff file:
        add_img.write(__this_dir__ / 'out.tiff')
        self.assertTrue((__this_dir__ / 'out.tiff').exists())

        # load from tiff file:
        pco_img_loaded = PCOImage.from_tiff(__this_dir__ / 'out.tiff')
        self.assertIsInstance(pco_img_loaded, PCOImage)
        self.assertIsInstance(pco_img_loaded.img, np.ndarray)
        np.testing.assert_array_equal(add_img.img, 0 * pco_img.img)

        # remove tiff file:
        (__this_dir__ / 'out.tiff').unlink()

        # subtract number:
        sub_img = pco_img - 1
        self.assertIsInstance(sub_img, PCOImage)
        self.assertIsInstance(sub_img.img, np.ndarray)
        np.testing.assert_array_equal(sub_img.img, pco_img.img - 1)

        # multiplication
        mul_img = pco_img2 * pco_img
        self.assertIsInstance(mul_img, PCOImage)
        self.assertIsInstance(mul_img.img, np.ndarray)
        np.testing.assert_array_equal(mul_img.img, pco_img.img ** 2)

        # multiply number:
        mul_img = pco_img * 2
        self.assertIsInstance(mul_img, PCOImage)
        self.assertIsInstance(mul_img.img, np.ndarray)
        np.testing.assert_array_equal(mul_img.img, pco_img.img * 2)

        # division
        with self.assertWarns(RuntimeWarning):
            # divides by zero
            div_img = pco_img2 / pco_img
        self.assertIsInstance(div_img, PCOImage)
        self.assertIsInstance(div_img.img, np.ndarray)
        cmp = np.ones(pco_img.img.shape)
        cmp[pco_img.img == 0] = np.nan
        np.testing.assert_array_equal(div_img.img, cmp)

        # divide number:
        div_img = pco_img / 2
        self.assertIsInstance(div_img, PCOImage)
        self.assertIsInstance(div_img.img, np.ndarray)
        np.testing.assert_array_equal(div_img.img, pco_img.img / 2)

    def test_multiple_timesteps(self):
        import shutil
        multiple_dir = pathlib.Path(__this_dir__ / 'multiple')
        multiple_dir.mkdir(exist_ok=True)
        for i in range(5000):
            shutil.copy('Cam1_0001A.b16', multiple_dir / f'cam{i:04d}.b16')

        filenames = sorted(multiple_dir.glob('*.b16'))
        from pco_image.image import get_timesteps
        ts = get_timesteps(filenames)
        self.assertEqual(len(ts), 5000)
        self.assertIsInstance(ts[0], datetime.datetime)
        shutil.rmtree(multiple_dir)
