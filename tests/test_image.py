import datetime
import pathlib
import time
import unittest

import numpy as np

from pco_image import PCOImage, config

__this_dir__ = pathlib.Path(__file__).parent


class TestPCOImage(unittest.TestCase):

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
