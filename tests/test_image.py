import logging
import pathlib
import time
import unittest

import numpy as np

from pco_image import PCOImage, config

logger = logging.getLogger('pco_image')
logger.setLevel('DEBUG')
__this_dir__ = pathlib.Path(__file__).parent


class TestPCOImage(unittest.TestCase):

    def test_core(self):
        pco_img = PCOImage(__this_dir__ / 'Cam1_0001A.b16')
        self.assertIsInstance(pco_img.filename, pathlib.Path)

        config.ENHANCED_READING = False
        st1 = time.perf_counter_ns()
        ts1 = pco_img.timestamp
        dt1 = time.perf_counter_ns() - st1

        pcoimg2 = PCOImage(__this_dir__ / 'Cam1_0001A.b16')
        config.ENHANCED_READING = True
        config.HEADER_SIZE = 50
        st2 = time.perf_counter_ns()
        ts2 = pcoimg2.timestamp
        dt2 = time.perf_counter_ns() - st2
        self.assertTrue(dt2 < dt1)

        self.assertEqual(ts1, ts2)

        self.assertIsInstance(pco_img.img, np.ndarray)

    def test_fails(self):
        pco_img = PCOImage(__this_dir__ / 'Cam1_0001A.b16', n_pixels=5)
        with self.assertRaises(ValueError):
            pco_img.timestamp
        pco_img = PCOImage(__this_dir__ / 'Cam1_0001B.b16')
        with self.assertRaises(ValueError):
            pco_img.timestamp
