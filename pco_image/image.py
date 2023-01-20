"""This is basically an extension to package `pco_tools` (https://github.com/henne-s/pco-tools).
It allows to read the image index and timestamp from the first 14 pixels. For this, write binary
timestamp must be set in the PCO software.
"""

import pathlib
from datetime import datetime
from typing import Tuple

from pco_tools import pco_reader


def bcdpixel_to_digits(pixel_value) -> str:
    """convert a BCD-encoded pco-pixel value into two digits"""
    # pixel value must be right shifted by 2 (don't know why. nowhere documented in pco)
    binary_string = bin(pixel_value)[2:].zfill(10)[:-2]
    digit1, digit2 = binary_string[0:4], binary_string[4:]
    return f'{int(digit1, 2)}{int(digit2, 2)}'


def get_stamp_from_pco_image(img, n_pixels: int = 14, return_raw=False) -> Tuple[int, str]:
    """Get image index and timestamp from the first `n_pixels` (usually 14)"""
    pixels = img.ravel()[0:n_pixels]
    full_string = ''.join([bcdpixel_to_digits(px) for px in pixels])
    if return_raw:
        return full_string[0:8], full_string[8:]
    try:
        dtime = datetime.strptime(full_string[8:], "%Y%m%d%H%M%S%f")
    except ValueError as e:
        raise ValueError('Could not convert the timestamp to an datetime object. Reason may be that `n_pixels` is '
                         f'too long (it is typically 14 or 16) or too small (at least 10). Orig. error: {e}')
    return int(full_string[0:8]), dtime


class PCOImage:
    """Interface class to a PCO image.

    Can read the image and (if available) reads image index
    and timestamp from the first pixels.

    Initializing the object will not directly load the image.
    Only when property `img` is called. On the next call of
    `img` the image then is NOT reloaded! If the image has
    changed, call `load_image` to reload the image.
    """

    def __init__(self, filename: pathlib.Path, n_pixels: int = 14):
        self.filename = pathlib.Path(filename)
        self._img = None
        self._idx = None
        self._dtime = None
        self._n_pixels = n_pixels

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.filename}>'

    def load_image(self) -> "np.ndarray":
        """Reads image and overwrites object variable `_img`"""
        self._img = pco_reader.load(self.filename)
        return self._img

    @property
    def img(self) -> "np.ndarray":
        """Return img as np.ndarray"""
        if self._img is None:
            self._img = pco_reader.load(self.filename)
        return self._img

    def _extract_stamp(self) -> Tuple[int, str]:
        if self._img is None:
            self._idx, self._dtime = get_stamp_from_pco_image(self.img, n_pixels=self._n_pixels)
        return self._idx

    @property
    def index(self) -> int:
        """return image index"""
        if self._idx is None:
            self._idx, self._dtime = get_stamp_from_pco_image(self.img, n_pixels=self._n_pixels)
        return self._idx

    @property
    def timestamp(self) -> datetime:
        """return timestamp as `datetime` object"""
        if self._dtime is None:
            self._idx, self._dtime = get_stamp_from_pco_image(self.img, n_pixels=self._n_pixels)
        return self._dtime
