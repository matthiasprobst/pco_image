"""This is basically an extension to package `pco_tools` (https://github.com/henne-s/pco-tools).
It allows to read the image index and timestamp from the first 14 pixels. For this, write "binary
timestamp" must be set in the PCO software.
"""

import pathlib
import struct
from datetime import datetime
from enum import Enum
from typing import Tuple, Union, List

import cv2
import numpy as np
from pco_tools import pco_reader

from . import config


class SourceType(Enum):
    """Image file type enumeration class"""
    b16 = 1
    tiff = 2
    nparray = 3


def bcd2digits(bcd_pixel: int, shift2bits: bool) -> str:
    """convert a BCD-encoded pco-pixel value into two digits"""
    if shift2bits:
        binary_string = bin(bcd_pixel)[2:].zfill(10)[:-2]
        digit1, digit2 = binary_string[0:4], binary_string[4:]
        return f'{int(digit1, 2)}{int(digit2, 2)}'
    binary_string = bin(bcd_pixel)[2:].zfill(8)
    digit1, digit2 = binary_string[0:4], binary_string[4:]
    return f'{int(digit1, 2)}{int(digit2, 2)}'


def get_stamp_from_16pixels(pixels, shift2bits: bool, return_raw=False):
    """Get image index and timestamp from pixels"""
    full_string = ''.join([bcd2digits(px, shift2bits=shift2bits) for px in pixels])
    if return_raw:
        return full_string[0:8], full_string[8:]
    try:
        dtime = datetime.strptime(full_string[8:26], "%Y%m%d%H%M%S%f")
    except ValueError as e:
        raise ValueError('Could not convert the timestamp to an datetime object.'
                         'Reason may be that there is no '
                         'timestamp written to the first `n_pixels` pixels or that `n_pixels` is '
                         'too long (it is typically 14 or 16) or too small (at least 10). '
                         'Consider calling .get_timestamp(True), because the original dat may be 14 bit '
                         'but are scaled to 16 bit in which case the decoding needs a small tweak. '
                         f'Orig. error: {e}')
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

    def __init__(self,
                 filename: Union[str, pathlib.Path, None],
                 n_pixels: int = 14,
                 timestamp_type='datetime',
                 stype: SourceType = None):
        """Int a PCOImage object.

        Parameters
        ----------
        filename: str or pathlib.Path or None
            Filename. Can be None. Obviously, function load() will not work.
            Thus, this is only reasonable if the file is initialized from an
            array (.from_array()).
            If a filename is provided (str or pathlib.Path), existence is checked.
        n_pixels: int=14
            Number of pixels to read from the image to get the timestamp and
            image index. This is typically 14 for tested PCO cameras.
        timestamp_type: str='datetime'
            Type of the timestamp. Can be 'datetime' or 'str'. If 'datetime',
            the timestamp is returned as a datetime object. If 'str', the
            timestamp is returned as a string.
        stype: SourceType=None
            Type of the image file. If None, the type is inferred from the
            filename suffix. If not None, the type is set to this value.

        Returns
        -------
        PCOImage
            The PCOImage object.

        """
        self.stype = stype
        if filename is None:
            self.filename = None
        else:
            self.filename = pathlib.Path(filename)
            if not self.filename.exists():
                raise FileNotFoundError(f'File not found: {self.filename.resolve().absolute()}')
            if self.stype is None:
                # get from image
                self.stype = SourceType.__getitem__(self.filename.suffix[1:])
            else:
                self.stype = stype
        self._img = None
        self._img_stamp = None
        self._idx = None
        self._dtime = None
        self._n_pixels = n_pixels
        self.timestamp_type = timestamp_type
        self._return_raw = timestamp_type != 'datetime'

    def __repr__(self):
        return f'<{self.__class__.__name__} {self.filename}>'

    def __getitem__(self, item) -> "np.ndarray":
        """Slice the image. Will be loaded if not yet done."""
        return self.img.__getitem__(item)

    def __sub__(self, other):
        img = PCOImage(None, n_pixels=self._n_pixels, timestamp_type=self.timestamp_type)
        if isinstance(other, (int, float)):
            img._img = self.img - other
        else:
            img._img = self.img - other.img
        return img

    def __add__(self, other):
        img = PCOImage(None, n_pixels=self._n_pixels, timestamp_type=self.timestamp_type)
        if isinstance(other, (int, float)):
            img._img = self.img + other
        else:
            img._img = self.img + other.img
        return img

    def __mul__(self, other):
        """Multiply two images or multiply by a number"""
        img = PCOImage(None, n_pixels=self._n_pixels, timestamp_type=self.timestamp_type)
        # check if "other" is a number:
        if isinstance(other, (int, float)):
            img._img = self.img * other
        else:
            img._img = self.img * other.img
        return img

    def __truediv__(self, other):
        """Divide two images or divide by a number"""
        img = PCOImage(None, n_pixels=self._n_pixels, timestamp_type=self.timestamp_type)
        # check if "other" is a number:
        if isinstance(other, (int, float)):
            img._img = self.img / other
        else:
            img._img = self.img / other.img
        return img

    @property
    def img(self) -> "np.ndarray":
        """Return img as np.ndarray"""
        if self._img is None:
            self._img = self.load_image()
        return self._img

    @img.setter
    def img(self, img: np.ndarray) -> None:
        """Set the image."""
        self._img = img

    @staticmethod
    def from_tiff(filename, n_pixels=14, timestamp_type='datetime') -> "PCOImage":
        """init from a .tiff image"""
        return PCOImage(filename, n_pixels=n_pixels, timestamp_type=timestamp_type, stype=SourceType.tiff)

    @staticmethod
    def from_b16(filename, n_pixels=14, timestamp_type='datetime') -> "PCOImage":
        """init from a .b16 image"""
        return PCOImage(filename, n_pixels=n_pixels, timestamp_type=timestamp_type, stype=SourceType.b16)

    @staticmethod
    def from_array(array: np.ndarray, n_pixels=14, timestamp_type='datetime') -> "PCOImage":
        """init from a numpy array

        Parameters
        ----------
        array: np.ndarray
            The array to initialize the image from.
        n_pixels: int=14
            Number of pixels to read from the image to get the timestamp and
            image index. This is typically 14 for tested PCO cameras.
        timestamp_type: str='datetime'
            Type of the timestamp. Can be 'datetime' or 'str'. If 'datetime',
            the timestamp is returned as a datetime object. If 'str', the
            timestamp is returned as a string.

        Returns
        -------
        PCOImage
            The PCOImage object.

        Examples
        --------
        >>> import numpy as np
        >>> from pco_image import PCOImage
        >>> img = PCOImage.from_array(np.zeros((100, 100), dtype=np.uint16))
        """
        pco_img = PCOImage(None, n_pixels=n_pixels, timestamp_type=timestamp_type, stype=SourceType.b16)
        pco_img._img = array
        return pco_img

    def info(self):
        """call `pco_reader.info`"""
        if self.stype == SourceType.b16:
            return pco_reader.info(self.filename)
        raise ValueError('Info only available for .b16-images')

    def load_image(self) -> "np.ndarray":
        """Reads image and overwrites object variable `_img`"""
        if self.stype == SourceType.b16:
            self._img = pco_reader.load(self.filename)
        elif self.stype == SourceType.tiff:
            self._img = cv2.imread(str(self.filename), cv2.IMREAD_UNCHANGED)
        return self._img

    def write(self, filename: Union[str, pathlib.Path, None]) -> Tuple[bool, pathlib.Path]:
        """Call cv2.imwrite() to write image"""
        if filename is None:
            filename = self.filename
        return cv2.imwrite(str(filename), self.img), pathlib.Path(filename).absolute()

    def get_pixels(self, stop, start=0) -> "np.ndarray":
        """return the pixels from `start` to `stop`"""
        if self._img is not None:
            return self._img.ravel()[start:stop]

        if self.stype == SourceType.tiff:
            self._img = cv2.imread(str(self.filename), cv2.IMREAD_UNCHANGED)
            return self._img.ravel()[start:stop]

        if not config.ENHANCED_READING:
            return self.img.ravel()[start:stop]

        header_size = config.B16_HEADER_SIZE
        found_header_size = False
        while not found_header_size:
            with open(self.filename, 'rb') as f:
                buf = f.read(header_size + stop * 2)
                actual_header_size = struct.unpack_from('<' + 'L' * 6, buf)[2]
                if actual_header_size != header_size:
                    header_size = actual_header_size
                    config.B16_HEADER_SIZE = actual_header_size
                else:
                    found_header_size = True

        data = buf[header_size:header_size + (stop * 2)]
        return np.frombuffer(data, dtype=np.dtype('<u2'))[start:stop]

    def get_index(self, shift2bits: bool = True) -> int:
        """return image index

        Parameters
        ----------
        shift2bits: bool=True
            Shift the data by 2 bits in order to convert 16bit to 14 bit.
            This is needed if original data is 14 bit was saved to 16 bit
        """
        if self._idx is None:
            self._idx, self._dtime = get_stamp_from_16pixels(self.get_pixels(self._n_pixels),
                                                             shift2bits=shift2bits,
                                                             return_raw=self._return_raw)
        return self._idx

    def get_timestamp(self, shift2bits: bool = True) -> datetime:
        """return timestamp as `datetime` object

        Parameters
        ----------
        shift2bits: bool=True
            Shift the data by 2 bits in order to convert 16bit to 14 bit.
            This is needed if original data is 14 bit was saved to 16 bit
        """
        if self._dtime is None:
            self._idx, self._dtime = get_stamp_from_16pixels(self.get_pixels(stop=self._n_pixels),
                                                             shift2bits=shift2bits,
                                                             return_raw=self._return_raw)
        return self._dtime


def get_timesteps(filenames: List[Union[str, pathlib.Path]], shift2bits: bool = True) -> List:
    """get timestep from multiple files and return as list"""
    ts = []
    for filename in filenames:
        ts.append(PCOImage(filename).get_timestamp(shift2bits=shift2bits))
    return ts
