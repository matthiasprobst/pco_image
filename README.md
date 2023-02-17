# PCO Image

Small python package that can read PCO images and its **metadata**, especially its **image index** and **time stamp**.

For reading the image, the package `pco_tools` is used.

It is tested for the pco.pixelfly camera (14 bit camera) but should work for other PCO cameras, too. For 
16 bit cameras you may disable 2 bit pixel shift in which case `get_timestamp(False)` is called.

## Example

````python
import matplotlib.pyplot as plt

from pco_image import PCOImage

pco_img = PCOImage('image.b16')  # not yet loading the image
plt.imshow(pco_img.img)  # only now loading the image

print(f'image index: {pco_img.get_index()}')
>>> image index: 1
print(f'image timestamp: {pco_img.get_timestamp()}')
>>> image timestamp: 2023-01-20 18:21:53.096300
# write to other file:
pco_img.write('out.tiff')

````

## Installation

```bash
pip install pco_image
```