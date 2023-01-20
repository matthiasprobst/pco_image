# PCO Image

Small python package that can read PCO images and its **metadata**, especially its **image index** and **time stamp**.

For reading the image, the package `pco_tools` is used.

It is tested for the pco.pixelfly camera but should work for other PCO cameras, too

## Example

````python
import matplotlib.pyplot as plt

from pco_image import PCOImage

pco_img = PCOImage('image.b16')  # not yet loading the image
plt.imshow(pco_img.img)  # only now loading the image

print(f'image index: {pco_img.index}')
print(f'image timestamp: {pco_img.timestamp}')

````

## Installation

```bash
pip install pco_image
```