
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scalebar.ocr import OCR
from segment.model import ParticleSegmentation
from analysis import particle_size_hist


image = np.array(Image.open('../examples/20 nm(2).png'))

ocr = OCR()
model = ParticleSegmentation(device='cpu')

# segment
pred = model.segment(image)

#scalebar
scalebar_text = ocr.get_scalebar_text(image)
print('scalebar_text', scalebar_text)

# analysis
sizes = particle_size_hist(pred)

fig, axes = plt.subplots(1, 3)
axes[0].imshow(image)
axes[1].imshow(pred, cmap='tab20')
axes[2].hist(sizes, bins=20)
plt.suptitle(scalebar_text)
plt.show()
