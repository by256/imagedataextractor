
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scalebar.ocr import OCR
from segment.model import ParticleSegmentation


image = np.array(Image.open('../examples/20 nm(2).png'))

ocr = OCR()
model = ParticleSegmentation(device='cpu')

# segment
pred = model.segment(image)

#scalebar
scalebar_text = ocr.get_scalebar_text(image)
print('scalebar_text', scalebar_text)


fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[1].imshow(pred, cmap='tab20')
plt.suptitle(scalebar_text)
plt.show()
