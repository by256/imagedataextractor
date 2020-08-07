import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from scalebar.ocr import OCR
from segment.model import ParticleSegmentation


image = np.array(Image.open('../examples/10 nm.png'))

model = ParticleSegmentation(device='cpu')

pred = model.segment(image)
print(pred.shape)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(image)
axes[1].imshow(pred)
plt.show()
