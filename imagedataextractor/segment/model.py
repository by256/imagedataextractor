import torch
import numpy as np
from PIL import Image
from .nnmodules import BranchedERFNet
from .cluster import Cluster


class ParticleSegmentation:

    def __init__(self, device='cpu'):
        self.seg_model = BranchedERFNet(num_classes=[4, 1]).to(device)
        self.model_path = '/home/by256/Documents/Projects/imagedataextractor/imagedataextractor/models/seg-model.pt'
        self.seg_model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.cluster = Cluster(n_sigma=2, device=device)
        self.device = device

    def preprocess(self, image):
        image = Image.fromarray(image)
        image = image.resize((512, 512), resample=Image.BICUBIC)
        image = np.array(image)
        image = image / 255.0
        return image

    def segment(self, image):
        image = self.preprocess(image)
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        self.seg_model.eval()
        model_out = self.seg_model(image)[0].detach()
        pred = self.cluster.cluster(model_out)[0]
        return pred.cpu().numpy()
