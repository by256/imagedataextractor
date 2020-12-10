import os
import torch
import numpy as np
from PIL import Image

from .cluster import Cluster
from .nnmodules import BranchedERFNet
from .uncertainty import expected_entropy, predictive_entropy, uncertainty_filtering

class ParticleSegmenter:

    def __init__(self, bayesian=True, n_samples=30, tu=0.0125, device='cpu'):
        """
        BPartIS particle segmentation model for particle identification.
        
        Parameters
        ----------
        bayesian: bool
            Option to use Bayesian inference for prediction. Trades off speed
            for accuracy (recommended) (default is True).
        n_samples: int
            Number of monte carlo samples used for Bayesian inference (default
            is 40).
        device: str {'cpu', 'cuda', None}
            Selected device to run inference on. If None, will select 'cuda' if a
            GPU is available, otherwise will default to 'cpu' (default is 'cpu').
        """
        self.bayesian = bayesian
        self.n_samples = n_samples
        self.tu = tu
        self.seg_model = BranchedERFNet(num_classes=[4, 1]).to(device).eval()
        self.model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../models/seg-model.pt')
        self.seg_model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.cluster = Cluster(n_sigma=2, device=device)
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
    def preprocess(self, image):
        """Pre-process image for segmentation model."""
        image = Image.fromarray(image)
        image = image.resize((512, 512), resample=Image.BICUBIC)
        image = np.array(image)
        image = image / 255.0
        return image

    def postprocess_pred(self, image, h, w):
        """Post-process output segmentation map. Return output to the original input size."""
        image = Image.fromarray(image)
        image = image.resize((w, h), resample=Image.NEAREST)
        return np.array(image)

    def postprocess_uncertainty(self, image, h, w):
        """
        Resize uncertainty map. This is strictly for visualisation purposes.
        The output of this function will not be used for anything other 
        than visualisation.
        """
        image = Image.fromarray(image)
        image = image.resize((w, h), resample=Image.BICUBIC)
        return np.array(image)

    def enable_eval_dropout(self):
        """Enables dropout in eval mode for Bayesian inference via Monte Carlo dropout."""
        for module in self.seg_model.modules():
            if 'Dropout' in type(module).__name__:
                module.train()

    def monte_carlo_predict(self, image):
        """Performs Bayesian inference and computes epistemic uncertainty."""
        h, w = image.shape[-2:]
        cluster = Cluster(n_sigma=2, h=h, w=w, device=self.device)
        self.enable_eval_dropout()

        # get monte carlo model samples
        mc_outputs = []
        mc_seed_maps = []
        for i in range(self.n_samples):
            output = self.seg_model(image).detach()
            seed_map = torch.sigmoid(output[0, -1]).unsqueeze(0)  # \phi_{k}(e_{i})
            mc_outputs.append(output)
            mc_seed_maps.append(seed_map)

        mc_outputs = torch.cat(mc_outputs, dim=0)
        mc_seed_maps = torch.cat(mc_seed_maps, dim=0)

        # MC prediction (cluster the mean of MC samples)
        mc_prediction, _ = cluster.cluster(mc_outputs.mean(dim=0))

        # Uncertainty
        total = predictive_entropy(mc_seed_maps)
        aleatoric = expected_entropy(mc_seed_maps)
        epistemic = total - aleatoric  # $MI(y, \theta | x)$

        return mc_prediction, epistemic

    def segment(self, image):
        """Main segmentation routine."""
        o_h, o_w = image.shape[:2]
        image = self.preprocess(image)
        image = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if self.bayesian:
            # monte carlo predict
            pred, uncertainty = self.monte_carlo_predict(image)
            original = pred.cpu().numpy().copy()
            original = self.postprocess_pred(original, o_h, o_w)
            pred = uncertainty_filtering(pred, uncertainty, tu=self.tu)
            pred = pred.cpu().numpy()
            uncertainty = uncertainty.cpu().numpy()
            # post-process uncertainty for visualisation
            uncertainty = self.postprocess_uncertainty(uncertainty, o_h, o_w)
        else:
            model_out = self.seg_model(image)[0].detach()
            pred = self.cluster.cluster(model_out)[0].cpu().numpy()
            uncertainty = None
            original = None
        pred = self.postprocess_pred(pred, o_h, o_w)
        return pred, uncertainty, original
