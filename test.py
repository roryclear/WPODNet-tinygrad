from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
from PIL import Image, ImageDraw
from typing import Optional, Tuple
import numpy as np
from torchvision.transforms.functional import to_tensor
from tinygrad import Tensor as tinyTensor
from torch import Tensor

def to_tiny(x): return tinyTensor(x.detach().numpy())

def to_torch(x): return Tensor(x.numpy())

@dataclass(frozen=True)
class Prediction:
    bounds: List[Tuple[int, int]]
    confidence: float

    def annotate(
        self,
        canvas: Image.Image,
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: int = 3,
    ) -> None:
        drawer = ImageDraw.Draw(canvas)
        drawer.polygon(self.bounds, fill=fill, outline=outline, width=width)

class BasicConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv_layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn_layer = torch.nn.BatchNorm2d(out_channels, eps=0.001)
        self.act_layer = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn_layer(x)
        x = to_tiny(x)
        x = tinyTensor.relu(x)
        return to_torch(x)


class ResBlock(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv_block = BasicConvBlock(channels, channels)
        self.sec_layer = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn_layer = torch.nn.BatchNorm2d(channels, eps=0.001)
        self.act_layer = torch.nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.conv_block(x)
        h = self.sec_layer(h)
        h = self.bn_layer(h)
        h = to_tiny(h)
        x = to_tiny(x)
        ret = tinyTensor.relu(x+h)
        return to_torch(ret)


class WPODNet(torch.nn.Module):
    stride = 16  # net_stride
    scale_factor = 7.75  # side
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            BasicConvBlock(3, 16),
            BasicConvBlock(16, 16),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(16, 32),
            ResBlock(32),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(32, 64),
            ResBlock(64),
            ResBlock(64),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(64, 64),
            ResBlock(64),
            ResBlock(64),
            torch.nn.MaxPool2d(2),
            BasicConvBlock(64, 128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
            ResBlock(128),
        )
        self.prob_layer = torch.nn.Conv2d(128, 2, kernel_size=3, padding=1)
        self.bbox_layer = torch.nn.Conv2d(128, 6, kernel_size=3, padding=1)


    def forward(self, image: torch.Tensor):
        feature = self.backbone(image)
        probs = self.prob_layer(feature)
        probs = torch.softmax(probs, dim=1)
        affines = self.bbox_layer(feature)

        return probs, affines

Q = np.array(
    [
        [-0.5, 0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5, 0.5],
        [1.0, 1.0, 1.0, 1.0],
    ]
)

class Predictor:
    def __init__(self, wpodnet: WPODNet) -> None:
        self.wpodnet = wpodnet
        self.wpodnet.eval()

    def _resize_to_fixed_ratio(
        self, image: Image.Image, dim_min: int, dim_max: int
    ) -> Image.Image:
        h, w = image.height, image.width

        wh_ratio = max(h, w) / min(h, w)
        side = int(wh_ratio * dim_min)
        bound_dim = min(side + side % self.wpodnet.stride, dim_max)

        factor = bound_dim / max(h, w)
        reg_w, reg_h = int(w * factor), int(h * factor)

        # Ensure the both width and height are the multiply of `self.wpodnet.stride`
        reg_w_mod = reg_w % self.wpodnet.stride
        if reg_w_mod > 0: reg_w += self.wpodnet.stride - reg_w_mod
        reg_h_mod = reg_h % self.wpodnet.stride
        if reg_h_mod > 0: reg_h += self.wpodnet.stride - reg_h_mod
        return image.resize((reg_w, reg_h))

    def _inference(self, image: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        probs, affines = self.wpodnet.forward(image)

        probs = np.squeeze(probs.detach().numpy())[0]  # (grid_h, grid_w)
        affines = np.squeeze(affines.detach().numpy())  # (6, grid_h, grid_w)

        return probs, affines

    def _get_max_anchor(self, probs: np.ndarray) -> Tuple[int, int]: return np.unravel_index(probs.argmax(), probs.shape)

    def _get_bounds(
        self,
        affines: np.ndarray,
        anchor_y: int,
        anchor_x: int,
        scaling_ratio: float = 1.0,
    ) -> np.ndarray:
        # Compute theta
        theta = affines[:, anchor_y, anchor_x]
        theta = theta.reshape((2, 3))
        theta[0, 0] = max(theta[0, 0], 0.0)
        theta[1, 1] = max(theta[1, 1], 0.0)

        # Convert theta into the bounding polygon
        bounds = np.matmul(theta, Q) * self.wpodnet.scale_factor * scaling_ratio

        # Normalize the bounds
        _, grid_h, grid_w = affines.shape
        bounds[0] = (bounds[0] + anchor_x + 0.5) / grid_w
        bounds[1] = (bounds[1] + anchor_y + 0.5) / grid_h

        return np.transpose(bounds)

    def predict(
        self,
        image: Image.Image,
        scaling_ratio: float = 1.0,
        dim_min: int = 512,
        dim_max: int = 768,
    ) -> Prediction:
        orig_h, orig_w = image.height, image.width
        resized = self._resize_to_fixed_ratio(image, dim_min=dim_min, dim_max=dim_max)
        resized = to_tensor(resized).unsqueeze(0)
        probs, affines = self._inference(resized)
        max_prob = np.amax(probs)
        anchor_y, anchor_x = self._get_max_anchor(probs)
        bounds = self._get_bounds(affines, anchor_y, anchor_x, scaling_ratio)

        bounds[:, 0] *= orig_w
        bounds[:, 1] *= orig_h

        return Prediction(
            bounds=[(x, y) for x, y in np.int32(bounds).tolist()],
            confidence=max_prob.item(),
        )

@dataclass(frozen=True)
class Prediction:
    bounds: List[Tuple[int, int]]
    confidence: float

    def annotate(
        self,
        canvas: Image.Image,
        fill: Optional[str] = None,
        outline: Optional[str] = None,
        width: int = 3,
    ) -> None:
        drawer = ImageDraw.Draw(canvas)
        drawer.polygon(self.bounds, fill=fill, outline=outline, width=width)

if __name__ == "__main__":
    model = WPODNet()
    checkpoint = torch.load("weights/wpodnet.pth", weights_only=True)
    model.load_state_dict(checkpoint)

    predictor = Predictor(model)


    image = Image.open("img.jpg")
    prediction = predictor.predict(image)

    expected = [(724, 1397), (1034, 1466), (1030, 1545), (720, 1477)]

    print("  bounds", prediction.bounds)
    print("  confidence", prediction.confidence)

    np.testing.assert_allclose(prediction.bounds, expected, rtol=1e-6, atol=1e-6)

    new_name = image.filename.split(".jpg")[0] + "_out.jpg"
    annotated_path = Path(new_name).name

    canvas = image.copy()
    prediction.annotate(canvas, outline="red")
    canvas.save(annotated_path)
    print(f"Saved the annotated image at {annotated_path}")
