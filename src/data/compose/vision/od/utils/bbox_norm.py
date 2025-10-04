import torch
from torchvision import tv_tensors

class BBoxNormalizationTransform(torch.nn.Module):
    def forward(self, img, label):
        if not "boxes" in label:
            label.update(
                {
                    "boxes": tv_tensors.BoundingBoxes(
                        torch.empty((0, 4)),
                        device=img.device,
                        format="XYXY",
                        canvas_size=(tuple(img.size()[1::])),
                    )
                }
            )
            return img, label

        label["boxes"][:, [0, 2]] /= label["boxes"].canvas_size[1]
        label["boxes"][:, [1, 3]] /= label["boxes"].canvas_size[0]

        return img, label