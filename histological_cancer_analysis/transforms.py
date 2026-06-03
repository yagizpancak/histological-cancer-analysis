from __future__ import annotations

from torchvision import transforms

from histological_cancer_analysis.constants import DEFAULT_IMAGE_MEAN, DEFAULT_IMAGE_STD


def build_image_transforms(
    image_size: int,
    augment: bool,
    mean: tuple[float, float, float] = DEFAULT_IMAGE_MEAN,
    std: tuple[float, float, float] = DEFAULT_IMAGE_STD,
) -> transforms.Compose:
    transform_steps: list[object] = [
        transforms.Resize((image_size, image_size)),
    ]
    if augment:
        transform_steps.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ],
        )
    transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ],
    )
    return transforms.Compose(transform_steps)
