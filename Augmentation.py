import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from skimage.transform import (
    AffineTransform,
    ProjectiveTransform,
    resize,
    rotate,
    swirl,
    warp,
)
from skimage.util import img_as_float32, img_as_ubyte


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".JPG", ".JPEG"}


def augment_image(image: np.ndarray) -> dict[str, np.ndarray]:
    h, w = image.shape[:2]

    flip_img = np.fliplr(image)

    rotate_img = rotate(
        image,
        angle=25,
        resize=False,
        mode="constant",
        cval=0,
        preserve_range=True,
    )

    src = np.array(
        [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
        dtype=np.float32,
    )
    skew_shift = max(1.0, 0.12 * w)
    dst_skew = np.array(
        [
            [skew_shift, 0],
            [w - 1, 0],
            [w - 1 - skew_shift, h - 1],
            [0, h - 1],
        ],
        dtype=np.float32,
    )
    skew_tform = ProjectiveTransform()
    skew_tform.estimate(src, dst_skew)
    skew_img = warp(
        image,
        inverse_map=skew_tform.inverse,
        mode="constant",
        cval=0,
        preserve_range=True,
    )

    shear_tform = AffineTransform(shear=0.25)
    shear_img = warp(
        image,
        inverse_map=shear_tform.inverse,
        mode="constant",
        cval=0,
        preserve_range=True,
    )

    crop_ratio = 0.8
    crop_h = max(1, int(h * crop_ratio))
    crop_w = max(1, int(w * crop_ratio))
    y0 = (h - crop_h) // 2
    x0 = (w - crop_w) // 2
    cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
    crop_img = resize(
        cropped,
        image.shape,
        mode="constant",
        cval=0,
        anti_aliasing=True,
        preserve_range=True,
    )

    distortion_img = swirl(
        image,
        strength=2.0,
        radius=min(h, w) / 2.0,
        mode="constant",
        cval=0,
        preserve_range=True,
    )

    return {
        "Flip": flip_img,
        "Rotate": rotate_img,
        "Skew": skew_img,
        "Shear": shear_img,
        "Crop": crop_img,
        "Distortion": distortion_img,
    }


def display_augmentations(
    original: np.ndarray,
    augmentations: dict[str, np.ndarray],
):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.canvas.manager.set_window_title("Image Augmentation")
    fig.suptitle("Original and Augmented Images", fontsize=14)

    ordered = [("Original", original)] + list(augmentations.items())

    for idx, (name, image) in enumerate(ordered):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(name, fontsize=10)
        ax.axis("off")

    axes[1, 3].axis("off")
    fig.tight_layout()


def save_augmentations(
    input_path: str,
    augmentations: dict[str, np.ndarray],
):
    directory, filename = os.path.split(input_path)
    stem, ext = os.path.splitext(filename)

    for name, image in augmentations.items():
        output_name = f"{stem}_{name}{ext}"
        output_path = os.path.join(directory, output_name)
        io.imsave(output_path, img_as_ubyte(image))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: ./Augmentation.py <image_path>")
        sys.exit(1)

    input_path = sys.argv[1]

    if not os.path.isfile(input_path):
        print(f"Error: '{input_path}' is not a valid file.")
        sys.exit(1)

    _, ext = os.path.splitext(input_path)

    if ext not in IMAGE_EXTENSIONS:
        print(f"Error: unsupported image extension '{ext}'.")
        sys.exit(1)

    image = img_as_float32(io.imread(input_path))

    augmentations = augment_image(image)
    display_augmentations(image, augmentations)
    save_augmentations(input_path, augmentations)

    print(f"Saved {len(augmentations)} augmented images in:")
    print(os.path.dirname(input_path) or ".")

    plt.show()
