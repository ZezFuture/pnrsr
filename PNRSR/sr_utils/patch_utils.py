
import numpy as np

def get_patch_positions(length, patch_size, stride):
    if length < patch_size:
        raise ValueError(
            f"Image dimension {length} is smaller than patch size {patch_size}"
        )

    positions = list(range(0, length - patch_size + 1, stride))
    last_position = length - patch_size

    if not positions:
        positions = [0]
    elif positions[-1] != last_position:
        positions.append(last_position)

    return np.asarray(positions, dtype=np.int64)


def image2patchs(image, patch_size, stride):
    image = image.transpose((0, 3, 1, 2))

    imheight, imwidth = image.shape[-2:]

    range_y = get_patch_positions(
        imheight,
        patch_size[0],
        stride[0],
    )
    range_x = get_patch_positions(
        imwidth,
        patch_size[1],
        stride[1],
    )

    sz = len(range_y) * len(range_x)

    res = np.zeros(
        (
            sz,
            *image.shape[:-2],
            patch_size[0],
            patch_size[1],
        ),
        dtype=np.float32,
    )

    index = 0
    for y in range_y:
        for x in range_x:
            res[index] = image[
                ...,
                y:y + patch_size[0],
                x:x + patch_size[1],
            ]
            index += 1

    return res

import numpy as np


def get_patch_positions(length, patch_size, stride):
    """
    Generate patch starting positions.

    Ensures:
    1. The first patch starts at 0.
    2. The final patch reaches the image boundary.
    3. When image size equals patch size, returns [0].
    """
    if length < patch_size:
        raise ValueError(
            f"Image dimension {length} is smaller than "
            f"patch size {patch_size}."
        )

    if stride <= 0:
        raise ValueError(
            f"Stride must be positive, but received {stride}."
        )

    positions = list(
        range(0, length - patch_size + 1, stride)
    )

    last_position = length - patch_size

    if not positions:
        positions = [0]
    elif positions[-1] != last_position:
        positions.append(last_position)

    return np.asarray(positions, dtype=np.int64)


def patchs2image(patchs, imsize, stride, padding=0, mode=1):
    """
    Merge image patches back into a complete image.

    Args:
        patchs:
            Patch array with shape
            [num_patches, ..., patch_h, patch_w].

        imsize:
            Output image size [height, width].

        stride:
            Patch stride [stride_h, stride_w].

        padding:
            Number of pixels ignored at internal patch boundaries
            when mode=1.

        mode:
            1: Average fusion with optional boundary padding.
            2: Adaptive feather blending according to actual overlap.

    Returns:
        Reconstructed image with shape [..., im_h, im_w].
    """
    patch_h, patch_w = patchs.shape[-2:]
    im_h, im_w = int(imsize[0]), int(imsize[1])

    range_y = get_patch_positions(
        im_h,
        patch_h,
        stride[0],
    )
    range_x = get_patch_positions(
        im_w,
        patch_w,
        stride[1],
    )

    expected_patches = len(range_y) * len(range_x)

    if patchs.shape[0] != expected_patches:
        raise ValueError(
            "Patch count mismatch: "
            f"received {patchs.shape[0]}, "
            f"expected {expected_patches}; "
            f"image_size={(im_h, im_w)}, "
            f"patch_size={(patch_h, patch_w)}, "
            f"stride={tuple(stride)}, "
            f"range_y={range_y.tolist()}, "
            f"range_x={range_x.tolist()}."
        )

    res = np.zeros(
        (*patchs.shape[1:-2], im_h, im_w),
        dtype=np.float32,
    )
    weight = np.zeros_like(res, dtype=np.float32)

    if mode == 1:
        index = 0

        for y in range_y:
            for x in range_x:
                y = int(y)
                x = int(x)

                # Internal edges discard `padding` pixels.
                # Image boundary edges are fully retained.
                y0 = y if y == 0 else y + padding
                y1 = (
                    y + patch_h
                    if y == im_h - patch_h
                    else y + patch_h - padding
                )

                x0 = x if x == 0 else x + padding
                x1 = (
                    x + patch_w
                    if x == im_w - patch_w
                    else x + patch_w - padding
                )

                if y1 <= y0 or x1 <= x0:
                    raise ValueError(
                        "Invalid fusion region. "
                        f"patch_position={(y, x)}, "
                        f"fusion_region={(y0, y1, x0, x1)}, "
                        f"padding={padding}."
                    )

                patch_y0 = y0 - y
                patch_y1 = y1 - y
                patch_x0 = x0 - x
                patch_x1 = x1 - x

                res[..., y0:y1, x0:x1] += patchs[index][
                    ...,
                    patch_y0:patch_y1,
                    patch_x0:patch_x1,
                ]

                weight[..., y0:y1, x0:x1] += 1.0
                index += 1

        res = res / np.maximum(weight, 1e-8)
        return res.astype(np.float32)

    if mode == 2:
        nominal_ov_y = patch_h - stride[0]
        nominal_ov_x = patch_w - stride[1]

        # If either direction has no nominal overlap,
        # use ordinary averaging.
        if nominal_ov_y <= 0 or nominal_ov_x <= 0:
            index = 0

            for y in range_y:
                for x in range_x:
                    y = int(y)
                    x = int(x)

                    y0 = y
                    y1 = y + patch_h
                    x0 = x
                    x1 = x + patch_w

                    res[..., y0:y1, x0:x1] += patchs[index][
                        ...,
                        :patch_h,
                        :patch_w,
                    ]

                    weight[..., y0:y1, x0:x1] += 1.0
                    index += 1

            res = res / np.maximum(weight, 1e-8)
            return res.astype(np.float32)

        # Adaptive feather blending based on actual patch positions.
        index = 0
        ny = len(range_y)
        nx = len(range_x)

        for iy, y in enumerate(range_y):
            y = int(y)

            y_prev = (
                int(range_y[iy - 1])
                if iy > 0
                else None
            )
            y_next = (
                int(range_y[iy + 1])
                if iy < ny - 1
                else None
            )

            dy_up = (
                y - y_prev
                if y_prev is not None
                else None
            )
            dy_down = (
                y_next - y
                if y_next is not None
                else None
            )

            ov_top = (
                patch_h - dy_up
                if dy_up is not None
                else 0
            )
            ov_bottom = (
                patch_h - dy_down
                if dy_down is not None
                else 0
            )

            ov_top = int(
                max(0, min(patch_h, ov_top))
            )
            ov_bottom = int(
                max(0, min(patch_h, ov_bottom))
            )

            w_y = np.ones(
                (patch_h,),
                dtype=np.float32,
            )

            if ov_top > 0:
                top_weight = np.linspace(
                    0.0,
                    1.0,
                    ov_top,
                    endpoint=False,
                    dtype=np.float32,
                )
                w_y[:ov_top] *= top_weight

            if ov_bottom > 0:
                bottom_weight = np.linspace(
                    1.0,
                    0.0,
                    ov_bottom,
                    endpoint=False,
                    dtype=np.float32,
                )
                w_y[-ov_bottom:] *= bottom_weight

            for ix, x in enumerate(range_x):
                x = int(x)

                x_prev = (
                    int(range_x[ix - 1])
                    if ix > 0
                    else None
                )
                x_next = (
                    int(range_x[ix + 1])
                    if ix < nx - 1
                    else None
                )

                dx_left = (
                    x - x_prev
                    if x_prev is not None
                    else None
                )
                dx_right = (
                    x_next - x
                    if x_next is not None
                    else None
                )

                ov_left = (
                    patch_w - dx_left
                    if dx_left is not None
                    else 0
                )
                ov_right = (
                    patch_w - dx_right
                    if dx_right is not None
                    else 0
                )

                ov_left = int(
                    max(0, min(patch_w, ov_left))
                )
                ov_right = int(
                    max(0, min(patch_w, ov_right))
                )

                w_x = np.ones(
                    (patch_w,),
                    dtype=np.float32,
                )

                if ov_left > 0:
                    left_weight = np.linspace(
                        0.0,
                        1.0,
                        ov_left,
                        endpoint=False,
                        dtype=np.float32,
                    )
                    w_x[:ov_left] *= left_weight

                if ov_right > 0:
                    right_weight = np.linspace(
                        1.0,
                        0.0,
                        ov_right,
                        endpoint=False,
                        dtype=np.float32,
                    )
                    w_x[-ov_right:] *= right_weight

                mask2d = (
                    w_y[:, None] * w_x[None, :]
                ).astype(np.float32)

                y0 = y
                y1 = y + patch_h
                x0 = x
                x1 = x + patch_w

                res[..., y0:y1, x0:x1] += (
                    patchs[index][
                        ...,
                        :patch_h,
                        :patch_w,
                    ]
                    * mask2d
                )

                weight[..., y0:y1, x0:x1] += mask2d
                index += 1

        res = res / np.maximum(weight, 1e-8)
        return res.astype(np.float32)

    raise ValueError(
        f"Unsupported mode={mode}; expected mode=1 or mode=2."
    )