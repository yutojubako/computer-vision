from typing import List

import cv2
import imageio
import numpy as np

from triangulation import get_triangles


# All the functions (apply_affine_transform, morph_triangle, morph_images) remain same
def apply_affine_transform(src, src_tri, dst_tri, size) -> np.ndarray :
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warp_mat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return dst


def morph_triangle(img1, img2, img, t1, t2, t, alpha) -> None :
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t_rect = []

    for i in range(0, 3):
        t_rect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t_rect), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]
    img2_rect = img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]]

    size = (r[2], r[3])
    warp_image1 = apply_affine_transform(img1_rect, t1_rect, t_rect, size)
    warp_image2 = apply_affine_transform(img2_rect, t2_rect, t_rect, size)

    # Alpha blend rectangular patches
    img_rect = (1.0 - alpha) * warp_image1 + alpha * warp_image2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] = (
        img[r[1] : r[1] + r[3], r[0] : r[0] + r[2]] * (1 - mask) + img_rect * mask
    )


def morph_images(img1, img2, points1, points2, alpha) -> np.ndarray:
    # Compute weighted average point coordinates
    points = (1 - alpha) * points1 + alpha * points2

    # Allocate space for final output
    img_morphed = np.zeros(img1.shape, dtype=img1.dtype)

    # Get the triangulations
    tri1 = get_triangles(points1)

    # Read triangles from tri.txt
    for triangle in tri1.simplices:
        x = triangle[0]
        y = triangle[1]
        z = triangle[2]

        t1 = [points1[x], points1[y], points1[z]]
        t2 = [points2[x], points2[y], points2[z]]
        t = [points[x], points[y], points[z]]

        # Morph one triangle at a time.
        morph_triangle(img1, img2, img_morphed, t1, t2, t, alpha)

    return img_morphed


def create_morph_frames(
    image1, image2, points1, points2, num_frames
) -> List[np.ndarray]:
    frames = []
    for i in range(num_frames):
        alpha = i / (num_frames - 1)
        morphed_image = morph_images(image1, image2, points1, points2, alpha)
        frames.append(morphed_image)
    return frames


def save_gif(frames, path) -> None:
    # Convert frames to uint8 (expected by imageio)
    frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    imageio.mimsave(path, frames, "GIF", dulation=20)  # adjust fps as needed
