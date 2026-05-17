"""
RGBXYZ Demo - Voxel Geometry to Image Mapper
---------------------------------------------
Technique 1: KDTree XYZ-to-RGB  (nearest-neighbor color match)
Technique 2: Luma Sort           (brightness-matched voxel colors)

Usage:
  python3 demo.py                          # uses sample heart + gradient image
  python3 demo.py heart.json photo.jpg     # uses your own files
"""

import sys
import json
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
import matplotlib.pyplot as plt


# ── helpers ────────────────────────────────────────────────────────────────────

def parse_voxels(data):
    """Parse voxel JSON into a float numpy array of shape (N, 3)."""
    coords = []
    source = data.get('voxels', data)
    for v in source:
        try:
            if isinstance(v, dict):
                coords.append([
                    float(v.get('x', v.get('X', 0))),
                    float(v.get('y', v.get('Y', 0))),
                    float(v.get('z', v.get('Z', 0))),
                ])
            elif isinstance(v, list) and len(v) >= 3:
                coords.append([float(i) for i in v[:3]])
        except (ValueError, TypeError):
            continue
    return np.array(coords)


def xyz_to_rgb_palette(coords):
    """Normalize XYZ coordinates into an RGB palette (0-255)."""
    min_c = coords.min(axis=0)
    range_c = coords.max(axis=0) - min_c
    range_c[range_c == 0] = 1
    return ((coords - min_c) / range_c * 255).astype(np.uint8)


def luma(rgb):
    """Perceived brightness: 0.299R + 0.587G + 0.114B"""
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


# ── technique 1: KDTree XYZ-to-RGB ────────────────────────────────────────────

def run_kdtree(data, img):
    print("\n[ Technique 1: KDTree XYZ-to-RGB ]")

    coords = parse_voxels(data)
    print(f"  Voxels loaded: {len(coords)}")

    palette = xyz_to_rgb_palette(coords)

    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape
    pixels = img_array.reshape(-1, 3)

    print("  Building KDTree and finding nearest-neighbor colors...")
    tree = KDTree(palette)
    _, indices = tree.query(pixels)

    result = palette[indices].reshape(h, w, 3)
    Image.fromarray(result).save("output_kdtree.png")
    print("  Saved: output_kdtree.png")
    return result


# ── technique 2: luma sort ─────────────────────────────────────────────────────

def run_luma_sort(data, img):
    print("\n[ Technique 2: Luma Sort ]")

    coords = parse_voxels(data)
    print(f"  Voxels loaded: {len(coords)}")

    voxel_colors = xyz_to_rgb_palette(coords)

    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape
    flat = img_array.reshape(-1, 3)
    n_pixels = len(flat)
    n_voxels = len(voxel_colors)
    print(f"  Image pixels: {n_pixels} | Voxel colors: {n_voxels}")

    # resample voxel palette to match pixel count
    if n_voxels < n_pixels:
        factor = int(np.ceil(n_pixels / n_voxels))
        voxel_colors = np.tile(voxel_colors, (factor, 1))[:n_pixels]
    elif n_voxels > n_pixels:
        idx = np.random.choice(n_voxels, n_pixels, replace=False)
        voxel_colors = voxel_colors[idx]

    print("  Sorting pixels by brightness...")
    img_lum = luma(flat)
    vox_lum = luma(voxel_colors)

    img_ranks = np.argsort(np.argsort(img_lum))
    sorted_voxels = voxel_colors[np.argsort(vox_lum)]
    final_pixels = sorted_voxels[img_ranks]

    result = final_pixels.reshape(h, w, 3)
    Image.fromarray(result).save("output_luma_sort.png")
    print("  Saved: output_luma_sort.png")
    return result


# ── sample data generators ─────────────────────────────────────────────────────

def make_heart_voxels(res=20):
    voxels = []
    for x in np.linspace(-1.5, 1.5, res):
        for y in np.linspace(-1.5, 1.5, res):
            for z in np.linspace(-1.5, 1.5, res):
                if (x**2 + (9/4)*y**2 + z**2 - 1)**3 - x**2*z**3 - (9/200)*y**2*z**3 <= 0:
                    voxels.append({"x": float(x), "y": float(y), "z": float(z)})
    return {"voxels": voxels}


def make_gradient_image(w=300, h=300):
    x = np.linspace(0, 255, w, dtype=np.uint8)
    y = np.linspace(0, 255, h, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    return Image.fromarray(
        np.stack([xx, yy, (255 - xx).astype(np.uint8)], axis=2), 'RGB'
    )


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 50)
    print("  RGBXYZ: Voxel Geometry -> Image Mapper")
    print("=" * 50)

    # Load inputs
    if len(sys.argv) == 3:
        json_path, img_path = sys.argv[1], sys.argv[2]
        print(f"\nUsing files: {json_path}, {img_path}")
        with open(json_path) as f:
            data = json.load(f)
        img = Image.open(img_path)
    else:
        print("\nNo files given — generating sample heart voxels + gradient image.")
        print("To use your own: python3 demo.py heart.json photo.jpg\n")
        data = make_heart_voxels(res=20)
        img = make_gradient_image()
        print(f"  Generated {len(data['voxels'])} voxels.")

    # Run both techniques
    kdtree_result = run_kdtree(data, img)
    luma_result   = run_luma_sort(data, img)

    # Side-by-side comparison plot
    print("\n[ Displaying Results ]")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("RGBXYZ: Voxel Geometry to Image Mapper", fontsize=14, fontweight='bold')

    for ax, image, title in zip(axes,
        [np.array(img.convert('RGB')), kdtree_result, luma_result],
        ["Original Image", "Technique 1: KDTree\n(Nearest-Neighbor Color Match)",
                           "Technique 2: Luma Sort\n(Brightness-Matched Voxel Colors)"]):
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("demo_comparison.png", dpi=150, bbox_inches='tight')
    print("  Saved: demo_comparison.png")
    plt.show()

    print("\nDone! Output files:")
    print("  output_kdtree.png    - KDTree color mapping")
    print("  output_luma_sort.png - Luma sort result")
    print("  demo_comparison.png  - Side-by-side comparison")


if __name__ == "__main__":
    main()
