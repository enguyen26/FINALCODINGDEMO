"""
RGBXYZ Demo - Voxel Geometry to Image Mapper
Standalone version (no Google Colab required)

Demonstrates 3 techniques:
  1. KDTree XYZ-to-RGB mapping
  2. Luma Sort (brightness-matched voxel coloring)
  3. Un-Normalizer (RGB back to 3D point cloud)
"""

import json
import numpy as np
from PIL import Image
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import io
import os

# ──────────────────────────────────────────────
# GENERATE SAMPLE DATA (heart-like voxel model)
# ──────────────────────────────────────────────

def generate_heart_voxels(resolution=20):
    """Parametric heart shape in 3D voxel space."""
    coords = []
    for x in np.linspace(-1.5, 1.5, resolution):
        for y in np.linspace(-1.5, 1.5, resolution):
            for z in np.linspace(-1.5, 1.5, resolution):
                # Heart surface equation (approximate)
                val = (x**2 + (9/4)*y**2 + z**2 - 1)**3 - x**2 * z**3 - (9/200)*y**2 * z**3
                if val <= 0:
                    coords.append({"x": float(x), "y": float(y), "z": float(z)})
    return {"voxels": coords}


def generate_sample_image(width=200, height=200):
    """Create a simple gradient test image."""
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    xx, yy = np.meshgrid(x, y)
    img_array = np.stack([xx, yy, (255 - xx).astype(np.uint8)], axis=2)
    return Image.fromarray(img_array, 'RGB')


# ──────────────────────────────────────────────
# CELL 1: KDTree XYZ-to-RGB Mapper
# ──────────────────────────────────────────────

def run_xyz_to_rgb(data, img):
    print("\n=== STEP 1: KDTree XYZ-to-RGB Mapper ===")

    coords = []
    source_list = data.get('voxels', data)
    for v in source_list:
        if isinstance(v, dict):
            x = float(v.get('x', v.get('X', 0)))
            y = float(v.get('y', v.get('Y', 0)))
            z = float(v.get('z', v.get('Z', 0)))
            coords.append([x, y, z])
        elif isinstance(v, list) and len(v) >= 3:
            coords.append([float(i) for i in v[:3]])

    coords = np.array(coords)
    print(f"   Found {len(coords)} voxels in model.")

    # Normalize XYZ -> RGB palette
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    range_c = max_c - min_c
    range_c[range_c == 0] = 1
    palette = ((coords - min_c) / range_c * 255).astype(np.uint8)

    # Map image pixels to nearest palette color
    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape
    target_pixels = img_array.reshape(-1, 3)

    print("   Building KDTree and mapping pixels...")
    tree = KDTree(palette)
    _, indices = tree.query(target_pixels)

    new_pixels = palette[indices]
    final_array = new_pixels.reshape(h, w, 3)

    out = Image.fromarray(final_array)
    out.save("output_kdtree.png")
    print("   Saved: output_kdtree.png")
    return final_array


# ──────────────────────────────────────────────
# CELL 2: Luma Sort
# ──────────────────────────────────────────────

def calculate_brightness(rgb_array):
    return np.dot(rgb_array[..., :3], [0.299, 0.587, 0.114])


def run_luma_sort(data, img):
    print("\n=== STEP 2: Luma Sort (Brightness Matching) ===")

    coords = []
    source_list = data.get('voxels', data)
    for v in source_list:
        try:
            if isinstance(v, dict):
                x = float(v.get('x', v.get('X', 0)))
                y = float(v.get('y', v.get('Y', 0)))
                z = float(v.get('z', v.get('Z', 0)))
                coords.append([x, y, z])
            elif isinstance(v, list) and len(v) >= 3:
                coords.append([float(i) for i in v[:3]])
        except ValueError:
            continue

    coords = np.array(coords)
    print(f"   Found {len(coords)} valid voxels.")

    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    range_c = max_c - min_c
    range_c[range_c == 0] = 1
    voxel_colors = ((coords - min_c) / range_c * 255).astype(np.uint8)

    img_array = np.array(img.convert('RGB'))
    h, w, _ = img_array.shape
    flat_img = img_array.reshape(-1, 3)
    num_pixels = len(flat_img)
    num_voxels = len(voxel_colors)
    print(f"   Image pixels: {num_pixels} | Voxel colors: {num_voxels}")

    # Resample voxels to match pixel count
    if num_voxels < num_pixels:
        repeat_factor = int(np.ceil(num_pixels / num_voxels))
        voxel_colors = np.tile(voxel_colors, (repeat_factor, 1))[:num_pixels]
    elif num_voxels > num_pixels:
        indices = np.random.choice(num_voxels, num_pixels, replace=False)
        voxel_colors = voxel_colors[indices]

    # Sort by brightness and map
    print("   Sorting by luminance...")
    img_lum = calculate_brightness(flat_img)
    vox_lum = calculate_brightness(voxel_colors)
    img_sort_order = np.argsort(img_lum)
    vox_sort_order = np.argsort(vox_lum)
    img_ranks = np.argsort(img_sort_order)
    sorted_voxels = voxel_colors[vox_sort_order]
    final_pixels = sorted_voxels[img_ranks]

    final_array = final_pixels.reshape(h, w, 3)
    out = Image.fromarray(final_array)
    out.save("output_luma_sort.png")
    print("   Saved: output_luma_sort.png")
    return final_array


# ──────────────────────────────────────────────
# CELL 3: Un-Normalizer (Image -> Point Cloud)
# ──────────────────────────────────────────────

def run_unnormalize(data, img):
    print("\n=== STEP 3: Un-Normalizer (RGB -> 3D Point Cloud) ===")

    coords = []
    source_list = data.get('voxels', data)
    for v in source_list:
        if isinstance(v, dict):
            coords.append([
                float(v.get('x', 0)),
                float(v.get('y', 0)),
                float(v.get('z', 0))
            ])
        elif isinstance(v, list) and len(v) >= 3:
            coords.append([float(i) for i in v[:3]])

    coords = np.array(coords)
    min_c = coords.min(axis=0)
    max_c = coords.max(axis=0)
    range_c = max_c - min_c
    range_c[range_c == 0] = 1
    print(f"   Original bounds: min={min_c.round(2)}, max={max_c.round(2)}")

    img_array = np.array(img.convert('RGB'))
    pixels = img_array.reshape(-1, 3)

    # Reverse normalization: RGB -> XYZ
    new_coords = (pixels / 255.0) * range_c + min_c

    ply_filename = "reconstructed_pointcloud.ply"
    with open(ply_filename, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(new_coords)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(new_coords)):
            x, y, z = new_coords[i]
            r, g, b = pixels[i]
            f.write(f"{x:.3f} {y:.3f} {z:.3f} {r} {g} {b}\n")

    print(f"   Saved: {ply_filename}  ({len(new_coords)} points)")
    return ply_filename


# ──────────────────────────────────────────────
# VISUALIZE ALL RESULTS
# ──────────────────────────────────────────────

def show_results(original_img, kdtree_result, luma_result):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("RGBXYZ: Voxel Geometry to Image Mapper", fontsize=14, fontweight='bold')

    axes[0].imshow(np.array(original_img.convert('RGB')))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(kdtree_result)
    axes[1].set_title("KDTree XYZ-to-RGB\n(Nearest-neighbor color match)")
    axes[1].axis('off')

    axes[2].imshow(luma_result)
    axes[2].set_title("Luma Sort\n(Brightness-matched voxel colors)")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig("demo_comparison.png", dpi=150, bbox_inches='tight')
    print("\n   Saved: demo_comparison.png")
    plt.show()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("   RGBXYZ Voxel Demo")
    print("=" * 50)

    # Check if user has their own files
    json_path = "heart.json"
    img_path = "IMG_2317.jpg"

    if os.path.exists(json_path) and os.path.exists(img_path):
        print(f"\nUsing your files: {json_path} and {img_path}")
        with open(json_path, 'r') as f:
            data = json.load(f)
        img = Image.open(img_path)
    else:
        print("\nNo input files found — generating sample heart voxel model + gradient image.")
        print("(To use your own files, place heart.json and your image in this folder)")
        data = generate_heart_voxels(resolution=20)
        img = generate_sample_image(width=200, height=200)
        print(f"   Generated {len(data['voxels'])} voxels.")

    # Run all 3 techniques
    kdtree_result = run_xyz_to_rgb(data, img)
    luma_result = run_luma_sort(data, img)
    ply_file = run_unnormalize(data, img)

    # Show side-by-side comparison
    print("\n=== Displaying Results ===")
    show_results(img, kdtree_result, luma_result)

    print("\nDone! Output files:")
    print("  output_kdtree.png      - KDTree color mapping")
    print("  output_luma_sort.png   - Luma sort result")
    print("  reconstructed_pointcloud.ply - 3D point cloud (open in Blender/MeshLab)")
    print("  demo_comparison.png    - Side-by-side comparison")
