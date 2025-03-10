import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import pprint

def explore_data(image_id, output_dir="output"):
    """
    Explore the content of the output files and visualize them
    """
    # Setup paths
    home = os.getcwd()
    data_path = os.path.join(home, 'data', 'NHANES2', 'Vertebrae')
    img_path = os.path.join(data_path, 'imgs', f'{image_id}.jpg')
    mask_path = os.path.join(data_path, output_dir, 'masks', f'{image_id}.pt')
    extra_path = os.path.join(data_path, output_dir, 'extras', f'{image_id}.pkl')
    
    # Load image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    
    # Load masks
    masks = torch.load(mask_path)
    
    # Print mask info
    print(f"Masks shape: {masks.shape}")
    print(f"Masks type: {masks.dtype}")
    
    # Load extras and print its structure
    with open(extra_path, 'rb') as f:
        extras = pickle.load(f)
    
    print("\nExtras content:")
    pprint.pprint(extras)
    
    # Convert masks to binary
    if hasattr(masks, 'numpy'):  # If it's a torch tensor
        masks_np = torch.sigmoid(masks).cpu().numpy() > 0.5
    else:  # If it's already a numpy array
        masks_np = masks > 0.5
    
    # Setup plot
    plt.figure(figsize=(12, 16))
    
    # Show original image
    plt.imshow(img_np)
    
    # Overlay each mask with a different color
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple']
    
    if len(masks_np.shape) == 4:  # [N, C, H, W]
        num_masks = masks_np.shape[0]
        for i in range(num_masks):
            color_idx = i % len(colors)
            mask_overlay = np.zeros((*img_np.shape[:2], 4))
            mask_overlay[masks_np[i, 0]] = (*plt.cm.colors.to_rgba(colors[color_idx])[:3], 0.3)
            plt.imshow(mask_overlay, alpha=0.5)
    else:
        print(f"Unexpected mask shape: {masks_np.shape}")
    
    # Plot all points found in extras with different styles
    for key, value in extras.items():
        if isinstance(value, dict):
            # For nested dictionaries
            for subkey, points in value.items():
                if isinstance(points, (list, tuple)) and len(points) > 0 and isinstance(points[0], (list, tuple)):
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    plt.scatter(x_coords, y_coords, label=f"{key}-{subkey}", marker='o', s=100, alpha=0.7)
        elif isinstance(value, (list, tuple)) and len(value) > 0:
            # For direct lists of points
            if isinstance(value[0], (list, tuple)) and len(value[0]) == 2:
                x_coords = [p[0] for p in value]
                y_coords = [p[1] for p in value]
                plt.scatter(x_coords, y_coords, label=key, marker='X', s=100, alpha=0.7)
                # Add labels to points
                for i, (x, y) in enumerate(zip(x_coords, y_coords)):
                    plt.text(x+10, y+10, f"{i+1}", fontsize=12, color='white', 
                            bbox=dict(facecolor='black', alpha=0.7))
    
    plt.title(f'Vertebrae Segmentation for {image_id} with Data Points')
    plt.legend(loc='best')
    plt.axis('off')
    
    # Save the visualization
    save_dir = os.path.join(data_path, output_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{image_id}_detailed_viz.png'), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import sys
    image_id = sys.argv[1] if len(sys.argv) > 1 else "image0"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    explore_data(image_id, output_dir)