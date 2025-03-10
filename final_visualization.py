import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle

def create_final_visualization(image_id, output_dir="output"):
    # Setup paths
    home = os.getcwd()
    data_path = os.path.join(home, 'data', 'NHANES2', 'Vertebrae')
    img_path = os.path.join(data_path, 'imgs', f'{image_id}.jpg')
    mask_path = os.path.join(data_path, output_dir, 'masks', f'{image_id}.pt')
    
    # Try to load the fixed extras if available
    fixed_extra_path = os.path.join(data_path, output_dir, 'extras', f'{image_id}_fixed.pkl')
    if os.path.exists(fixed_extra_path):
        with open(fixed_extra_path, 'rb') as f:
            extras = pickle.load(f)
    else:
        with open(os.path.join(data_path, output_dir, 'extras', f'{image_id}.pkl'), 'rb') as f:
            extras = pickle.load(f)
    
    # Load image and masks
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img)
    masks = torch.load(mask_path)
    masks_np = torch.sigmoid(masks).cpu().numpy() > 0.5
    
    # Create visualization
    plt.figure(figsize=(12, 16))
    plt.imshow(img_np)
    
    # Overlay masks with different colors
    colors = ['red', 'green', 'blue']
    for i in range(masks.shape[0]):
        mask = masks_np[i, 0]
        
        # Create colored mask overlay
        color_mask = np.zeros((*mask.shape, 4))
        color_mask[mask] = plt.cm.colors.to_rgba(colors[i % len(colors)], alpha=0.4)
        plt.imshow(color_mask, alpha=0.5)
    
    # Plot starting points (which are actually the calculated centroids)
    if 'starting_points' in extras and image_id in extras['starting_points']:
        points = extras['starting_points'][image_id]
        for i, (x, y) in enumerate(points):
            plt.scatter(x, y, color='yellow', marker='*', s=200, edgecolor='black')
            plt.annotate(f"V{i+1}", (x+10, y+10), color='white', fontsize=14,
                        bbox=dict(facecolor='black', alpha=0.7))
    
    # Plot debug centroids if available
    if 'debug_centroids' in extras:
        points = extras['debug_centroids']
        for i, (x, y) in enumerate(points):
            plt.scatter(x, y, color='white', marker='o', s=120, edgecolor='black')
            plt.annotate(f"C{i+1}", (x+10, y-10), color='black', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.7))
    
    # Plot rough points
    if 'rough_points' in extras and extras['rough_points']:
        for i, point in enumerate(extras['rough_points']):
            x, y = float(point[0]), float(point[1])
            plt.scatter(x, y, color='cyan', marker='x', s=150, edgecolor='black')
            plt.annotate(f"R{i+1}", (x+10, y+10), color='white', fontsize=12,
                        bbox=dict(facecolor='blue', alpha=0.7))
    
    plt.title(f'Vertebrae Segmentation with Centroids - {image_id}')
    plt.axis('off')
    
    # Save the visualization
    save_dir = os.path.join(data_path, output_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{image_id}_final_visualization.png'), bbox_inches='tight')
    print(f"Final visualization saved to: {os.path.join(save_dir, f'{image_id}_final_visualization.png')}")

if __name__ == "__main__":
    import sys
    image_id = sys.argv[1] if len(sys.argv) > 1 else "image0"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    create_final_visualization(image_id, output_dir)