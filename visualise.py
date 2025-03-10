import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import argparse

def visualize_prediction(image_id, output_dir="output"):
    """
    Visualize the vertebrae masks overlaid on the original image.
    
    Args:
        image_id (str): ID of the image (without extension)
        output_dir (str): Directory where outputs were saved
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
    masks_np = torch.sigmoid(masks).cpu().numpy() > 0.9  # Convert logits to binary masks
    
    # Load extras for points
    with open(extra_path, 'rb') as f:
        extras = pickle.load(f)
    
    # Setup plot
    plt.figure(figsize=(12, 16))
    
    # Show original image
    plt.imshow(img_np, cmap='gray')
    
    # Overlay each mask with a different color and transparency
    colors = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'purple', 'brown', 'pink']
    for i, mask in enumerate(masks_np):
        color = colors[i % len(colors)]
        mask_overlay = np.zeros((*mask.shape, 4))  # RGBA
        mask_overlay[mask == 1] = plt.cm.colors.to_rgba(color, alpha=0.3)
        plt.imshow(mask_overlay[0], alpha=0.5)  # Adjust alpha for visibility
    
    # Plot points if available
    if 'refined_points' in extras:
        points = extras['refined_points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        plt.scatter(x_coords, y_coords, c='white', s=50, alpha=0.8, edgecolors='black')
    
    # # Plot starting points
    # if 'starting_points' in extras:
    #     for points_set in extras['starting_points'].values():  # This is a dictionary
    #         for point in points_set:
    #             plt.scatter(point[0], point[1], c='yellow', s=80, marker='*', alpha=0.8, edgecolors='black')
    
    # Highlight landmark point if available
    if 'landmark_centroid' in extras and extras['landmark_centroid'] != (0, 0):
        lm = extras['landmark_centroid']
        plt.scatter([lm[0]], [lm[1]], c='red', s=100, marker='X', alpha=0.8, edgecolors='black')
        plt.annotate("Landmark", (lm[0], lm[1]), color='white', fontsize=12, 
                    xytext=(10, 10), textcoords='offset points')
    
    plt.title(f'Vertebrae Segmentation for {image_id}')
    plt.axis('off')
    
    # Save the visualization
    save_dir = os.path.join(data_path, output_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'{image_id}_visualization.png'), bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize vertebrae segmentation')
    parser.add_argument('image_id', help='ID of the image to visualize')
    parser.add_argument('--output_dir', default='output', help='Output directory name')
    args = parser.parse_args()
    
    visualize_prediction(args.image_id, args.output_dir)