import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import sys

def debug_refined_points(image_id, output_dir="output"):
    """
    Analyze output and check intermediate results to debug empty refined_points
    """
    # Setup paths
    home = os.getcwd()
    data_path = os.path.join(home, 'data', 'NHANES2', 'Vertebrae')
    img_path = os.path.join(data_path, 'imgs', f'{image_id}.jpg')
    mask_path = os.path.join(data_path, output_dir, 'masks', f'{image_id}.pt')
    extra_path = os.path.join(data_path, output_dir, 'extras', f'{image_id}.pkl')
    
    # Load data
    print(f"Loading data from {extra_path}...")
    with open(extra_path, 'rb') as f:
        extras = pickle.load(f)
    
    # Load masks
    print(f"Loading masks from {mask_path}...")
    masks = torch.load(mask_path)
    
    # Analyze masks and calculate centroids manually
    print("Analyzing masks to manually compute centroids...")
    masks_sigmoid = torch.sigmoid(masks).cpu().numpy() > 0.5
    
    # Calculate centroids for each mask manually
    manual_centroids = []
    for i in range(masks.shape[0]):
        mask = masks_sigmoid[i, 0]
        
        # Skip if mask is empty
        if not np.any(mask):
            print(f"Mask {i} is empty")
            continue
        
        # Find coordinates of mask pixels
        y_coords, x_coords = np.where(mask)
        
        # Calculate centroid
        if len(x_coords) > 0 and len(y_coords) > 0:
            centroid_x = np.mean(x_coords)
            centroid_y = np.mean(y_coords)
            manual_centroids.append((centroid_x, centroid_y))
            print(f"Mask {i} centroid: ({centroid_x:.2f}, {centroid_y:.2f})")
        else:
            print(f"Mask {i} has no positive pixels")
    
    print(f"\nManually calculated centroids: {manual_centroids}")
    print(f"Original extras['refined_points']: {extras['refined_points']}")
    
    # Check if starting_points exist and were used correctly
    if 'starting_points' in extras:
        print(f"\nStarting points: {extras['starting_points']}")
    
    # Check if there were issues with the weighted centroid calculation
    print("\nAttempting to trace potential issues:")
    
    # Load original image for visualization
    img = np.array(Image.open(img_path).convert('RGB'))
    
    # Create visualization
    fig, axs = plt.subplots(1, 4, figsize=(20, 8))

    # In your visualization script
    with open(os.path.join(data_path, output_dir, 'extras', f'{image_id}_fixed.pkl'), 'rb') as f:
        fixed_extras = pickle.load(f)

    if 'debug_centroids' in fixed_extras and fixed_extras['debug_centroids']:
        manual_centroids = fixed_extras['debug_centroids']
    # Plot these centroids on your visualization
    
    # Original image with starting points
    axs[0].imshow(img)
    axs[0].set_title("Original Image with Starting Points")
    
    if 'starting_points' in extras:
        for id_key, points in extras['starting_points'].items():
            for i, (x, y) in enumerate(points):
                axs[0].scatter(x, y, color='yellow', marker='*', s=200)
                axs[0].text(x+10, y+10, f"S{i+1}", fontsize=12, color='white', 
                          bbox=dict(facecolor='black', alpha=0.7))
    
    # Mask overlays
    for i in range(min(3, masks.shape[0])):
        mask = masks_sigmoid[i, 0]
        axs[i+1].imshow(img)
        axs[i+1].imshow(mask, alpha=0.5, cmap='jet')
        
        # Plot manually calculated centroid
        if i < len(manual_centroids):
            cx, cy = manual_centroids[i]
            axs[i+1].scatter(cx, cy, color='white', marker='o', s=100, edgecolor='black')
            axs[i+1].text(cx+10, cy+10, f"C{i+1}", fontsize=12, color='white', 
                        bbox=dict(facecolor='black', alpha=0.7))
        
        axs[i+1].set_title(f"Mask {i+1}")
    
    # Save figure
    save_dir = os.path.join(data_path, output_dir, 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    debug_path = os.path.join(save_dir, f'{image_id}_debug_centroids.png')
    plt.savefig(debug_path)
    print(f"\nDebug visualization saved to: {debug_path}")
    
    # Update extras with manual centroids and save
    extras['manual_centroids'] = manual_centroids
    debug_extra_path = os.path.join(data_path, output_dir, 'extras', f'{image_id}_debug.pkl')
    with open(debug_extra_path, 'wb') as f:
        pickle.dump(extras, f)
    print(f"Updated extras saved to: {debug_extra_path}")

    # Issue analysis
    print("\nPossible issues:")
    print("1. compute_weighted_centroid() function might have failed")
    print("2. refined_points was incorrectly overwritten during the algorithm")
    print("3. There may be a condition that prevents adding points to refined_points")
    print("4. The torch.sigmoid() might not behave as expected with the mask format")
    print("\nSuggestion: Compare manually calculated centroids with the visualization")
    
    return manual_centroids

if __name__ == "__main__":
    image_id = sys.argv[1] if len(sys.argv) > 1 else "image0"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output"
    debug_refined_points(image_id, output_dir)