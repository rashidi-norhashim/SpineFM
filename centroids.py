import torch
import numpy as np
import os
import pickle
from utils import compute_weighted_centroid  # Import the actual function

# Load the mask file
home = os.getcwd()
data_path = os.path.join(home, 'data', 'NHANES2', 'Vertebrae')
mask_path = os.path.join(data_path, 'output', 'masks', 'image0.pt')
masks = torch.load(mask_path)

# Try to compute centroids using the original function
print("Testing original compute_weighted_centroid function...")
centroids = []
for i in range(masks.shape[0]):
    mask_np = np.array(torch.sigmoid(masks[i]).squeeze(0))
    try:
        centroid = compute_weighted_centroid(mask_np)
        centroids.append(centroid)
        print(f"Mask {i}: Centroid = {centroid}")
    except Exception as e:
        print(f"Error computing centroid for mask {i}: {e}")

# Implement our own centroid computation for validation
print("\nComputing centroids manually for validation...")
manual_centroids = []
for i in range(masks.shape[0]):
    mask = (torch.sigmoid(masks[i]).squeeze(0).cpu().numpy() > 0.5).astype(np.float32)
    
    # Find non-zero pixel coordinates
    y_indices, x_indices = np.nonzero(mask)
    
    if len(x_indices) > 0 and len(y_indices) > 0:
        # Calculate centroid as average of coordinates
        centroid_x = np.mean(x_indices)
        centroid_y = np.mean(y_indices)
        manual_centroids.append((centroid_x, centroid_y))
        print(f"Manual centroid for mask {i}: ({centroid_x:.2f}, {centroid_y:.2f})")
    else:
        print(f"Mask {i} has no positive pixels")

# Save the computed centroids
extra_path = os.path.join(data_path, 'output', 'extras', 'image0.pkl')
with open(extra_path, 'rb') as f:
    extras = pickle.load(f)

extras['debug_centroids'] = manual_centroids
with open(os.path.join(data_path, 'output', 'extras', 'image0_fixed.pkl'), 'wb') as f:
    pickle.dump(extras, f)

print(f"\nOriginal centroid calculation returned {len(centroids)} centroids")
print(f"Manual centroid calculation returned {len(manual_centroids)} centroids")
print("Updated extras saved with debug_centroids")