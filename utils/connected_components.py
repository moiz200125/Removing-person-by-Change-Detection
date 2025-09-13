import numpy as np
from collections import deque

def find_connected_components(mask, connectivity=8):
    """
    Finds connected components in binary mask using BFS.

    Parameters:
        mask (np.ndarray): Binary mask [H, W], values 0 or 255
        connectivity (int): 4 or 8 connectivity

    Returns:
        tuple:
            num_components (int): number of connected components
            labeled_mask (np.ndarray): [H, W] mask with labels 0..num_components
            component_info (list of dict): info for each component
    """
    H, W = mask.shape
    labeled = np.zeros((H, W), dtype=np.int32)
    visited = np.zeros((H, W), dtype=bool)
    comp_id = 0
    component_info = []

    # Neighbor offsets
    if connectivity == 4:
        neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    else:  # 8-connectivity
        neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                     (-1,-1),(-1,1),(1,-1),(1,1)]

    for y in range(H):
        for x in range(W):
            if mask[y, x] == 255 and not visited[y, x]:
                comp_id += 1
                q = deque()
                q.append((y, x))
                visited[y, x] = True
                labeled[y, x] = comp_id

                pixels = []
                while q:
                    cy, cx = q.popleft()
                    pixels.append((cy, cx))

                    for dy, dx in neighbors:
                        ny, nx = cy+dy, cx+dx
                        if 0 <= ny < H and 0 <= nx < W:
                            if mask[ny, nx] == 255 and not visited[ny, nx]:
                                visited[ny, nx] = True
                                labeled[ny, nx] = comp_id
                                q.append((ny, nx))

                # Compute stats
                pixels = np.array(pixels)
                area = len(pixels)
                centroid = (float(pixels[:,1].mean()), float(pixels[:,0].mean()))  # (x,y)
                y_min, x_min = pixels[:,0].min(), pixels[:,1].min()
                y_max, x_max = pixels[:,0].max(), pixels[:,1].max()
                bbox = (x_min, y_min, x_max, y_max)

                component_info.append({
                    "id": comp_id,
                    "area": area,
                    "centroid": centroid,
                    "bbox": bbox
                })

    return comp_id, labeled, component_info
