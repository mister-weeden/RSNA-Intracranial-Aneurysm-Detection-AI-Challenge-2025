import csv
import ast
import math

def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def order_by_distance(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Parse coordinates
    coords = []
    for row in rows:
        coord_dict = ast.literal_eval(row['coordinates'])
        coords.append((coord_dict['x'], coord_dict['y']))
    
    # Nearest neighbor ordering
    ordered_indices = [0]
    remaining = set(range(1, len(coords)))
    
    while remaining:
        current_point = coords[ordered_indices[-1]]
        min_dist = float('inf')
        nearest_idx = None
        
        for idx in remaining:
            dist = euclidean_distance(current_point, coords[idx])
            if dist < min_dist:
                min_dist = dist
                nearest_idx = idx
        
        ordered_indices.append(nearest_idx)
        remaining.remove(nearest_idx)
    
    # Write ordered CSV
    with open('train_localizers_ordered.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for idx in ordered_indices:
            writer.writerow(rows[idx])
    
    return len(ordered_indices)

# Apply ordering
count = order_by_distance('train_localizers.csv')
print(f"Ordered {count} entries by distance")
