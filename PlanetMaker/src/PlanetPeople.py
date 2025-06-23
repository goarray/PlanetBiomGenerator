import numpy as np
from scipy.spatial import KDTree
import heapq


def generate_road_mask(
    colony_mask: np.ndarray,
    river_mask: np.ndarray,
    mountain_mask: np.ndarray,
    ocean_mask: np.ndarray,
    max_connections: int = 5,
) -> np.ndarray:
    height, width = colony_mask.shape
    road_mask = np.zeros_like(colony_mask, dtype=np.uint8)
    connection_cache = set()

    colony_coords = np.column_stack(np.where(colony_mask == 255))
    if len(colony_coords) < 2:
        print("[Roads] Not enough colonies to connect.")
        return road_mask

    tree = KDTree(colony_coords)

    # Cost grid: high cost on rivers (assuming river_mask==0 means river)
    cost_grid = np.ones_like(colony_mask, dtype=np.float32)
    cost_grid[river_mask == 0] = 1e6
    cost_grid[mountain_mask == 0] = 1e6
    cost_grid[ocean_mask == 0] = 1e6


    def dijkstra(start, end, relax_mountains=False):
        visited = set()
        heap = [(0.0, start, [])]

        # Adjusted cost grid
        local_cost_grid = cost_grid.copy()
        if relax_mountains:
            local_cost_grid[mountain_mask == 0] = (
                1.0  # simulate "cutting" through mountains
            )

        while heap:
            cost, current, path = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            path = path + [current]

            if current == end:
                return path

            y, x = current
            for dy, dx in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < height and 0 <= nx < width:
                    step_cost = local_cost_grid[ny, nx] * (
                        1.414 if abs(dy) + abs(dx) == 2 else 1
                    )
                    next_cost = cost + step_cost
                    heapq.heappush(heap, (float(next_cost), (ny, nx), path))
        return []

    connected = {tuple(coord): 0 for coord in colony_coords}

    for i, origin in enumerate(colony_coords):
        origin_tup = tuple(origin)
        if connected[origin_tup] >= max_connections:
            continue

        k = min(max_connections + 1, len(colony_coords))
        dists, indices = tree.query(origin, k=k)

        if not isinstance(indices, np.ndarray):
            indices = np.array([indices])  # Wrap scalar into 1-element array

        for j in indices[1:]:
            if j == i:
                continue  # skip self

            target = tuple(colony_coords[j])
            if (
                connected[origin_tup] >= max_connections
                or connected[target] >= max_connections
            ):
                continue

            key = tuple(sorted((origin_tup, target)))
            if key in connection_cache:
                continue

            path = dijkstra(origin_tup, target)
            if path:
                for px, py in path:
                    road_mask[px, py] = 255
                connected[origin_tup] += 1
                connected[target] += 1
                connection_cache.add(key)
            if not path:
                path = dijkstra(origin_tup, target, relax_mountains=True)

    print(f"[Roads] Connected {len(connected)} colonies with roads.")
    return road_mask
