# engine/canvas/canvas_environment.py

import math

class CanvasEnvironment:

    def __init__(self, config):
        self.width = config.get('width', 800.0)
        self.height = config.get('height', 800.0)
        self.attractors = config.get('attractors', [])  # [(x, y, strength), ...]
        self.repulsors = config.get('repulsors', [])    # [(x, y, strength), ...]
        self.noise_scale = config.get('noise_scale', 0.0)
        self.density_field = self._empty_density_field()

    def _empty_density_field(self):
        """A simple grid tracking ink density."""
        resolution = 50
        return [[0.0] * resolution for _ in range(resolution)]

    def record_position(self, x, y):
        """Mark a position as visited in the density field."""
        resolution = len(self.density_field)
        gx = int((x / self.width) * (resolution - 1))
        gy = int((y / self.height) * (resolution - 1))
        gx = max(0, min(resolution - 1, gx))
        gy = max(0, min(resolution - 1, gy))
        self.density_field[gy][gx] += 1.0

    def sample_density(self, x, y):
        """How much ink is near this position, normalized 0-1."""
        resolution = len(self.density_field)
        gx = int((x / self.width) * (resolution - 1))
        gy = int((y / self.height) * (resolution - 1))
        gx = max(0, min(resolution - 1, gx))
        gy = max(0, min(resolution - 1, gy))
        raw = self.density_field[gy][gx]
        return min(1.0, raw / 10.0)

    def attractor_inputs(self, x, y):
        """
        Return (distance, angle) to nearest attractor
        and (distance, angle) to nearest repulsor,
        all normalized. Returns zeros if none defined.
        """
        def nearest(points):
            if not points:
                return 0.0, 0.0
            best = min(
                points,
                key=lambda p: math.hypot(x - p[0], y - p[1])
            )
            dx = best[0] - x
            dy = best[1] - y
            dist = math.hypot(dx, dy)
            norm_dist = 1.0 - min(1.0, dist / math.hypot(
                self.width, self.height
            ))
            angle = math.atan2(dy, dx) / math.pi  # normalized -1 to 1
            return norm_dist, angle

        attr_dist, attr_angle = nearest(
            [(p[0], p[1]) for p in self.attractors]
        )
        rep_dist, rep_angle = nearest(
            [(p[0], p[1]) for p in self.repulsors]
        )
        return attr_dist, attr_angle, rep_dist, rep_angle

    def edge_distance(self, x, y):
        """Normalized distance from nearest edge, 0=at edge, 1=center."""
        dist = min(x, y, self.width - x, self.height - y)
        return min(1.0, dist / (min(self.width, self.height) * 0.5))

    def reset(self):
        """Clear density field between renders."""
        self.density_field = self._empty_density_field()