"""Lightweight checks for Grad-CAM rendering helpers."""

import base64
import unittest

try:
    import numpy as np

    from retinastage.explainability import (
        colourise_heatmap,
        encode_rgb_png_data_url,
    )
except ImportError:  # pragma: no cover - optional local test environment
    np = None
    colourise_heatmap = None
    encode_rgb_png_data_url = None


@unittest.skipIf(np is None, "Explainability dependencies are not installed")
class ExplainabilityTests(unittest.TestCase):
    def test_colourised_heatmap_is_rgb_and_bounded(self) -> None:
        heatmap = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        coloured = colourise_heatmap(heatmap)
        self.assertEqual(coloured.shape, (4, 4, 3))
        self.assertGreaterEqual(float(coloured.min()), 0.0)
        self.assertLessEqual(float(coloured.max()), 1.0)

    def test_png_data_url_contains_png_signature(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.float32)
        data_url = encode_rgb_png_data_url(image)
        prefix = "data:image/png;base64,"
        self.assertTrue(data_url.startswith(prefix))
        payload = base64.b64decode(data_url[len(prefix) :])
        self.assertEqual(payload[:8], b"\x89PNG\r\n\x1a\n")


if __name__ == "__main__":
    unittest.main()
