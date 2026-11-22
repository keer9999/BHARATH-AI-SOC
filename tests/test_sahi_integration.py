"""
Tests for SAHI (Slicing Aided Hyper Inference) detector and web_app integration.
"""
import unittest
import numpy as np
from sahi_detector import (
    SAHIConfig,
    compute_tiles,
    pad_to_square,
    weighted_box_fusion,
    hard_nms,
    SAHIWasteDetector,
)
from waste_detector import WasteDetector, WasteFrame, WasteDetection
from config import DetectionConfig


class TestSAHIGeometry(unittest.TestCase):
    def test_compute_tiles_grid(self):
        # 1280x720 with tile size 416, overlap 0.25
        tiles = compute_tiles(1280, 720, tile_size=416, overlap_ratio=0.25, max_tiles=16)
        self.assertGreater(len(tiles), 0)
        self.assertLessEqual(len(tiles), 16)
        for x1, y1, x2, y2 in tiles:
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 1280)
            self.assertLessEqual(y2, 720)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)

    def test_compute_tiles_max_cap(self):
        tiles = compute_tiles(1920, 1080, tile_size=320, overlap_ratio=0.20, max_tiles=4)
        self.assertEqual(len(tiles), 4)

    def test_pad_to_square(self):
        img = np.zeros((200, 300, 3), dtype=np.uint8)
        padded = pad_to_square(img, target=416)
        self.assertEqual(padded.shape, (416, 416, 3))


class TestSAHIFusion(unittest.TestCase):
    def test_weighted_box_fusion_basic(self):
        # 2 overlapping boxes of same class
        boxes = np.array([
            [0.10, 0.10, 0.50, 0.50],
            [0.12, 0.11, 0.52, 0.49],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        labels = np.array([0, 0], dtype=np.int32)

        f_boxes, f_scores, f_labels = weighted_box_fusion(boxes, scores, labels, iou_thr=0.45)
        self.assertEqual(len(f_boxes), 1)
        self.assertEqual(f_labels[0], 0)
        self.assertAlmostEqual(f_scores[0], 0.85, places=2)

    def test_hard_nms(self):
        boxes = np.array([
            [10, 10, 50, 50],
            [12, 12, 52, 52],
            [100, 100, 200, 200],
        ], dtype=np.float32)
        scores = np.array([0.9, 0.7, 0.8], dtype=np.float32)

        keep = hard_nms(boxes, scores, iou_thr=0.45)
        self.assertEqual(len(keep), 2)
        self.assertIn(0, keep)
        self.assertIn(2, keep)


class TestSAHIProfiles(unittest.TestCase):
    def test_hardware_profiles(self):
        rpi = SAHIConfig.raspberry_pi_4b()
        self.assertEqual(rpi.tile_size, 320)
        self.assertEqual(rpi.max_tiles, 4)
        self.assertFalse(rpi.use_weighted_box_fusion)

        gpu = SAHIConfig.laptop_gpu()
        self.assertEqual(gpu.tile_size, 640)
        self.assertTrue(gpu.enable_multiscale)

        jetson = SAHIConfig.jetson_nano()
        self.assertEqual(jetson.tile_size, 416)


class TestSAHIAppEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import web_app
        cls.app = web_app.app.test_client()

    def test_sahi_status_endpoint(self):
        res = self.app.get('/api/sahi/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertIn("available", data)
        self.assertIn("enabled", data)

    def test_sahi_toggle_endpoint(self):
        # Toggle on
        res = self.app.post('/api/sahi/toggle', json={"enabled": True, "profile": "default"})
        if res.status_code == 200:
            data = res.get_json()
            self.assertTrue(data["success"])
            self.assertTrue(data["enabled"])

            # Toggle off
            res2 = self.app.post('/api/sahi/toggle', json={"enabled": False})
            self.assertEqual(res2.status_code, 200)
            data2 = res2.get_json()
            self.assertFalse(data2["enabled"])


if __name__ == '__main__':
    unittest.main()
