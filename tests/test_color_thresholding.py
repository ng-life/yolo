import unittest

import numpy as np

from color_thresholding import crop_rotated_rect


class CropRotatedRectTests(unittest.TestCase):
    def test_crop_rotated_rect_returns_portrait_when_object_is_tall(self) -> None:
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        rect = ((60.0, 60.0), (20.0, 60.0), 0.0)

        cropped = crop_rotated_rect(image, rect)

        self.assertIsNotNone(cropped)
        assert cropped is not None
        self.assertGreater(cropped.shape[0], cropped.shape[1])

    def test_crop_rotated_rect_rotates_result_180_degrees(self) -> None:
        image = np.zeros((120, 120, 3), dtype=np.uint8)
        image[30:60, 50:70] = (0, 0, 255)
        image[60:90, 50:70] = (255, 0, 0)
        rect = ((60.0, 60.0), (20.0, 60.0), 0.0)

        cropped = crop_rotated_rect(image, rect)

        self.assertIsNotNone(cropped)
        assert cropped is not None
        top_pixel = cropped[5, cropped.shape[1] // 2]
        bottom_pixel = cropped[-6, cropped.shape[1] // 2]
        self.assertTrue(np.array_equal(top_pixel, np.array([255, 0, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(bottom_pixel, np.array([0, 0, 255], dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()