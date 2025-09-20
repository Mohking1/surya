from typing import List

from PIL import Image, ImageDraw

from surya.common.surya.schema import TaskNames
from surya.recognition import OCRResult


def test_latex_ocr(recognition_predictor, test_image_latex):
    width, height = test_image_latex.size
    results: List[OCRResult] = recognition_predictor(
        [test_image_latex], [TaskNames.block_without_boxes], bboxes=[[[0, 0, width, height]]]
    )
    text = results[0].text_lines[0].text
    assert len(results) == 1

    assert text.startswith("<math")
    assert text.endswith("</math>")
