from PIL import Image

from mat_agentflow.image_executor import execute_opencv


def test_real_opencv_execution_changes_image():
    image = Image.new("RGB", (12, 8), "white")
    code = """<code>```python
import cv2
img = cv2.imread('path_to_input_image.jpg')
img[:] = 0
cv2.imwrite('path_to_output_image.jpg', img)
```</code>"""
    result = execute_opencv(code, image)
    assert result.success and result.changed
    assert result.image.size == image.size


def test_forbidden_code_rejected():
    result = execute_opencv("<code>import subprocess</code>", Image.new("RGB", (4, 4)))
    assert not result.success
