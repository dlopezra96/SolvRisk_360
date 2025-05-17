"""PDF utilities: draw headers and capture Dash map screenshots."""
import os
from datetime import datetime
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib.units import cm
import io
import time
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from typing import Any


def draw_header(c: Canvas, width: float, height: float,
                logo_img: Any, logo_w: float, logo_h: float,
                top_margin: float) -> float:
    """PDF Header.

    Draw the corporate header (timestamp and logo) at the top of
    the current PDF page.

    Parameters:
    - c (Canvas): reportlab.pdfgen.canvas.Canvas instance
    - width (float): page width
    - height (float): page height
    - logo_img (Any): an ImageReader or compatible object for the logo
    - logo_w (float): desired logo width (in points)
    - logo_h (float): desired logo height (in points)
    - top_margin (float): top margin offset (in points)

    Returns:
    - baseline_y (float): the Y coordinate of the header baseline (for
      subsequent drawing)
    """
    # Calculate the baseline Y position
    baseline_y = height - top_margin - 0.3 * cm

    # Draw timestamp at top-left
    gen_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    c.setFont("Helvetica", 8)
    c.drawString(1 * cm, baseline_y, f"Generated: {gen_time}")

    # Draw logo at top-right, vertically centered on the baseline
    logo_y = baseline_y - (logo_h / 2)
    c.drawImage(
        logo_img,
        width - 1 * cm - logo_w,
        logo_y,
        width=logo_w,
        height=logo_h,
        preserveAspectRatio=True,
        mask="auto",
    )

    return float(baseline_y)


def capture_dash_map(url: str, css_selector: str,
                     width: int = 800,
                     height: int = 600,
                     wait: float = 2.0) -> bytes:
    """Capture and crop a Dash map element as PNG bytes.

    Uses a headless Chrome WebDriver to navigate to the given URL,
    waits for the page to load, takes a full-page screenshot, then
    crops to the bounding box of the element matching the CSS selector.

    Parameters:
    - url (str): The file:// or http URL to load in headless Chrome.
    - css_selector (str): CSS selector for the map container to crop.
    - width (int): Browser window width in pixels (default 800).
    - height (int): Browser window height in pixels (default 600).
    - wait (float): Seconds to wait after navigation before screenshot.

    Returns:
    - bytes: PNG image bytes of the cropped map area.
    """
    # 1) Launch Chrome in headless mode using webdriver-manager.
    opts = Options()
    opts.add_argument("--headless")
    opts.add_argument(f"--window-size={width},{height}")
    opts.add_experimental_option("excludeSwitches", ["enable-logging"])
    opts.add_argument("--log-level=3")
    service = ChromeService(ChromeDriverManager().install(),
                            log_path=os.devnull)
    driver = webdriver.Chrome(service=service, options=opts)

    try:
        # 2) Navigate and wait for it to load.
        driver.get(url)
        time.sleep(wait)

        # 3) Take full-page screenshot
        png = driver.get_screenshot_as_png()

        # 4) Locate the map div and get its position and size.
        try:
            el = driver.find_element("css selector", css_selector)
            loc = el.location
            size = el.size
        except NoSuchElementException:
            loc = {"x": 0, "y": 0}
            size = {"width": width, "height": height}

    finally:
        driver.quit()

    # 5) Crop the image to the map area.
    img = Image.open(io.BytesIO(png))
    box = (
        int(loc["x"]),
        int(loc["y"]),
        int(loc["x"] + size["width"]),
        int(loc["y"] + size["height"]),
    )
    crop = img.crop(box)

    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    buf.seek(0)
    return buf.read()
