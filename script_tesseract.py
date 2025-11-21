
# ocr_numbers.py (updated with ignore_bottom_pct)
import cv2
import pytesseract
import numpy as np
import pandas as pd
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

@dataclass
class NumberBox:
    text: str
    left: int
    top: int
    width: int
    height: int
    conf: float

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (self.left + self.width // 2, self.top + self.height // 2)


def _preprocess(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    return thr


def _ocr_boxes(img: np.ndarray, allow_scientific: bool = True, psm: int = 6) -> List[NumberBox]:
    whitelist = "0123456789.-"
    if allow_scientific:
        whitelist += "eE+"
    config = f'--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}'
    df = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DATAFRAME)
    df = df.dropna(subset=["text"])
    mask = df["text"].str.fullmatch(r"[0-9][0-9eE+\-\.]*")
    df = df[mask].copy()

    boxes: List[NumberBox] = []
    for _, r in df.iterrows():
        try:
            conf = float(r["conf"])
        except Exception:
            conf = -1.0
        boxes.append(NumberBox(
            text=str(r["text"]),
            left=int(r["left"]),
            top=int(r["top"]),
            width=int(r["width"]),
            height=int(r["height"]),
            conf=conf,
        ))
    return boxes


def _merge_adjacent(boxes: List[NumberBox], x_gap: int = 6, y_align: int = 6) -> List[NumberBox]:
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda b: (b.top, b.left))
    merged: List[NumberBox] = []
    current = boxes[0]
    for b in boxes[1:]:
        same_line = abs(b.top - current.top) <= y_align
        touching = b.left - current.right <= x_gap
        if same_line and touching:
            new_left = min(current.left, b.left)
            new_top = min(current.top, b.top)
            new_right = max(current.right, b.right)
            new_bottom = max(current.bottom, b.bottom)
            current = NumberBox(
                text=current.text + b.text,
                left=new_left,
                top=new_top,
                width=new_right - new_left,
                height=new_bottom - new_top,
                conf=max(current.conf, b.conf),
            )
        else:
            merged.append(current)
            current = b
    merged.append(current)
    return merged


def _valid_number(s: str, allow_scientific: bool = True) -> bool:
    try:
        if allow_scientific:
            float(s.replace("E", "e"))
        else:
            import re
            return re.fullmatch(r"[+-]?\\d+(\\.\\d+)?", s) is not None
        return True
    except Exception:
        return False


def detect_numbers(
    image_path: str,
    allow_scientific: bool = True,
    visualize: bool = False,
    out_image_path: Optional[str] = None,
    ignore_bottom_pct: int = 0,
):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    H, W = img.shape[:2]
    cutoff_y = H * (1.0 - max(0, min(100, ignore_bottom_pct)) / 100.0)

    prep = _preprocess(img)
    raw = _ocr_boxes(prep, allow_scientific=allow_scientific)
    merged = _merge_adjacent(raw, x_gap=6, y_align=6)

    final_boxes = []
    for b in merged:
        if not _valid_number(b.text, allow_scientific=allow_scientific):
            continue
        cy = b.center[1]
        if cy >= cutoff_y:
            continue
        final_boxes.append(b)

    rows = []
    for b in final_boxes:
        cx, cy = b.center
        rows.append({
            "text": b.text,
            "left": b.left,
            "top": b.top,
            "width": b.width,
            "height": b.height,
            "center_x": cx,
            "center_y": cy,
            "conf": b.conf,
        })

    df = pd.DataFrame(rows).sort_values(by=["top", "left"]).reset_index(drop=True)

    if visualize:
        vis = img.copy()
        for _, r in df.iterrows():
            pt1 = (int(r["left"]), int(r["top"]))
            pt2 = (int(r["left"] + r["width"]), int(r["top"] + r["height"]))
            cv2.rectangle(vis, pt1, pt2, (0, 0, 255), 2)
            cv2.putText(vis, r["text"], (pt1[0], max(0, pt1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
        if out_image_path:
            cv2.imwrite(out_image_path, vis)

    return df




if __name__ == "__main__":
    detect_numbers("data/test_data.png",visualize=True,allow_scientific=True, out_image_path="./result/test_noignore.png")
