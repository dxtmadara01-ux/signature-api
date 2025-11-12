import io, base64, traceback
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import fitz
import cv2

app = FastAPI()

def pdf_first_page_to_image(pdf_bytes: bytes, dpi: int = 200):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc.load_page(0)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img

def np_to_data_uri(arr, quality=90):
    if len(arr.shape) == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    else:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return "data:image/jpeg;base64," + b64

@app.post("/signature")
async def signature(pdf: UploadFile = File(...), dpi: int = 200, debug: bool = False):
    try:
        pdf_bytes = await pdf.read()
        img = pdf_first_page_to_image(pdf_bytes, dpi)
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        H, W = arr.shape[:2]
        bottom = int(H * 0.65)
        roi = arr[bottom:H]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((5, 3), np.uint8)
        mor = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_score = -1

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            X, Y = x, y + bottom
            area = w * h
            ar = w / max(1, h)

            if w < 120 or h < 20 or ar < 2.0 or area < 1500:
                continue

            score = ar + (Y / H) + (area / (W * H))
            if score > best_score:
                best_score = score
                best = (X, Y, w, h)

        if not best:
            return {"success": False, "message": "Signature not detected"}

        x, y, w, h = best
        crop = arr[y:y+h, x:x+w]

        return {
            "success": True,
            "signatureIMG": np_to_data_uri(crop),
            "bbox": {"x": x, "y": y, "w": w, "h": h},
            "pageSize": {"width": W, "height": H}
        }

    except Exception as e:
        return {"success": False, "error": str(e), "trace": traceback.format_exc()}
