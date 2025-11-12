from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import fitz  # PyMuPDF
import cv2
import numpy as np
import base64
import tempfile
from PIL import Image

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Signature API Running ✅"}

@app.post("/signature")
async def extract_signature(pdf: UploadFile = File(...)):
    try:
        # --- Step 1: Save uploaded PDF temporarily ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf.read())
            tmp_path = tmp.name

        # --- Step 2: Convert 1st page to image using PyMuPDF ---
        pdf_doc = fitz.open(tmp_path)
        page = pdf_doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom = better quality
        img_data = pix.tobytes("png")
        pdf_doc.close()

        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse({"success": False, "message": "PDF page rendering failed"})

        H, W, _ = img.shape

        # --- Step 3: Try smart detection (threshold + contours) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(blur, 180, 255, cv2.THRESH_BINARY_INV)[1]

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        signature_crop = None
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)
            area = w * h
            # Filter possible signature zones (wide + short + mid/lower area)
            if 2.5 < aspect_ratio < 10 and 5000 < area < 500000 and y > H * 0.55:
                signature_crop = img[y:y + h, x:x + w]
                break

        # --- Step 4: If smart detection failed → fallback fixed crop ---
        if signature_crop is None:
            # Crop bottom-right zone (relative to userIMG area)
            sig_y = int(H * 0.75)
            sig_h = int(H * 0.20)
            sig_x = int(W * 0.45)
            sig_w = int(W * 0.45)
            signature_crop = img[sig_y:sig_y + sig_h, sig_x:sig_x + sig_w]

        # --- Step 5: Convert cropped signature to Base64 ---
        sig_pil = Image.fromarray(cv2.cvtColor(signature_crop, cv2.COLOR_BGR2RGB))
        buf = tempfile.SpooledTemporaryFile()
        sig_pil.save(buf, format="JPEG", quality=85)
        buf.seek(0)
        b64_sig = base64.b64encode(buf.read()).decode("utf-8")

        return JSONResponse({
            "success": True,
            "message": "Signature extracted successfully",
            "signatureIMG": f"data:image/jpeg;base64,{b64_sig}"
        })

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})
