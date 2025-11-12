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
    return {"message": "Smart Signature Box Detection API ✅"}

@app.post("/signature")
async def extract_signature(pdf: UploadFile = File(...)):
    try:
        # Step 1: Save PDF temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf.read())
            pdf_path = tmp.name

        # Step 2: Convert first page to image
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img_data = pix.tobytes("png")
        pdf_doc.close()

        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"success": False, "message": "Image conversion failed"})

        H, W, _ = img.shape

        # Step 3: Detect face
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        if len(faces) == 0:
            user_y, user_h = int(H * 0.25), int(H * 0.25)
            user_x, user_w = int(W * 0.08), int(W * 0.25)
        else:
            (user_x, user_y, user_w, user_h) = faces[0]

        # Step 4: Region below user photo to search for signature box
        sig_y1 = int(user_y + user_h + (H * 0.05))
        sig_y2 = int(sig_y1 + user_h * 2.5)
        sig_y2 = min(sig_y2, H)
        search_area = img[sig_y1:sig_y2, :]

        gray_crop = cv2.cvtColor(search_area, cv2.COLOR_BGR2GRAY)

        # --- enhance contrast and edges ---
        gray_eq = cv2.equalizeHist(gray_crop)
        blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
        edges = cv2.Canny(blurred, 40, 120)
        dilated = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        # Step 5: Find rectangular contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidate_boxes = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / h if h > 0 else 0
            area = w * h
            if 2.0 < aspect < 9 and 15000 < area < 300000:
                candidate_boxes.append((x, y, w, h))

        if len(candidate_boxes) == 0:
            # fallback: crop directly below user photo
            sig_y = int(user_y + user_h + (H * 0.07))
            sig_h = int(user_h * 1.2)
            sig_x = int(user_x - (user_w * 0.2))
            sig_w = int(user_w * 1.8)
            sig_y = min(sig_y, H - sig_h)
            sig_h = min(sig_h, H - sig_y)
            sig_x = max(0, sig_x)
            sig_w = min(sig_w, W - sig_x)
            signature_crop = img[sig_y:sig_y + sig_h, sig_x:sig_x + sig_w]
            msg = "Fallback used (box not detected)"
        else:
            # choose the largest candidate box
            x, y, w, h = max(candidate_boxes, key=lambda b: b[2] * b[3])
            pad = 10
            y1 = max(y + pad, 0)
            y2 = min(y + h - pad, search_area.shape[0])
            x1 = max(x + pad, 0)
            x2 = min(x + w - pad, search_area.shape[1])
            signature_crop = search_area[y1:y2, x1:x2]
            msg = "Signature box detected successfully"

        # Step 6: Convert to Base64
        sig_pil = Image.fromarray(cv2.cvtColor(signature_crop, cv2.COLOR_BGR2RGB))
        buf = tempfile.SpooledTemporaryFile()
        sig_pil.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        b64_sig = base64.b64encode(buf.read()).decode("utf-8")

        return JSONResponse({
            "success": True,
            "message": msg,
            "signatureIMG": f"data:image/jpeg;base64,{b64_sig}"
        })

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})
