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
    return {"message": "Dynamic Signature API Running ✅"}

@app.post("/signature")
async def extract_signature(pdf: UploadFile = File(...)):
    try:
        # --- Step 1: Save PDF temporarily ---
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await pdf.read())
            pdf_path = tmp.name

        # --- Step 2: Convert first page to image ---
        pdf_doc = fitz.open(pdf_path)
        page = pdf_doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # High DPI
        img_data = pix.tobytes("png")
        pdf_doc.close()

        np_img = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        if img is None:
            return JSONResponse({"success": False, "message": "Image conversion failed"})

        H, W, _ = img.shape

        # --- Step 3: Detect face (user photo) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))

        if len(faces) == 0:
            # fallback: use typical left-top region for photo
            user_y, user_h = int(H * 0.25), int(H * 0.25)
            user_x, user_w = int(W * 0.08), int(W * 0.25)
        else:
            (user_x, user_y, user_w, user_h) = faces[0]  # Take first detected face

        # --- Step 4: Define signature region BELOW the user photo ---
        sig_y = int(user_y + user_h + (H * 0.05))
        sig_h = int(user_h * 0.7)
        sig_x = int(user_x)
        sig_w = int(user_w * 1.5)

        # Safety boundary
        sig_y = min(sig_y, H - sig_h)
        sig_h = min(sig_h, H - sig_y)
        sig_x = min(sig_x, W - sig_w)
        sig_w = min(sig_w, W - sig_x)

        signature_crop = img[sig_y:sig_y + sig_h, sig_x:sig_x + sig_w]

        # --- Step 5: Convert to Base64 image ---
        sig_pil = Image.fromarray(cv2.cvtColor(signature_crop, cv2.COLOR_BGR2RGB))
        buf = tempfile.SpooledTemporaryFile()
        sig_pil.save(buf, format="JPEG", quality=90)
        buf.seek(0)
        b64_sig = base64.b64encode(buf.read()).decode("utf-8")

        return JSONResponse({
            "success": True,
            "message": "Signature extracted successfully",
            "signatureIMG": f"data:image/jpeg;base64,{b64_sig}",
            "debug": {
                "user_face_box": [int(user_x), int(user_y), int(user_w), int(user_h)],
                "sig_crop_box": [int(sig_x), int(sig_y), int(sig_w), int(sig_h)],
                "page_size": [W, H]
            }
        })

    except Exception as e:
        return JSONResponse({"success": False, "message": str(e)})
