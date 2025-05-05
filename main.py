from fastapi import FastAPI, UploadFile, File
from PIL import Image
import openai
import base64
from io import BytesIO
import os

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")

def resize_and_encode(file):
    image = Image.open(file.file)
    if image.width > 1024:
        ratio = 1024 / image.width
        image = image.resize((1024, int(image.height * ratio)))
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_data = resize_and_encode(file)
    prompt = "この画像に含まれる数字から、インプレッション数、いいね、リポスト、引用、ブックマーク数を特定して、それらの合計をインプレッション数で割って共感率を出して下さい。"
    response = openai.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]}
        ],
        max_tokens=1000
    )
    return { "result": response.choices[0].message.content }
