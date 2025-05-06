
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import openai
import base64
from io import BytesIO
import os
import re

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

@app.get("/")
def read_root():
    return {"message": "API is running"}

def resize_and_encode(file):
    image = Image.open(file.file)
    if image.width > 1024:
        ratio = 1024 / image.width
        image = image.resize((1024, int(image.height * ratio)))
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

def get_comment_by_rate(rate):
    if rate >= 10:
        return "この投稿は極めて高い共感を得ています。内容が多くの人に深く届いた結果といえるでしょう。"
    elif rate >= 5:
        return "相当共感されています。ここまで響く投稿はそう多くありません。"
    elif rate >= 3:
        return "多くの人に“わかる”と感じさせた投稿です。"
    elif rate >= 2:
        return "一部には刺さっていますが、大多数には届いていません。"
    elif rate >= 1:
        return "目には留まったけど、心には届いてないようです。"
    elif rate >= 0.5:
        return "“無難”以上、“共鳴”未満。よくある投稿です。"
    elif rate >= 0.2:
        return "多くの人がスルーしています。内容の再考を。"
    elif rate >= 0.1:
        return "響かない投稿。見られただけで、何も残していません。"
    else:
        return "これは…誰にも共感されていません。"

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_data = resize_and_encode(file)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "以下の画像に写っているSNS投稿について、次の2点を回答してください：\n\n1. 投稿に記載されている「いいね数」と「インプレッション数」を教えてください。\n2. 投稿内容について、なぜ共感された（またはされなかった）と思うか、言葉選びや雰囲気、タイミングを踏まえて簡単なコメントを出してください。"},
                {"type": "image_url", "image_url": {"url": image_data}},
            ],
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4-vision-preview",
        messages=messages,
        max_tokens=300,
    )

    text = response.choices[0].message["content"]

    # 数値を抽出（簡易的に正規表現を使う）
    likes = 0
    impressions = 0
    match_likes = re.search(r"(?:いいね|Likes?)[:：]?\s*(\d[\d,]*)", text, re.IGNORECASE)
    match_impr = re.search(r"(?:インプレッション|Impressions?)[:：]?\s*(\d[\d,]*)", text, re.IGNORECASE)
    if match_likes:
        likes = int(match_likes.group(1).replace(",", ""))
    if match_impr:
        impressions = int(match_impr.group(1).replace(",", ""))

    kyokan_rate = round((likes / impressions) * 100, 2) if impressions > 0 else 0.0
    comment = get_comment_by_rate(kyokan_rate)

    # 内容に対するコメントを抽出（2点目の回答）
    ai_comment_match = re.search(r"2[\）\.]\s*(.+)", text, re.DOTALL)
    ai_comment = ai_comment_match.group(1).strip() if ai_comment_match else ""

    return {
        "likes": likes,
        "impressions": impressions,
        "kyokan_rate": kyokan_rate,
        "comment": comment,
        "ai_comment": ai_comment,
        "raw_text": text,
    }
