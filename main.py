from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
import io
import base64
import re
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
あなたはSNS投稿の反応を分析するアシスタントです。画像から読み取ったX（旧Twitter）の投稿内容に対して、共感度（エンゲージメント率）を判断し、定量的な評価（反応率に応じたコメント）と、定性的な分析（なぜ共感された／されなかったのか）を一言で述べてください。

定量的なコメントは以下に基づいて返答してください：
0%：これは…誰にも共感されていません。
0.01%〜0.09%：ごく一部だけが反応しました。
0.1%〜0.49%：少数ながら心に響いた人がいたようです。
0.5%〜0.99%：一部の人には刺さった投稿です。
1%〜1.99%：やや共感を得た投稿です。
2%〜3.99%：一定の共感を集めました。
4%〜6.99%：かなり共感された投稿です。
7%〜9.99%：強く共感を呼んでいます。
10%以上：非常に多くの共感を得た優れた投稿です。
"""

class AnalyzeResponse(BaseModel):
    kyokan_rate: float
    comment: str
    ai_comment: str

def parse_number(s):
    s = s.replace(',', '').strip()
    if '万' in s:
        return float(s.replace('万', '')) * 10000
    if '千' in s:
        return float(s.replace('千', '')) * 1000
    return float(s)

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    contents = await file.read()

    # サイズ制限（2MB）
    MAX_SIZE = 2 * 1024 * 1024
    if len(contents) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail="画像サイズが大きすぎます（最大2MBまで対応しています）"
        )

    # 画像処理
    image = Image.open(io.BytesIO(contents))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_str}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "この画像に含まれるSNS投稿の反応数（インプレッション・いいね・リポストなど）を読み取り、いいね数とインプレッション数から共感率を計算してください。また、反応率の評価と簡潔なコメントを含めてください。"
                    }
                ]
            }
        ]
    )

    text = response.choices[0].message.content
    print("----- GPT出力 -----")
    print(text)
    print("------------------")

    match_likes = re.search(r"(?:いいね数?|Likes?)[:：]?\s*(\d+(?:\.\d+)?[万千]?)", text)
    match_impr = re.search(r"(?:インプレッション数?|Impressions?)[:：]?\s*(\d+(?:\.\d+)?[万千]?)", text)

    likes = parse_number(match_likes.group(1)) if match_likes else 0
    impressions = parse_number(match_impr.group(1)) if match_impr else 0

    if impressions == 0:
        kyokan = 0.0
    else:
        kyokan = round((likes / impressions) * 100, 2)

    if kyokan == 0:
        comment = "これは…誰にも共感されていません。"
    elif kyokan < 0.1:
        comment = "ごく一部だけが反応しました。"
    elif kyokan < 0.5:
        comment = "少数ながら心に響いた人がいたようです。"
    elif kyokan < 1:
        comment = "一部の人には刺さった投稿です。"
    elif kyokan < 2:
        comment = "やや共感を得た投稿です。"
    elif kyokan < 4:
        comment = "一定の共感を集めました。"
    elif kyokan < 7:
        comment = "かなり共感された投稿です。"
    elif kyokan < 10:
        comment = "強く共感を呼んでいます。"
    else:
        comment = "非常に多くの共感を得た優れた投稿です。"

    return AnalyzeResponse(
        kyokan_rate=kyokan,
        comment=comment,
        ai_comment=text
    )
