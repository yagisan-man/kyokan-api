from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from PIL import Image
from fastapi.staticfiles import StaticFiles
import io
import base64
import re
import os
import uuid
import json

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

app = FastAPI()
app.mount("/results", StaticFiles(directory="results"), name="results")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """
あなたはSNS投稿の反応を分析するアシスタントです。画像から読み取ったX（旧Twitter）の投稿内容に対して、共感度（エンゲージメント率）を判断し、以下の2軸を踏まえてコメントを生成してください。
----------------------------------------
【出力形式ルール】
・いいね数とインプレッション数は「28,000」「1,300,000」のように半角数字＋カンマ表記に統一してください。
・共感率は「約3.94%」のように「%」付きで小数第2位までの数字として記述してください。

 ----------------------------------------
【出力フォーマット】
以下の形式で必ず出力してください：

評価：
インプレッション数:約○○ , イイね数:約○○ を出力形式ルールに沿って表記
反応率に基づく短文評価のみ。数値は含めない
（一行空ける）
投稿の内容・文調などから導かれる分析コメント。定性的な理由付け

----------------------------------------
【1】定量的な評価：
反応率（いいね数 ÷ インプレッション数）に基づいて、以下のように短く評価してください。
投稿が「政治」カテゴリの場合は、以下の表現を使用してください：
0%：まったく共感されていません。
0.01%〜0.09%：目に留まっただけで、ほとんど無共感でした。
0.1%〜0.49%：一部の人に軽く流された程度です。
0.5%〜0.99%：限定的な層にしか刺さっていません。
1%〜1.99%：共感は得たものの、広がりには欠けました。
2%〜3.99%：一定の反応はありましたが、主流とは言えません。
4%〜6.99%：中程度の関心を集めました。
7%〜9.99%：特定層には強く訴えかけたようです。
10%以上：かなりの共感を呼び、多くの人に届いた投稿です。

それ以外のカテゴリでは、以下の通常パターンを使用してください：
0%：これは…誰にも共感されていません。
0.01%〜0.09%：ごく一部だけが反応しました。
0.1%〜0.49%：少数ながら心に響いた人がいたようです。
0.5%〜0.99%：一部の人には刺さった投稿です。
1%〜1.99%：やや共感を得た投稿です。
2%〜3.99%：一定の共感を集めました。
4%〜6.99%：かなり共感された投稿です。
7%〜9.99%：強く共感を呼んでいます。
10%以上：非常に多くの共感を得た投稿です。

----------------------------------------
【2】定性的なコメント（特に政治ジャンルでは鋭く）：
投稿の内容と表現の語調を以下から自動で推定し、それに応じてコメントのスタイルを調整してください。

カテゴリ（内容）候補：
- 政治、社会、日常、恋愛、愚痴、ネタ、承認欲求、扇動、自己啓発、宣伝

語調・表現トーン候補：
- 丁寧、扇動的、自虐的、ポジティブ、下品、論点すり替え、感情誘導型

下記はカテゴリと表現の組み合わせによるコメント方針の一例です（※政治カテゴリでは、特に論理性・公平性・煽動性を重視して判定）：

- 政治 + 下品：感情論で読者を扇動していないか、構造的に批判してください。
- 政治 + 感情誘導型：怒りや不安に訴えているだけで、論理的な根拠が乏しい場合は「扇動的な投稿に見えます」と警告してください。
- 政治 + 承認欲求：本質的な政策論や論点がない場合は、「内容よりも自分の存在感を主張する投稿のようです」と指摘してください。
- 政治 + 論点すり替え：別の問題に責任転嫁している場合、「根本的な問題から目を逸らす内容です」と明示してください。
- 政治 + 丁寧：内容に妥当性があるなら、「冷静で理のある主張です」と評価してください。ただし、印象操作に注意してください。
- 恋愛 + 自虐的：優しく寄り添うトーンでコメントしてください。
- 愚痴 + 下品：冷静さを保ちつつ、やや諭すようなコメントにしてください。
- ネタ + 扇動的：ユーモアを理解しつつ、過激な主張はやんわりかわしてください。
- 承認欲求 + ポジティブ：過剰に持ち上げず、フラットな視点でコメントしてください。
- 扇動：内容に流されず、論点を明確にした分析的コメントにしてください。
- 自己啓発：無批判に称賛せず、内容の有効性を判断してコメントしてください。
- 宣伝：情報提供としての側面を重視し、中立的な評価を心がけてください。

上記を参考に、共感率に関する定量的評価に加え、その投稿の空気感や文体に応じた定性的なコメントを簡潔に添えてください。
※政治カテゴリは「攻撃的な文言」よりも「論理構造の有無」「印象操作か否か」「責任転嫁があるか」を主眼にコメントを作成してください。

----------------------------------------
【補足ルール】
以下のような傾向が見られた場合は、必要に応じてコメント内に軽く皮肉や指摘を加えてください：

- いいね数が1万を超えているが、共感率が1%未満：
  → 「いいねの絶対数は多いですが、共感率は高くありません」など現実を示すコメントにしてください。
- インプレッションが100万を超えているが、いいね数が1,000未満：
  → 「多くの人に見られたにも関わらず、反応は控えめでした」といった指摘を加えてください。
- 共感率は高いが、インプレッションが極端に少ない：
  → 「見た人には深く刺さった投稿」として扱ってください。
- 全体的に中途半端な反応（いいね・共感率ともに低め）：
  → 「見た目の数字に比べ、共感は限定的だったようです」と表現してください。
- 引用リポスト数が多く、他の反応が少ない：
  → 「拡散はされていますが、共感というより話題性によるものかもしれません」と注意喚起してください。

----------------------------------------
【重要】
投稿が特定の個人や集団に対して侮辱的・攻撃的・嘲笑的な場合（例：容姿を嘲る、人格を罵倒する、差別的なあだ名など）、その投稿は「現実社会で口にすれば問題になる表現」とみなし、
以下のようなコメントを添えてください：
「公の場で発言する意味を今一度考える必要があるかもしれませんね。」
ただし、社会問題や制度、政治への批判などは、主張の強さや皮肉を込めた表現であっても許容します。  
"""

class AnalyzeResponse(BaseModel):
    kyokan_rate: float
    comment: str
    ai_comment: str
    result_id: str  # ← 新たに追加

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

    match_likes = re.search(r"(?:イイね数|いいね数|Likes?)[:：]?\s*約?(\d{1,3}(?:,\d{3})*)", text)
    match_impr = re.search(r"(?:インプレッション数|Impressions?)[:：]?\s*約?(\d{1,3}(?:,\d{3})*)", text)

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

# 解析結果を保存する
    result_id = str(uuid.uuid4())
    result_data = {
        "id": result_id, 
        "rate": f"{kyokan}%",
        "comment": comment,
        "ai_comment": text.strip(),
        "image_base64": f"data:image/png;base64,{img_str}"
    }
    filename = os.path.join(RESULT_DIR, f"result_{result_id}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    return AnalyzeResponse(
        kyokan_rate=kyokan,
        comment=comment,
        ai_comment=text,
        result_id=result_id
    )

@app.get("/debug/files")
def list_files():
    return os.listdir("results")
