"""把网关 403 拆成最小变量来定位是哪一层被拦。

跑法：
    python lessons/lesson6/lesson6_smoke_test.py

期望：四个测试至少能跑通 1~2 个，从对比里就能看出 WAF 卡的是什么。
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import httpx
from langchain_openai import ChatOpenAI
from openai import APIStatusError, OpenAI

import config

API_KEY = config.OPENAI_API_KEY
BASE_URL = config.OPENAI_BASE_URL
MODEL = "gpt-oss-120b"


def case(label: str):
    print("\n" + "=" * 60)
    print(f"CASE: {label}")
    print("=" * 60)


def show_status(ok: bool, info: str):
    flag = "[OK]" if ok else "[FAIL]"
    print(f"{flag} {info}")


# ---------- 1. 裸 httpx 短消息（等价于 tester.py 已通过的那个） ----------
def test_httpx_short():
    case("1. 裸 httpx + 'hi' + max_tokens=5（已知能通的基线）")
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 5,
        "temperature": 0,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30
        )
        show_status(resp.status_code == 200, f"HTTP {resp.status_code}  body={resp.text[:200]}")
    except Exception as e:
        show_status(False, repr(e))


# ---------- 2. 裸 httpx 长 prompt（拷贝 RAG 的实际内容） ----------
def test_httpx_long():
    case("2. 裸 httpx + RAG 风格的长中文 prompt")
    long_prompt = (
        "基于以下上下文回答问题。如果上下文里没有相关信息，请直接说\"无法回答\"，不要编造。\n\n"
        "上下文：\n"
        "[来源1 | chunk #58]\n"
        "### 错误 1：`git add .` 后直接提交 问题： - 容易把无关文件一起带上 - 可能误提交配置、本地缓存、调试文件\n\n"
        "问题：为什么不推荐使用 git add . 来提交代码？\n\n回答："
    )
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": long_prompt}],
        "max_tokens": 200,
        "temperature": 0.3,
        "stream": False,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    try:
        resp = httpx.post(
            f"{BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60
        )
        show_status(resp.status_code == 200, f"HTTP {resp.status_code}  body={resp.text[:200]}")
    except Exception as e:
        show_status(False, repr(e))


# ---------- 3. OpenAI SDK 短消息 ----------
def test_openai_sdk_short():
    case("3. OpenAI SDK + 'hi'")
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            temperature=0,
        )
        show_status(True, f"reply={resp.choices[0].message.content!r}")
    except APIStatusError as e:
        show_status(False, f"HTTP {e.status_code}  body={e.response.text[:200]}")
    except Exception as e:
        show_status(False, repr(e))


# ---------- 4. LangChain ChatOpenAI 短消息 ----------
def test_langchain_short():
    case("4. LangChain ChatOpenAI + 'hi'")
    llm = ChatOpenAI(
        model=MODEL, api_key=API_KEY, base_url=BASE_URL, temperature=0, max_tokens=5
    )
    try:
        resp = llm.invoke("hi")
        show_status(True, f"reply={resp.content!r}")
    except APIStatusError as e:
        show_status(False, f"HTTP {e.status_code}  body={e.response.text[:200]}")
    except Exception as e:
        show_status(False, repr(e))


# ---------- 5. OpenAI SDK + 自定义 User-Agent（验证是不是 UA 被拦） ----------
def test_openai_sdk_custom_ua():
    case("5. OpenAI SDK + 浏览器 User-Agent + 清空 x-stainless-*")
    # 把 OpenAI SDK 默认带的特征 header 全部覆盖成无害值
    safe_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "x-stainless-arch": "",
        "x-stainless-lang": "",
        "x-stainless-os": "",
        "x-stainless-package-version": "",
        "x-stainless-runtime": "",
        "x-stainless-runtime-version": "",
    }
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, default_headers=safe_headers)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            temperature=0,
        )
        show_status(True, f"reply={resp.choices[0].message.content!r}")
    except APIStatusError as e:
        show_status(False, f"HTTP {e.status_code}  body={e.response.text[:200]}")
    except Exception as e:
        show_status(False, repr(e))


if __name__ == "__main__":
    print(f"BASE_URL = {BASE_URL}")
    print(f"MODEL    = {MODEL}")
    print(f"KEY      = {API_KEY[:8]}...{API_KEY[-4:]}  (len={len(API_KEY)})")
    test_httpx_short()
    test_httpx_long()
    test_openai_sdk_short()
    test_langchain_short()
    test_openai_sdk_custom_ua()
