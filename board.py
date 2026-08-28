"""ゼミ室 LLM 利用状況ボード。

FastAPI gateway の /queue/status と /health を表示する。

起動:
    streamlit run board.py \
        --server.address 0.0.0.0 \
        --server.port 8501

外観は同じディレクトリの style.css で管理する。
"""

from __future__ import annotations

import html
import ipaddress
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# =========================================================
# Configuration
# =========================================================

GATEWAY_URL = os.getenv(
    "GATEWAY_URL",
    "http://127.0.0.1:8080",
)

REFRESH_MS = 2000
GRACE_SEC = 6

# gateway の model 名と一致させる
MODELS = {
    "qwen35": (
        "Qwen 3.5 9B",
        "汎用・推論 (Alibaba)",
    ),
    "gemma4": (
        "Gemma 4 12B",
        "推論・MTP (Google)",
    ),
    "phi4": (
        "Phi-4 Reasoning 9B",
        "推論特化 (Microsoft)",
    ),
    "absa": (
        "Qwen 3 4B",
        "レビュー構造化 / ABSA (Alibaba)",
    ),
    "qwenvl": (
        "Qwen 3 8B VL",
        "画像・マルチモーダル (Alibaba)",
    ),
    "gemma3": (
        "Gemma 3 1B",
        "CPU・軽量 (Google)",
    ),
    "embed": (
        "Ruri V3 310M",
        "埋め込み / RAG (Nagoya University)",
    ),
}

# X-User / Tailscale端末名 / IP の表示名上書き
NAMES = {}


# =========================================================
# Streamlit
# =========================================================

st.set_page_config(
    page_title="ゼミ室ワークステーション 利用状況",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st_autorefresh(
    interval=REFRESH_MS,
    key="board_refresh",
)


# =========================================================
# CSS
# =========================================================

def load_css() -> str:
    candidates = [
        Path(__file__).with_name("style.css"),
        Path.home() / "seting" / "style.css",
    ]

    for path in candidates:
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            continue

    st.warning("style.css が見つかりません。")
    return ""


st.markdown(
    f"<style>{load_css()}</style>",
    unsafe_allow_html=True,
)


# =========================================================
# Identity
# =========================================================

def is_ip(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
        return True
    except ValueError:
        return False


@st.cache_data(ttl=300, show_spinner=False)
def tailscale_name(ip: str) -> str | None:
    """Tailscale IP から端末名を取得する。"""

    if not is_ip(ip):
        return None

    try:
        result = subprocess.run(
            ["tailscale", "whois", "--json", ip],
            capture_output=True,
            text=True,
            timeout=2,
        )

        if result.returncode != 0:
            return None

        data = json.loads(result.stdout)
        node = data.get("Node") or {}
        hostinfo = node.get("Hostinfo") or {}

        return (
            node.get("ComputedName")
            or hostinfo.get("Hostname")
            or (node.get("Name") or "").split(".")[0]
            or None
        )

    except (
        OSError,
        subprocess.TimeoutExpired,
        json.JSONDecodeError,
    ):
        return None


def who(req: dict) -> str:
    """gateway が返す user を表示名へ変換する。"""

    identity = str(
        req.get("user")
        or "unknown"
    )

    name = (
        NAMES.get(identity)
        or tailscale_name(identity)
        or identity
    )

    return html.escape(name)


# =========================================================
# Gateway
# =========================================================

def fetch_gateway() -> tuple[dict, dict]:
    """queue/status と health を1回ずつ取得する。"""

    try:
        status_response = requests.get(
            f"{GATEWAY_URL}/queue/status",
            timeout=3,
        )
        status_response.raise_for_status()

        health_response = requests.get(
            f"{GATEWAY_URL}/health",
            timeout=3,
        )
        health_response.raise_for_status()

        status = status_response.json()
        health = health_response.json()

        return (
            status,
            health.get("models", {}),
        )

    except requests.RequestException as exc:
        st.error(
            f"ゲートウェイに接続できません: {GATEWAY_URL}"
        )
        st.caption(str(exc))
        st.stop()

    except ValueError as exc:
        st.error(
            "ゲートウェイから不正なJSONが返されました。"
        )
        st.caption(str(exc))
        st.stop()


# =========================================================
# Queue processing
# =========================================================

def fmt_sec(value: float | int | None) -> str:
    if value is None:
        return ""

    sec = int(value)

    if sec < 60:
        return f"{sec}秒"

    return f"{sec // 60}分{sec % 60}秒"


def dedup_by_user(
    items: list[dict],
    key: str,
) -> list[dict]:
    """同じユーザーの複数JOBを1人として表示する。"""

    best: dict[str, dict] = {}

    for item in items:
        user = str(
            item.get("user")
            or "unknown"
        )

        current = item.get(key) or 0

        if (
            user not in best
            or current > (best[user].get(key) or 0)
        ):
            best[user] = item

    return list(best.values())


def apply_grace(
    requests_: list[dict],
) -> list[dict]:
    """終了直後のJOBを短時間残し、表示の点滅を抑える。"""

    now = time.time()

    recent: dict = st.session_state.setdefault(
        "recent_running",
        {},
    )

    live_ids = {
        req["id"]
        for req in requests_
    }

    for req in requests_:
        if req.get("state") == "running":
            recent[req["id"]] = {
                "info": req.copy(),
                "last_seen": now,
            }

    for job_id in list(recent):
        if (
            now - recent[job_id]["last_seen"]
            > GRACE_SEC
        ):
            del recent[job_id]

    effective = list(requests_)

    for job_id, entry in recent.items():
        if job_id not in live_ids:
            effective.append(entry["info"])

    return effective


# =========================================================
# Card rendering
# =========================================================

def render_model_card(
    model: str,
    label: str,
    subtitle: str,
    items: list[dict],
    limit,
    online: bool,
) -> None:

    running = dedup_by_user(
        [
            req
            for req in items
            if req.get("state") == "running"
        ],
        "run_sec",
    )

    waiting = dedup_by_user(
        [
            req
            for req in items
            if req.get("state") == "waiting"
        ],
        "wait_sec",
    )

    waiting.sort(
        key=lambda req: req.get("wait_sec") or 0,
        reverse=True,
    )

    if not online:
        css = "offline"
    elif running:
        css = "busy"
    elif waiting:
        css = "queue"
    else:
        css = ""

    parts = [
        f'<div class="model-card {css}">',
        (
            '<!-- <span class="limit-badge">'
            f"同時 {html.escape(str(limit))} まで"
            "</span>-->"
        ),
        (
            '<div class="model-title">'
            f"{html.escape(label)}"
            "</div>"
        ),
        (
            '<div class="model-sub">'
            f"{html.escape(subtitle)}"
            "</div>"
        ),
    ]

    # ONLINE/OFFLINE と実行状態を排他的に表示
    if not online:
        parts.append(
            '<div class="idle-line">'
            "🔴 停止中"
            "</div>"
        )

    elif running:
        for req in running:
            parts.append(
                '<div class="run-line pulse">'
                f"🟢 {who(req)} "
                '<span style="'
                "font-size:0.9rem;"
                'font-weight:400;">'
                f"（{fmt_sec(req.get('run_sec'))} 実行中）"
                "</span>"
                "</div>"
            )

    else:
        parts.append(
            '<div class="idle-line">'
            "🟢 空いています"
            "</div>"
        )

    # OFFLINEでは待ち表示を出さない
    if online and waiting:
        parts.append(
            '<div class="wait-line" '
            'style="margin-top:0.5rem;">'
            f"⏳ 待ち {len(waiting)} 人"
            "</div>"
        )

        for pos, req in enumerate(
            waiting,
            start=1,
        ):
            parts.append(
                '<div class="wait-line">'
                f"　{pos}. {who(req)} "
                '<span style="'
                "color:#999;"
                'font-size:0.85rem;">'
                f"（{fmt_sec(req.get('wait_sec'))} 待ち）"
                "</span>"
                "</div>"
            )

    parts.append("</div>")

    st.markdown(
        "".join(parts),
        unsafe_allow_html=True,
    )


# =========================================================
# Main
# =========================================================

st.title(
    "🖥️ ゼミ室ワークステーション 利用状況"
)

data, health = fetch_gateway()

requests_raw = data.get(
    "requests",
    [],
)

limits = data.get(
    "limits",
    {},
)

requests_effective = apply_grace(
    requests_raw
)


# =========================================================
# Summary
# =========================================================

running_users = {
    req.get("user")
    for req in requests_effective
    if req.get("state") == "running"
    and req.get("user")
}

waiting_users = {
    req.get("user")
    for req in requests_effective
    if req.get("state") == "waiting"
    and req.get("user")
}

st.caption(
    "最終更新: "
    f"{datetime.now().strftime('%H:%M:%S')}"
    f"（{REFRESH_MS // 1000}秒ごとに自動更新）"
)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        '<div class="section-label">'
        "実行中"
        "</div>",
        unsafe_allow_html=True,
    )

    color = (
        "#2B9E6B"
        if running_users
        else "#C8BBD2"
    )

    st.markdown(
        '<div class="big-stat" '
        f'style="color:{color}">'
        f"{len(running_users)}"
        "</div>",
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        '<div class="section-label">'
        "待っている人"
        "</div>",
        unsafe_allow_html=True,
    )

    color = (
        "#F2925B"
        if waiting_users
        else "#C8BBD2"
    )

    st.markdown(
        '<div class="big-stat" '
        f'style="color:{color}">'
        f"{len(waiting_users)}"
        "</div>",
        unsafe_allow_html=True,
    )


# =========================================================
# Model cards
# =========================================================

st.markdown("---")

by_model: dict[str, list[dict]] = {
    model: []
    for model in MODELS
}

for req in requests_effective:
    model = req.get("model")

    if model in by_model:
        by_model[model].append(req)


columns = st.columns(
    2,
    gap="large",
)

for index, (
    model,
    metadata,
) in enumerate(MODELS.items()):

    label, subtitle = metadata

    limit = limits.get(
        model,
        "?",
    )

    online = bool(
        health.get(model, False)
    )

    with columns[index % 2]:
        render_model_card(
            model=model,
            label=label,
            subtitle=subtitle,
            items=by_model[model],
            limit=limit,
            online=online,
        )


# =========================================================
# Footer
# =========================================================

st.markdown("---")

online_count = sum(
    bool(health.get(model, False))
    for model in MODELS
)

st.caption(
    f"稼働モデル: {online_count}/{len(MODELS)}　|　"
    f"Gateway: {GATEWAY_URL}　|　"
    "Shared Workstation"
)