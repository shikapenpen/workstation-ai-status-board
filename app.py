import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Any

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field, model_validator
from sentence_transformers import SentenceTransformer


# =========================================================
# Configuration
# =========================================================
#
# Gatewayの責務:
#   - model routing
#   - queue / concurrency control
#   - health
#   - timing
#   - request/response passthrough
#
# Gatewayはモデル固有のgeneration policyを持たない。
# max_tokens / temperature 等は、クライアントが明示した場合だけ転送する。
# 未指定時は llama-server 側の設定を採用する。
# =========================================================

MODELS = {
    "qwen35": {
        "url": "http://127.0.0.1:8002/v1/chat/completions",
        "limit": 1,
        "timeout": 600,
    },
    "qwenvl": {
        "url": "http://127.0.0.1:8003/v1/chat/completions",
        "limit": 1,
        "timeout": 240,
    },
    "gemma4": {
        "url": "http://127.0.0.1:8011/v1/chat/completions",
        "limit": 1,
        "timeout": 600,
    },
    "absa": {
        "url": "http://127.0.0.1:8012/v1/chat/completions",
        "limit": 1,
        "timeout": 180,
    },
    "phi4": {
        "url": "http://127.0.0.1:8013/v1/chat/completions",
        "limit": 1,
        "timeout": 240,
    },
}

RURI_MODEL = os.getenv("RURI_MODEL", "cl-nagoya/ruri-v3-310m")
EMBED_LIMIT = 1

MODEL_SEMAPHORES = {
    name: asyncio.Semaphore(cfg["limit"])
    for name, cfg in MODELS.items()
}
embed_semaphore = asyncio.Semaphore(EMBED_LIMIT)

request_state: dict[str, dict] = {}
request_lock = asyncio.Lock()


# =========================================================
# Application lifecycle
# =========================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.AsyncClient()
    app.state.ruri = SentenceTransformer(
        RURI_MODEL,
        device="cpu",
    )

    yield

    await app.state.http.aclose()


app = FastAPI(
    title="Zemi LLM Gateway",
    version="3.0",
    lifespan=lifespan,
)


# =========================================================
# Request schemas
# =========================================================

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatRequest(BaseModel):
    # 後方互換: prompt または messages のどちらか一方
    prompt: str | None = None
    messages: list[Message] | None = None

    # NoneならGatewayはpayloadへ含めない。
    # 生成上限・sampling既定値はllama-server側に任せる。
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)

    # 推論制御
    reasoning: bool | None = None

    response_format: dict[str, Any] | None = None
    json_schema: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_input(self):
        has_prompt = bool(self.prompt and self.prompt.strip())
        has_messages = bool(self.messages)

        if has_prompt == has_messages:
            raise ValueError(
                "prompt または messages のどちらか一方を指定してください"
            )

        return self


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1)


# =========================================================
# Client identity
# =========================================================

def client_id(
    request: Request,
    x_user: str | None,
) -> str:
    if x_user and x_user.strip():
        return x_user.strip()

    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()

    if request.client:
        return request.client.host

    return "unknown"


# =========================================================
# Queue tracking
# =========================================================

async def register_job(
    model: str,
    user: str,
) -> str:
    job_id = str(uuid.uuid4())

    async with request_lock:
        request_state[job_id] = {
            "id": job_id,
            "model": model,
            "user": user,
            "state": "waiting",
            "created": time.monotonic(),
            "started": None,
        }

    return job_id


async def set_running(
    job_id: str,
) -> None:
    async with request_lock:
        job = request_state.get(job_id)

        if job is not None:
            job["state"] = "running"
            job["started"] = time.monotonic()


async def finish_job(
    job_id: str,
) -> None:
    async with request_lock:
        request_state.pop(job_id, None)


# =========================================================
# Message conversion
# =========================================================

def build_messages(
    req: ChatRequest,
) -> list[dict[str, str]]:
    if req.messages:
        return [
            message.model_dump()
            for message in req.messages
        ]

    return [
        {
            "role": "user",
            "content": req.prompt,
        }
    ]


# =========================================================
# Backend call
# =========================================================

async def call_model(
    model: str,
    req: ChatRequest,
    request: Request,
    x_user: str | None,
):
    cfg = MODELS.get(model)

    if cfg is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown model: {model}",
        )

    # Gatewayはモデル固有の既定値を注入しない。
    payload: dict = {
        "messages": build_messages(req),
    }

    if req.response_format is not None and req.json_schema is not None:
        raise HTTPException(
            status_code=400,
            detail="response_format and json_schema cannot be used together",
        )
    if req.response_format is not None:
        payload["response_format"] = req.response_format
    if req.json_schema is not None:
        payload["json_schema"] = req.json_schema

    if req.max_tokens is not None:
        payload["max_tokens"] = req.max_tokens

    if req.temperature is not None:
        payload["temperature"] = req.temperature

    if req.reasoning is False:
        payload["reasoning_effort"] = "none"

    job_id = await register_job(
        model,
        client_id(request, x_user),
    )

    queued_at = time.perf_counter()

    try:
        async with MODEL_SEMAPHORES[model]:
            acquired_at = time.perf_counter()
            queue_wait_sec = acquired_at - queued_at

            await set_running(job_id)

            inference_started = time.perf_counter()

            try:
                response = await app.state.http.post(
                    cfg["url"],
                    json=payload,
                    timeout=cfg["timeout"],
                )
                response.raise_for_status()

            except httpx.TimeoutException as exc:
                raise HTTPException(
                    status_code=504,
                    detail=f"{model} timeout",
                ) from exc

            except httpx.HTTPStatusError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        f"{model} backend returned "
                        f"HTTP {exc.response.status_code}: "
                        f"{exc.response.text[:1000]}"
                    ),
                ) from exc

            except httpx.HTTPError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"{model} backend error: {exc}",
                ) from exc

            inference_sec = time.perf_counter() - inference_started

            try:
                result = response.json()

            except ValueError as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"{model} backend returned invalid JSON",
                ) from exc

            # SDKの性能測定用。backend response自体は変更しない。
            result["server_timing"] = {
                "queue_wait_sec": round(queue_wait_sec, 6),
                "inference_sec": round(inference_sec, 6),
            }

            return result

    finally:
        await finish_job(job_id)


# =========================================================
# Health
# =========================================================

async def backend_health(
    name: str,
    cfg: dict,
) -> tuple[str, bool]:
    url = cfg["url"].replace(
        "/v1/chat/completions",
        "/health",
    )

    try:
        response = await app.state.http.get(
            url,
            timeout=1.5,
        )
        return name, response.status_code == 200

    except httpx.HTTPError:
        return name, False


@app.get("/health")
async def health():
    results = await asyncio.gather(
        *(
            backend_health(name, cfg)
            for name, cfg in MODELS.items()
        )
    )

    models = dict(results)
    models["embed"] = True

    return {
        "status": "ok",
        "service": "llm-gateway",
        "models": models,
    }


# =========================================================
# Queue
# =========================================================

@app.get("/queue/status")
async def queue_status():
    now = time.monotonic()

    async with request_lock:
        jobs = []

        for job in request_state.values():
            started = job["started"]

            if started is None:
                wait_sec = now - job["created"]
                run_sec = None
            else:
                wait_sec = started - job["created"]
                run_sec = now - started

            jobs.append({
                "id": job["id"],
                "model": job["model"],
                "user": job["user"],
                "state": job["state"],
                "wait_sec": round(wait_sec, 1),
                "run_sec": (
                    round(run_sec, 1)
                    if run_sec is not None
                    else None
                ),
            })

    return {
        "limits": {
            **{
                name: cfg["limit"]
                for name, cfg in MODELS.items()
            },
            "embed": EMBED_LIMIT,
        },
        "requests": jobs,
    }


# =========================================================
# Chat API
# =========================================================

@app.post("/chat/{model}")
async def chat(
    model: str,
    req: ChatRequest,
    request: Request,
    x_user: str | None = Header(default=None),
):
    return await call_model(
        model,
        req,
        request,
        x_user,
    )


# =========================================================
# Embedding API
# =========================================================

@app.post("/embed")
async def embed(
    req: EmbedRequest,
    request: Request,
    x_user: str | None = Header(default=None),
):
    if any(
        not text.strip()
        for text in req.texts
    ):
        raise HTTPException(
            status_code=400,
            detail="texts must not contain empty strings",
        )

    job_id = await register_job(
        "embed",
        client_id(request, x_user),
    )

    try:
        async with embed_semaphore:
            await set_running(job_id)

            try:
                vectors = await asyncio.to_thread(
                    app.state.ruri.encode,
                    req.texts,
                    normalize_embeddings=True,
                )

            except Exception as exc:
                raise HTTPException(
                    status_code=500,
                    detail=f"embedding failed: {exc}",
                ) from exc

            return {
                "embeddings": vectors.tolist(),
                "count": len(vectors),
                "dim": (
                    vectors.shape[1]
                    if len(vectors)
                    else 0
                ),
            }

    finally:
        await finish_job(job_id)
