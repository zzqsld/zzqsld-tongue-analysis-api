#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""API 密钥 + HMAC-SHA256 签名鉴权工具（服务端与客户端共用参考实现）。

签名规则（类阿里云/AWS 风格）：
    string_to_sign = METHOD\nPATH\nTIMESTAMP\nNONCE\nBODY_SHA256_HEX
    signature      = HMAC_SHA256(secret, string_to_sign) 的 hex

请求头：
    X-Api-Key    API 密钥 ID
    X-Timestamp  Unix 秒级时间戳
    X-Nonce      随机字符串（防重放）
    X-Signature  上一步计算出的 hex 签名
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import time
from typing import Tuple

# 时间戳允许的最大偏移（秒），防止旧请求被重放
DEFAULT_TIMESTAMP_WINDOW = 300


def body_sha256_hex(body: bytes) -> str:
    return hashlib.sha256(body or b"").hexdigest()


def build_string_to_sign(method: str, path: str, timestamp: str, nonce: str, body: bytes) -> str:
    return "\n".join([method.upper(), path, timestamp, nonce, body_sha256_hex(body)])


def sign_request(secret: str, method: str, path: str, timestamp: str, nonce: str, body: bytes) -> str:
    """客户端/服务端共用的签名计算。"""
    string_to_sign = build_string_to_sign(method, path, timestamp, nonce, body)
    return hmac.new(secret.encode("utf-8"), string_to_sign.encode("utf-8"), hashlib.sha256).hexdigest()


def make_auth_headers(api_key: str, secret: str, method: str, path: str, body: bytes = b"") -> dict:
    """客户端调用：生成完整鉴权请求头。"""
    timestamp = str(int(time.time()))
    nonce = secrets.token_hex(8)
    signature = sign_request(secret, method, path, timestamp, nonce, body)
    return {
        "X-Api-Key": api_key,
        "X-Timestamp": timestamp,
        "X-Nonce": nonce,
        "X-Signature": signature,
    }


def verify_signature(
    api_key: str,
    secret: str,
    method: str,
    path: str,
    timestamp: str,
    nonce: str,
    provided_signature: str,
    body: bytes,
    window: int = DEFAULT_TIMESTAMP_WINDOW,
) -> Tuple[bool, str]:
    """服务端校验：时间戳窗口 + 签名比对。返回 (是否通过, 失败原因)。"""
    try:
        ts = int(timestamp)
    except (TypeError, ValueError):
        return False, "invalid timestamp"
    now = int(time.time())
    if abs(now - ts) > window:
        return False, "timestamp expired"

    expected = sign_request(secret, method, path, timestamp, nonce, body)
    if not hmac.compare_digest(expected, provided_signature or ""):
        return False, "signature mismatch"
    return True, ""


class NonceCache:
    """简单内存防重放缓存：记录最近见过的 nonce。"""

    def __init__(self, ttl: int = DEFAULT_TIMESTAMP_WINDOW, max_size: int = 10000) -> None:
        self.ttl = ttl
        self.max_size = max_size
        self._store: dict[str, int] = {}

    def seen(self, nonce: str) -> bool:
        now = int(time.time())
        # 惰性清理过期项
        if len(self._store) > self.max_size:
            expired = [k for k, v in self._store.items() if v < now]
            for k in expired:
                self._store.pop(k, None)
        if nonce in self._store and self._store[nonce] >= now:
            return True
        self._store[nonce] = now + self.ttl
        return False


class RateLimiter:
    """简单令牌桶限流：按 API Key 统计每分钟调用次数。"""

    def __init__(self, per_minute: int = 60) -> None:
        self.per_minute = per_minute
        self._counters: dict[str, tuple[int, int]] = {}  # key -> (window_start, count)

    def allow(self, key: str) -> bool:
        if self.per_minute <= 0:
            return True
        now = int(time.time())
        window = now // 60
        start, count = self._counters.get(key, (window, 0))
        if start != window:
            start, count = window, 0
        if count >= self.per_minute:
            self._counters[key] = (start, count)
            return False
        self._counters[key] = (start, count + 1)
        return True
