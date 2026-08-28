#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""调用推理服务的最小示例（Python）。

密钥通过环境变量传入，不要把密钥写进代码：
    set TONGUE_API_KEY=webapp-001          :: Windows cmd
    set TONGUE_API_SECRET=sk-xxxx
    $env:TONGUE_API_KEY="webapp-001"       # PowerShell
    export TONGUE_API_KEY=webapp-001       # Linux/macOS

用法：
    python example_client.py 舌象图片.jpg
"""
import base64
import json
import os
import sys
import urllib.request

from auth_utils import make_auth_headers

BASE_URL = os.getenv("TONGUE_API_BASE", "http://localhost:8000")
API_KEY = os.getenv("TONGUE_API_KEY", "")
SECRET = os.getenv("TONGUE_API_SECRET", "")


def predict(image_path: str, color_level: int = 2):
    body = json.dumps({
        "image_base64": base64.b64encode(open(image_path, "rb").read()).decode("ascii"),
        "color_level": color_level,
    }).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if API_KEY and SECRET:
        headers.update(make_auth_headers(API_KEY, SECRET, "POST", "/predict_base64", body))
    req = urllib.request.Request(f"{BASE_URL}/predict_base64", data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    result = predict(sys.argv[1])
    for det in result.get("detections", []):
        print(f"{det['name']}  置信度 {det['confidence']:.2%}  框 {det['box']}")
    print("主体质:", result.get("constitution", {}).get("primary"))
