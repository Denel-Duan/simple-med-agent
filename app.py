import os
import io
import json
import random
import re
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 环境变量 / Secrets
# =========================
load_dotenv()

def get_secret(name: str, default: str = "") -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

API_KEY = get_secret("API_KEY", "")
API_BASE = get_secret("API_BASE", "")
MODEL = get_secret("MODEL", "")

if not API_KEY or not API_BASE or not MODEL:
    raise RuntimeError("请检查配置：缺少 API_KEY / API_BASE / MODEL。可在 .env 或 Streamlit secrets 中设置。")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE,
)

# =========================
# 页面配置
# =========================
st.set_page_config(
    page_title="医学智能体平台",
    page_icon="🧪",
    layout="wide",
)

# =========================
# 全局样式
# =========================
st.markdown("""
<style>
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 1.2rem;
    max-width: 1500px;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7f9fc 0%, #eef3ff 100%);
    border-right: 1px solid #e6ebf5;
}

.panel-card {
    background: #ffffff;
    border: 1px solid #e7edf7;
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(35, 55, 80, 0.05);
}

.panel-title {
    font-size: 16px;
    font-weight: 700;
    margin-bottom: 8px;
}

.panel-desc {
    color: #5f6b84;
    font-size: 13px;
    line-height: 1.65;
}

.hero-wrap {
    background: linear-gradient(135deg, #ffffff 0%, #f5f8ff 100%);
    border: 1px solid #e7edf7;
    border-radius: 18px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 14px rgba(38, 58, 90, 0.05);
}

.hero-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 4px;
}

.hero-subtitle {
    color: #5b6780;
    font-size: 14px;
    line-height: 1.7;
}

.work-card {
    background: #ffffff;
    border: 1px solid #e7edf7;
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 12px;
    box-shadow: 0 2px 12px rgba(35, 55, 80, 0.04);
}

.copilot-wrap {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #e6edf7;
    border-radius: 18px;
    padding: 14px;
    box-shadow: 0 2px 14px rgba(35, 55, 80, 0.05);
}

.copilot-title {
    font-size: 20px;
    font-weight: 800;
    margin-bottom: 2px;
}

.copilot-sub {
    color: #627089;
    font-size: 13px;
    margin-bottom: 10px;
}

.result-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #e2eaf6;
    border-radius: 16px;
    padding: 12px 14px;
    margin-top: 8px;
    margin-bottom: 8px;
}

.result-card-title {
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 6px;
}

.result-tag {
    display: inline-block;
    background: #eef4ff;
    color: #335caa;
    border: 1px solid #d8e5ff;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 12px;
    margin-right: 6px;
    margin-bottom: 6px;
}

.evidence-item {
    background: #fbfcff;
    border: 1px solid #e7edf8;
    border-radius: 14px;
    padding: 10px 12px;
    margin-top: 8px;
}

.evidence-title {
    font-weight: 700;
    font-size: 13px;
    margin-bottom: 4px;
}

.evidence-text {
    font-size: 13px;
    color: #475467;
    line-height: 1.7;
}

div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5ebf6;
    border-radius: 14px;
    padding: 8px 10px;
}

.small-note {
    color: #667085;
    font-size: 13px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 默认知识片段 / 默认数据库路径
# =========================
DEFAULT_KNOWLEDGE = """\
rHDL 制剂在胆固醇比例过高时，可能出现沉淀或体系不稳定。
超声过程温度过低时，均质效率可能下降，导致颗粒分散不充分。
适中的脂质/蛋白比例通常更有利于颗粒稳定性。
包封率通常受脂质比例、蛋白比例、温度和处理时间共同影响。
若体系出现明显白色沉淀，应优先排查胆固醇比例、温度以及原料溶解状态。
"""

DEFAULT_DB_PATHS = [
    "data/rHDL脂质组成表最新版.xlsx",
    "rHDL脂质组成表最新版.xlsx",
]

# =========================
# Session State
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "knowledge_text" not in st.session_state:
    st.session_state.knowledge_text = DEFAULT_KNOWLEDGE

if "project_name" not in st.session_state:
    st.session_state.project_name = "脑靶向 rHDL 项目"

if "project_type" not in st.session_state:
    st.session_state.project_type = "处方开发"

if "project_desc" not in st.session_state:
    st.session_state.project_desc = "围绕纳米制剂/脂蛋白体系，进行文献检索、数据库查询、处方预测与逆向推荐。"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

if "db_uploaded_bytes" not in st.session_state:
    st.session_state.db_uploaded_bytes = None

if "db_uploaded_name" not in st.session_state:
    st.session_state.db_uploaded_name = None

if "db_sheet_name" not in st.session_state:
    st.session_state.db_sheet_name = "最终版本"

if "active_df" not in st.session_state:
    st.session_state.active_df = None

if "active_db_name" not in st.session_state:
    st.session_state.active_db_name = "未加载"

# =========================
# 数据处理函数
# =========================
def find_local_db_path() -> Optional[str]:
    for p in DEFAULT_DB_PATHS:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def get_excel_sheet_names_from_path(path: str) -> List[str]:
    excel_file = pd.ExcelFile(path)
    return excel_file.sheet_names


@st.cache_data(show_spinner=False)
def get_excel_sheet_names_from_bytes(file_bytes: bytes) -> List[str]:
    excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
    return excel_file.sheet_names


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 去掉 Unnamed 和全空列
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    # 清理字符串字段
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": None, "None": None, "": None})

    # 尝试转数值
    numeric_candidates = []
    for col in df.columns:
        lc = str(col).lower()
        if any(k in lc for k in [
            "ratio", "percent", "size", "pdi", "zeta", "temp", "time", "ph_",
            "efflux", "dl", "ee", "cycles"
        ]):
            numeric_candidates.append(col)

    numeric_candidates = list(set(numeric_candidates))
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_dataframe_from_path(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)
    return clean_dataframe(df)


@st.cache_data(show_spinner=False)
def load_dataframe_from_bytes(file_bytes: bytes, file_name: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    return clean_dataframe(df)


def get_active_df() -> Optional[pd.DataFrame]:
    df = st.session_state.get("active_df", None)
    if isinstance(df, pd.DataFrame):
        return df
    return None


def apply_dashboard_filters(
    df: pd.DataFrame,
    phos_1_type: Optional[str] = None,
    apo_type: Optional[str] = None,
    indication: Optional[str] = None,
    method_assembly: Optional[str] = None,
    shape_observed: Optional[str] = None,
    max_size_nm: Optional[float] = None,
    max_pdi: Optional[float] = None,
    min_ee_percent: Optional[float] = None,
) -> pd.DataFrame:
    out = df.copy()

    if phos_1_type and "phos_1_type" in out.columns:
        out = out[out["phos_1_type"] == phos_1_type]

    if apo_type and "apo_type" in out.columns:
        out = out[out["apo_type"] == apo_type]

    if indication and "Indication" in out.columns:
        out = out[out["Indication"] == indication]

    if method_assembly and "method_assembly" in out.columns:
        out = out[out["method_assembly"] == method_assembly]

    if shape_observed and "Shape_Observed" in out.columns:
        out = out[out["Shape_Observed"] == shape_observed]

    if max_size_nm is not None and "Size_Mean_nm" in out.columns:
        out = out[out["Size_Mean_nm"].fillna(10**9) <= max_size_nm]

    if max_pdi is not None and "PDI" in out.columns:
        out = out[out["PDI"].fillna(10**9) <= max_pdi]

    if min_ee_percent is not None and "EE_Percent" in out.columns:
        out = out[out["EE_Percent"].fillna(-10**9) >= min_ee_percent]

    return out


def extract_terms(text: str) -> List[str]:
    text = text.lower().strip()
    zh_terms = re.findall(r'[\u4e00-\u9fff]{2,}', text)
    en_terms = re.findall(r'[a-zA-Z0-9_]+', text)
    terms = zh_terms + en_terms
    terms = [t.strip() for t in terms if t.strip()]

    seen = set()
    uniq = []
    for t in terms:
        if t not in seen:
            uniq.append(t)
            seen.add(t)
    return uniq


def normalize_uploaded_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned_lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r'^\s*#+\s*', '', line)
        line = re.sub(r'^\s*[-*+]\s*', '', line)
        line = re.sub(r'^\s*\d+\.\s*', '', line)
        if line:
            cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def decode_uploaded_file(file) -> str:
    raw = file.read()
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb2312"):
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")

# =========================
# Tool 实现
# =========================
def search_knowledge_impl(question: str, top_k: int = 4) -> Dict[str, Any]:
    kb = st.session_state.knowledge_text.strip()
    if not kb:
        return {"hits": []}

    lines = [line.strip() for line in kb.splitlines() if line.strip()]
    query_terms = extract_terms(question)

    scored = []
    for idx, line in enumerate(lines):
        text = line.lower()
        score = 0

        for term in query_terms:
            if term in text:
                score += 2

        if any(k in text for k in ["沉淀", "包封率", "温度", "脂质", "蛋白", "cholesterol", "precipitation"]):
            score += 1

        if score > 0:
            scored.append({
                "chunk_id": f"local_{idx+1}",
                "score": score,
                "text": line,
                "source": "本地知识库"
            })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"hits": scored[:top_k]}


def _toy_formula(lipid_ratio: float, protein_ratio: float, temperature: float, time_min: float) -> Dict[str, float]:
    ee = (
        55
        + 5.0 * lipid_ratio
        + 6.0 * protein_ratio
        + 0.18 * temperature
        + 0.25 * time_min
        - 0.45 * (lipid_ratio - 5.0) ** 2
        - 0.90 * (protein_ratio - 1.5) ** 2
    )
    size = (
        160
        - 7.0 * lipid_ratio
        + 6.5 * protein_ratio
        - 0.7 * temperature
        - 0.4 * time_min
        + 0.30 * (lipid_ratio - 4.0) ** 2
    )
    pdi = (
        0.26
        + 0.015 * abs(protein_ratio - 1.3)
        + 0.005 * abs(lipid_ratio - 4.8)
        - 0.0015 * time_min
    )

    ee = max(20.0, min(95.0, ee))
    size = max(45.0, min(300.0, size))
    pdi = max(0.05, min(0.60, pdi))

    return {
        "predicted_ee": round(ee, 2),
        "predicted_size_nm": round(size, 2),
        "predicted_pdi": round(pdi, 3),
    }


def predict_formulation_impl(
    lipid_ratio: float,
    protein_ratio: float,
    temperature: float,
    time_min: float
) -> Dict[str, Any]:
    pred = _toy_formula(lipid_ratio, protein_ratio, temperature, time_min)
    return {
        "输入": {
            "脂质比例": lipid_ratio,
            "蛋白比例": protein_ratio,
            "温度": temperature,
            "时间(min)": time_min,
        },
        "预测包封率(%)": pred["predicted_ee"],
        "预测粒径(nm)": pred["predicted_size_nm"],
        "预测PDI": pred["predicted_pdi"],
        "说明": "当前为演示公式，后续可替换为真实 sklearn/xgboost 模型。"
    }


def reverse_design_impl(target_ee_min: float = 80.0, top_k: int = 5) -> Dict[str, Any]:
    candidates = []

    for _ in range(400):
        params = {
            "lipid_ratio": round(random.uniform(1.0, 8.0), 2),
            "protein_ratio": round(random.uniform(0.4, 2.5), 2),
            "temperature": round(random.uniform(20.0, 45.0), 1),
            "time_min": round(random.uniform(5.0, 40.0), 1),
        }
        pred = _toy_formula(**params)

        if pred["predicted_ee"] >= target_ee_min:
            score = pred["predicted_ee"] - 0.03 * pred["predicted_size_nm"] - 15 * pred["predicted_pdi"]
            candidates.append({
                "脂质比例": params["lipid_ratio"],
                "蛋白比例": params["protein_ratio"],
                "温度": params["temperature"],
                "时间(min)": params["time_min"],
                "预测包封率(%)": pred["predicted_ee"],
                "预测粒径(nm)": pred["predicted_size_nm"],
                "预测PDI": pred["predicted_pdi"],
                "综合评分": round(score, 3),
            })

    candidates.sort(key=lambda x: x["综合评分"], reverse=True)
    return {
        "目标最低包封率(%)": target_ee_min,
        "candidates": candidates[:top_k],
        "说明": "当前为随机采样 + 演示评分，后续可替换为真实优化算法。"
    }


def query_formulation_database_impl(
    phos_1_type: Optional[str] = None,
    apo_type: Optional[str] = None,
    indication: Optional[str] = None,
    method_assembly: Optional[str] = None,
    shape_observed: Optional[str] = None,
    max_size_nm: Optional[float] = None,
    max_pdi: Optional[float] = None,
    min_ee_percent: Optional[float] = None,
    top_k: int = 10,
    sort_by: Optional[str] = None,
    ascending: bool = True,
) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。请先上传或放置 Excel 数据表。"}

    filtered = apply_dashboard_filters(
        df,
        phos_1_type=phos_1_type,
        apo_type=apo_type,
        indication=indication,
        method_assembly=method_assembly,
        shape_observed=shape_observed,
        max_size_nm=max_size_nm,
        max_pdi=max_pdi,
        min_ee_percent=min_ee_percent,
    )

    if sort_by and sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=ascending)
    else:
        if "EE_Percent" in filtered.columns:
            filtered = filtered.sort_values(by="EE_Percent", ascending=False, na_position="last")
        elif "Size_Mean_nm" in filtered.columns:
            filtered = filtered.sort_values(by="Size_Mean_nm", ascending=True, na_position="last")

    preferred_cols = [
        "ref_id", "formulation_name", "phos_1_type", "phos_1_ratio", "chol_ratio",
        "apo_type", "apo_ratio", "method_assembly", "Shape_Observed",
        "Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent", "Indication"
    ]
    cols = [c for c in preferred_cols if c in filtered.columns]
    if not cols:
        cols = list(filtered.columns[:12])

    preview_df = filtered[cols].head(top_k).copy()
    preview_df = preview_df.where(pd.notnull(preview_df), None)

    return {
        "matched_count": int(len(filtered)),
        "preview_count": int(len(preview_df)),
        "records": preview_df.to_dict(orient="records"),
        "used_columns": cols,
    }


def aggregate_formulation_database_impl(
    group_by: str,
    metric: str,
    agg: str = "mean",
    top_k: int = 10,
    ascending: bool = False,
) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。"}

    if group_by not in df.columns:
        return {"error": f"group_by 字段不存在：{group_by}"}

    if metric not in df.columns:
        return {"error": f"metric 字段不存在：{metric}"}

    work = df[[group_by, metric]].copy()
    work = work.dropna()
    if work.empty:
        return {"error": "可用于聚合的数据为空。"}

    if agg not in ["mean", "median", "max", "min", "count"]:
        agg = "mean"

    if agg == "count":
        result = work.groupby(group_by)[metric].count().reset_index(name=f"{metric}_{agg}")
    else:
        result = work.groupby(group_by)[metric].agg(agg).reset_index(name=f"{metric}_{agg}")

    result = result.sort_values(by=f"{metric}_{agg}", ascending=ascending).head(top_k)
    result = result.where(pd.notnull(result), None)

    return {
        "group_by": group_by,
        "metric": metric,
        "agg": agg,
        "table": result.to_dict(orient="records")
    }

# =========================
# Tool 注册
# =========================
TOOL_IMPL = {
    "search_knowledge": search_knowledge_impl,
    "predict_formulation": predict_formulation_impl,
    "reverse_design": reverse_design_impl,
    "query_formulation_database": query_formulation_database_impl,
    "aggregate_formulation_database": aggregate_formulation_database_impl,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "检索本地知识库中的实验经验、文献摘要、故障诊断说明。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 4}
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "predict_formulation",
            "description": "根据数值工艺参数预测处方表现。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lipid_ratio": {"type": "number"},
                    "protein_ratio": {"type": "number"},
                    "temperature": {"type": "number"},
                    "time_min": {"type": "number"}
                },
                "required": ["lipid_ratio", "protein_ratio", "temperature", "time_min"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_design",
            "description": "根据目标包封率推荐若干候选处方参数。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_ee_min": {"type": "number", "default": 80.0},
                    "top_k": {"type": "integer", "default": 5}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_formulation_database",
            "description": "按条件筛选结构化处方数据库，返回符合条件的样本记录。",
            "parameters": {
                "type": "object",
                "properties": {
                    "phos_1_type": {"type": "string"},
                    "apo_type": {"type": "string"},
                    "indication": {"type": "string"},
                    "method_assembly": {"type": "string"},
                    "shape_observed": {"type": "string"},
                    "max_size_nm": {"type": "number"},
                    "max_pdi": {"type": "number"},
                    "min_ee_percent": {"type": "number"},
                    "top_k": {"type": "integer", "default": 10},
                    "sort_by": {"type": "string"},
                    "ascending": {"type": "boolean", "default": True}
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_formulation_database",
            "description": "对结构化数据库按某字段分组，并对某个指标做聚合统计，例如平均粒径、平均包封率等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string"},
                    "metric": {"type": "string"},
                    "agg": {"type": "string", "default": "mean"},
                    "top_k": {"type": "integer", "default": 10},
                    "ascending": {"type": "boolean", "default": False}
                },
                "required": ["group_by", "metric"]
            }
        }
    }
]

# =========================
# Agent
# =========================
BASE_SYSTEM_PROMPT = """
你是一个医学/药剂学科研智能体原型。

规则：
1. 用户涉及文献经验检索、数据库筛选、统计分析、预测、逆向推荐时，优先调用工具。
2. 不要伪造数值结果。
3. 用户提到异常现象、原因分析、经验总结时，优先调用 search_knowledge。
4. 用户提到筛选处方、比较样本、按字段统计时，优先调用 query_formulation_database 或 aggregate_formulation_database。
5. 默认使用中文回答。
6. 如果结果来自演示公式，必须明确说明它只是 demo。
7. 输出风格尽量清晰、结构化、简洁。
"""


def build_system_prompt() -> str:
    prompt = BASE_SYSTEM_PROMPT
    df = get_active_df()
    if df is not None and not df.empty:
        cols = list(df.columns)
        prompt += f"\n当前已加载结构化数据库，共 {len(df)} 条记录。可用字段包括：{', '.join(cols[:40])}。"
    return prompt


def run_agent(user_text: str) -> Dict[str, Any]:
    tool_logs: List[Dict[str, Any]] = []
    tool_result_by_call_id: Dict[str, Any] = {}

    messages = [{"role": "system", "content": build_system_prompt()}]
    for msg in st.session_state.messages[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    for _ in range(6):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message

        if not getattr(msg, "tool_calls", None):
            return {
                "answer": msg.content or "",
                "tool_logs": tool_logs
            }

        assistant_message = {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": []
        }

        for tool_call in msg.tool_calls:
            tool_name = tool_call.function.name
            arguments = tool_call.function.arguments or "{}"

            try:
                args = json.loads(arguments)
            except json.JSONDecodeError:
                args = {}

            if tool_name not in TOOL_IMPL:
                result = {"error": f"未知工具：{tool_name}"}
            else:
                try:
                    result = TOOL_IMPL[tool_name](**args)
                except Exception as e:
                    result = {"error": f"{tool_name} 执行失败：{str(e)}"}

            tool_logs.append({
                "tool": tool_name,
                "arguments": args,
                "result": result,
            })
            tool_result_by_call_id[tool_call.id] = result

            assistant_message["tool_calls"].append({
                "id": tool_call.id,
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args, ensure_ascii=False)
                }
            })

        messages.append(assistant_message)

        for tool_call in msg.tool_calls:
            tool_result = tool_result_by_call_id.get(tool_call.id, {"error": "没有找到工具结果"})
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(tool_result, ensure_ascii=False)
            })

    return {
        "answer": "工具调用次数超过上限，本轮已停止。",
        "tool_logs": tool_logs
    }

# =========================
# 渲染函数
# =========================
def render_knowledge_hits(hits: List[Dict[str, Any]]):
    if not hits:
        return
    st.markdown("""
    <div class="result-card">
        <div class="result-card-title">检索到的本地证据</div>
    """, unsafe_allow_html=True)
    for hit in hits:
        st.markdown(
            f"""
            <div class="evidence-item">
                <div class="evidence-title">条目 {hit.get("chunk_id", "-")} · 相关度 {hit.get("score", "-")}</div>
                <div class="evidence-text">{hit.get("text", "")}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_card(result: Dict[str, Any]):
    st.markdown("""
    <div class="result-card">
        <div class="result-card-title">预测结果卡片</div>
        <span class="result-tag">演示模型</span>
        <span class="result-tag">数值预测</span>
    </div>
    """, unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([result]), use_container_width=True)


def render_reverse_card(result: Dict[str, Any]):
    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-title">逆向推荐结果</div>
        <span class="result-tag">目标包封率 ≥ {result.get("目标最低包封率(%)", "-")}</span>
        <span class="result-tag">候选参数</span>
    </div>
    """, unsafe_allow_html=True)
    if result.get("candidates"):
        st.dataframe(pd.DataFrame(result["candidates"]), use_container_width=True)


def render_query_card(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-title">数据库筛选结果</div>
        <span class="result-tag">匹配记录 {result.get("matched_count", 0)} 条</span>
        <span class="result-tag">展示 {result.get("preview_count", 0)} 条</span>
    </div>
    """, unsafe_allow_html=True)

    records = result.get("records", [])
    if records:
        st.dataframe(pd.DataFrame(records), use_container_width=True)


def render_aggregate_card(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        return

    st.markdown(f"""
    <div class="result-card">
        <div class="result-card-title">分组统计结果</div>
        <span class="result-tag">{result.get("group_by", "-")}</span>
        <span class="result-tag">{result.get("metric", "-")} · {result.get("agg", "-")}</span>
    </div>
    """, unsafe_allow_html=True)

    table = result.get("table", [])
    if table:
        st.dataframe(pd.DataFrame(table), use_container_width=True)


def trigger_prompt(prompt_text: str):
    st.session_state.pending_prompt = prompt_text
    st.rerun()

# =========================
# 侧边栏：项目区 / 数据区 / 知识区
# =========================
with st.sidebar:
    # 项目区
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🗂️ 项目区</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">维护当前项目基本信息。</div>', unsafe_allow_html=True)

    st.session_state.project_name = st.text_input("项目名称", value=st.session_state.project_name)
    st.session_state.project_type = st.selectbox(
        "项目类型",
        ["处方开发", "文献综述", "异常诊断", "工艺优化", "数据库整理", "其他"],
        index=["处方开发", "文献综述", "异常诊断", "工艺优化", "数据库整理", "其他"].index(st.session_state.project_type)
        if st.session_state.project_type in ["处方开发", "文献综述", "异常诊断", "工艺优化", "数据库整理", "其他"] else 0
    )
    st.session_state.project_desc = st.text_area("项目说明", value=st.session_state.project_desc, height=90)
    st.markdown('</div>', unsafe_allow_html=True)

    # 数据区
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🧾 数据区</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">支持读取仓库中的 data 文件夹，也支持手动上传 xlsx / csv。</div>', unsafe_allow_html=True)

    db_file = st.file_uploader(
        "上传数据库文件",
        type=["xlsx", "csv"],
        help="可上传 xlsx 或 csv"
    )

    if db_file is not None:
        st.session_state.db_uploaded_bytes = db_file.getvalue()
        st.session_state.db_uploaded_name = db_file.name

    source_mode = "未找到数据文件"
    sheet_options = []
    chosen_sheet = None

    if st.session_state.db_uploaded_bytes is not None:
        source_mode = f"已上传：{st.session_state.db_uploaded_name}"
        if str(st.session_state.db_uploaded_name).lower().endswith(".xlsx"):
            sheet_options = get_excel_sheet_names_from_bytes(st.session_state.db_uploaded_bytes)
        else:
            sheet_options = ["CSV"]
    else:
        local_path = find_local_db_path()
        if local_path:
            source_mode = f"仓库数据文件：{local_path}"
            if local_path.lower().endswith(".xlsx"):
                sheet_options = get_excel_sheet_names_from_path(local_path)
            else:
                sheet_options = ["CSV"]

    st.info(f"当前数据源：{source_mode}")

    if sheet_options:
        default_idx = 0
        if st.session_state.db_sheet_name in sheet_options:
            default_idx = sheet_options.index(st.session_state.db_sheet_name)
        chosen_sheet = st.selectbox("选择工作表", sheet_options, index=default_idx)
        st.session_state.db_sheet_name = chosen_sheet

        try:
            if st.session_state.db_uploaded_bytes is not None:
                if str(st.session_state.db_uploaded_name).lower().endswith(".csv"):
                    df_active = load_dataframe_from_bytes(
                        st.session_state.db_uploaded_bytes,
                        st.session_state.db_uploaded_name,
                        None,
                    )
                else:
                    df_active = load_dataframe_from_bytes(
                        st.session_state.db_uploaded_bytes,
                        st.session_state.db_uploaded_name,
                        chosen_sheet,
                    )
                st.session_state.active_db_name = st.session_state.db_uploaded_name
            else:
                local_path = find_local_db_path()
                if local_path:
                    if local_path.lower().endswith(".csv"):
                        df_active = load_dataframe_from_path(local_path, None)
                    else:
                        df_active = load_dataframe_from_path(local_path, chosen_sheet)
                    st.session_state.active_db_name = os.path.basename(local_path)
                else:
                    df_active = None
                    st.session_state.active_db_name = "未加载"

            st.session_state.active_df = df_active
        except Exception as e:
            st.error(f"数据库加载失败：{e}")
            st.session_state.active_df = None

    if st.button("清除已上传数据库", use_container_width=True):
        st.session_state.db_uploaded_bytes = None
        st.session_state.db_uploaded_name = None
        st.session_state.active_df = None
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # 知识库区
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📚 知识库区</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">可粘贴经验记录，也支持导入 txt / md 文件。</div>', unsafe_allow_html=True)

    kb_files = st.file_uploader(
        "导入知识文件",
        type=["txt", "md"],
        accept_multiple_files=True
    )

    import_mode = st.radio("导入方式", ["追加到现有内容", "替换现有内容"])

    colk1, colk2 = st.columns(2)
    with colk1:
        if st.button("加载示例知识", use_container_width=True):
            st.session_state.knowledge_text = DEFAULT_KNOWLEDGE
            st.rerun()
    with colk2:
        if st.button("清空知识库", use_container_width=True):
            st.session_state.knowledge_text = ""
            st.rerun()

    if st.button("导入知识文件", use_container_width=True):
        if kb_files:
            all_text = []
            for f in kb_files:
                text = decode_uploaded_file(f)
                text = normalize_uploaded_text(text)
                if text.strip():
                    all_text.append(text)
            merged = "\n".join(all_text).strip()
            if merged:
                if import_mode == "替换现有内容":
                    st.session_state.knowledge_text = merged
                else:
                    current = st.session_state.knowledge_text.strip()
                    st.session_state.knowledge_text = current + ("\n" if current else "") + merged
                st.success("知识文件导入成功。")
                st.rerun()

    st.text_area("知识片段", key="knowledge_text", height=220)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 顶部
# =========================
df_main = get_active_df()
db_count = len(df_main) if isinstance(df_main, pd.DataFrame) else 0

st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-title">医学智能体平台 · 数据驱动版</div>
    <div class="hero-subtitle">
        当前项目：<b>{st.session_state.project_name}</b> ｜ 模型：<b>{MODEL}</b> ｜ 当前数据库：<b>{st.session_state.active_db_name}</b><br>
        当前已支持：本地知识检索、结构化数据库筛选、分组统计、演示型参数预测、逆向推荐。
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# 主体布局：左工作区 + 右 Copilot
# =========================
left_col, right_col = st.columns([3.2, 1.4], gap="large")

# =========================
# 左侧主工作区
# =========================
with left_col:
    st.markdown('<div class="work-card">', unsafe_allow_html=True)
    st.markdown("### 📊 数据总览")

    if df_main is None or df_main.empty:
        st.warning("当前还没有可用的结构化数据库。你可以把 Excel 放到 data/ 目录，或者在左侧上传 xlsx/csv。")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)

        ref_count = int(df_main["ref_id"].nunique()) if "ref_id" in df_main.columns else 0
        phos_count = int(df_main["phos_1_type"].nunique()) if "phos_1_type" in df_main.columns else 0
        apo_count = int(df_main["apo_type"].nunique()) if "apo_type" in df_main.columns else 0
        valid_size = int(df_main["Size_Mean_nm"].notna().sum()) if "Size_Mean_nm" in df_main.columns else 0
        valid_ee = int(df_main["EE_Percent"].notna().sum()) if "EE_Percent" in df_main.columns else 0

        c1.metric("记录数", len(df_main))
        c2.metric("文献ID数", ref_count)
        c3.metric("磷脂类型数", phos_count)
        c4.metric("Apo类型数", apo_count)
        c5.metric("有效EE数", valid_ee)

    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📌 总览看板", "🗃️ 数据库查询", "📈 统计分析", "📚 知识库预览"])

    # ---------- Tab1: 总览 ----------
    with tab1:
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            r1c1, r1c2 = st.columns(2)

            with r1c1:
                if "phos_1_type" in df_main.columns:
                    top_phos = (
                        df_main["phos_1_type"]
                        .dropna()
                        .value_counts()
                        .head(10)
                        .reset_index()
                    )
                    top_phos.columns = ["phos_1_type", "count"]
                    fig = px.bar(top_phos, x="phos_1_type", y="count", title="Top 10 磷脂类型分布")
                    st.plotly_chart(fig, use_container_width=True)

            with r1c2:
                if "apo_type" in df_main.columns:
                    top_apo = (
                        df_main["apo_type"]
                        .dropna()
                        .value_counts()
                        .head(10)
                        .reset_index()
                    )
                    top_apo.columns = ["apo_type", "count"]
                    fig = px.bar(top_apo, x="apo_type", y="count", title="Top 10 Apo 类型分布")
                    st.plotly_chart(fig, use_container_width=True)

            r2c1, r2c2 = st.columns(2)

            with r2c1:
                if "Size_Mean_nm" in df_main.columns and df_main["Size_Mean_nm"].notna().sum() > 0:
                    fig = px.histogram(df_main.dropna(subset=["Size_Mean_nm"]), x="Size_Mean_nm", nbins=30, title="粒径分布")
                    st.plotly_chart(fig, use_container_width=True)

            with r2c2:
                if "EE_Percent" in df_main.columns and df_main["EE_Percent"].notna().sum() > 0:
                    fig = px.histogram(df_main.dropna(subset=["EE_Percent"]), x="EE_Percent", nbins=30, title="包封率分布")
                    st.plotly_chart(fig, use_container_width=True)

            if all(col in df_main.columns for col in ["Size_Mean_nm", "PDI"]) and df_main[["Size_Mean_nm", "PDI"]].dropna().shape[0] > 0:
                color_col = "phos_1_type" if "phos_1_type" in df_main.columns else None
                fig = px.scatter(
                    df_main.dropna(subset=["Size_Mean_nm", "PDI"]).head(400),
                    x="Size_Mean_nm",
                    y="PDI",
                    color=color_col,
                    title="粒径 - PDI 散点图",
                    hover_data=[c for c in ["apo_type", "EE_Percent", "method_assembly"] if c in df_main.columns]
                )
                st.plotly_chart(fig, use_container_width=True)

    # ---------- Tab2: 数据库查询 ----------
    with tab2:
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            st.markdown("#### 条件筛选")

            q1, q2, q3, q4 = st.columns(4)

            phos_options = ["全部"] + sorted([x for x in df_main["phos_1_type"].dropna().unique().tolist()]) if "phos_1_type" in df_main.columns else ["全部"]
            apo_options = ["全部"] + sorted([x for x in df_main["apo_type"].dropna().unique().tolist()]) if "apo_type" in df_main.columns else ["全部"]
            indication_options = ["全部"] + sorted([x for x in df_main["Indication"].dropna().unique().tolist()]) if "Indication" in df_main.columns else ["全部"]
            assembly_options = ["全部"] + sorted([x for x in df_main["method_assembly"].dropna().unique().tolist()]) if "method_assembly" in df_main.columns else ["全部"]
            shape_options = ["全部"] + sorted([x for x in df_main["Shape_Observed"].dropna().unique().tolist()]) if "Shape_Observed" in df_main.columns else ["全部"]

            with q1:
                sel_phos = st.selectbox("phos_1_type", phos_options)
                sel_apo = st.selectbox("apo_type", apo_options)
            with q2:
                sel_ind = st.selectbox("Indication", indication_options)
                sel_method = st.selectbox("method_assembly", assembly_options)
            with q3:
                sel_shape = st.selectbox("Shape_Observed", shape_options)
                max_size = st.number_input("最大粒径 (nm)", min_value=0.0, value=150.0, step=5.0)
            with q4:
                max_pdi = st.number_input("最大 PDI", min_value=0.0, value=0.30, step=0.01, format="%.2f")
                min_ee = st.number_input("最小 EE_Percent", min_value=0.0, value=0.0, step=1.0)

            filtered_df = apply_dashboard_filters(
                df_main,
                phos_1_type=None if sel_phos == "全部" else sel_phos,
                apo_type=None if sel_apo == "全部" else sel_apo,
                indication=None if sel_ind == "全部" else sel_ind,
                method_assembly=None if sel_method == "全部" else sel_method,
                shape_observed=None if sel_shape == "全部" else sel_shape,
                max_size_nm=max_size if "Size_Mean_nm" in df_main.columns else None,
                max_pdi=max_pdi if "PDI" in df_main.columns else None,
                min_ee_percent=min_ee if "EE_Percent" in df_main.columns and min_ee > 0 else None,
            )

            st.success(f"筛选后共 {len(filtered_df)} 条记录。")

            preferred_cols = [
                "ref_id", "formulation_name", "phos_1_type", "phos_1_ratio", "chol_ratio",
                "apo_type", "apo_ratio", "method_assembly", "Shape_Observed",
                "Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent", "Indication"
            ]
            show_cols = [c for c in preferred_cols if c in filtered_df.columns]
            if not show_cols:
                show_cols = list(filtered_df.columns[:15])

            st.dataframe(filtered_df[show_cols], use_container_width=True, height=420)

            csv_data = filtered_df[show_cols].to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "下载当前筛选结果 CSV",
                data=csv_data,
                file_name="filtered_formulations.csv",
                mime="text/csv",
                use_container_width=False
            )

    # ---------- Tab3: 统计分析 ----------
    with tab3:
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            st.markdown("#### 分组统计")

            categorical_cols = [c for c in ["phos_1_type", "apo_type", "method_assembly", "Shape_Observed", "Indication"] if c in df_main.columns]
            numeric_cols = [c for c in ["Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent", "cholesterol efflux(peptide c=10ug/mL)", "cholesterol efflux(peptide c=50ug/mL)"] if c in df_main.columns]

            a1, a2, a3 = st.columns(3)
            with a1:
                group_col = st.selectbox("分组字段", categorical_cols if categorical_cols else ["无可用字段"])
            with a2:
                metric_col = st.selectbox("统计指标", numeric_cols if numeric_cols else ["无可用字段"])
            with a3:
                agg_method = st.selectbox("聚合方式", ["mean", "median", "max", "min", "count"])

            if categorical_cols and numeric_cols:
                agg_result = aggregate_formulation_database_impl(group_col, metric_col, agg_method, top_k=20, ascending=False)
                if "table" in agg_result:
                    result_df = pd.DataFrame(agg_result["table"])
                    st.dataframe(result_df, use_container_width=True)

                    value_col = f"{metric_col}_{agg_method}"
                    if not result_df.empty and value_col in result_df.columns:
                        fig = px.bar(result_df, x=group_col, y=value_col, title=f"{group_col} - {metric_col} ({agg_method})")
                        st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### 双变量分析")

            s1, s2, s3 = st.columns(3)
            with s1:
                x_metric = st.selectbox("X 轴", numeric_cols if numeric_cols else ["无可用字段"], key="x_metric")
            with s2:
                y_metric = st.selectbox("Y 轴", numeric_cols if numeric_cols else ["无可用字段"], key="y_metric", index=1 if len(numeric_cols) > 1 else 0)
            with s3:
                color_field_candidates = ["无"] + categorical_cols
                color_field = st.selectbox("颜色分组", color_field_candidates)

            if numeric_cols:
                plot_df = df_main.dropna(subset=[x_metric, y_metric]).copy()
                if not plot_df.empty:
                    fig = px.scatter(
                        plot_df.head(600),
                        x=x_metric,
                        y=y_metric,
                        color=None if color_field == "无" else color_field,
                        title=f"{x_metric} vs {y_metric}"
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # ---------- Tab4: 知识库预览 ----------
    with tab4:
        st.markdown("#### 当前知识库内容")
        lines = [x for x in st.session_state.knowledge_text.splitlines() if x.strip()]
        st.metric("知识条目数", len(lines))
        st.text_area("知识内容预览", value=st.session_state.knowledge_text, height=420)

# =========================
# 右侧 Copilot
# =========================
with right_col:
    st.markdown('<div class="copilot-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="copilot-title">🤖 Copilot 助手</div>', unsafe_allow_html=True)
    st.markdown('<div class="copilot-sub">右侧窄栏问答区。你可以直接问知识库问题、数据库筛选问题，或者做预测/推荐。</div>', unsafe_allow_html=True)

    qa1, qa2 = st.columns(2)
    with qa1:
        if st.button("沉淀分析", use_container_width=True):
            trigger_prompt("为什么超声后出现白色沉淀？")
    with qa2:
        if st.button("参数预测", use_container_width=True):
            trigger_prompt("预测一下：lipid_ratio=4.5, protein_ratio=1.2, temperature=37, time_min=20")

    qb1, qb2 = st.columns(2)
    with qb1:
        if st.button("逆向推荐", use_container_width=True):
            trigger_prompt("帮我设计几组包封率大于80的参数")
    with qb2:
        if st.button("数据库筛选", use_container_width=True):
            trigger_prompt("帮我筛选 apo_type=22A 且 Size_Mean_nm 小于 100 的处方")

    st.markdown("---")

    for msg in st.session_state.messages:
        avatar = "🧑‍🔬" if msg["role"] == "user" else "🤖"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

            if msg.get("tool_logs"):
                for log in msg["tool_logs"]:
                    if log["tool"] == "search_knowledge" and "hits" in log["result"]:
                        render_knowledge_hits(log["result"]["hits"])
                    elif log["tool"] == "predict_formulation" and "预测包封率(%)" in log["result"]:
                        render_prediction_card(log["result"])
                    elif log["tool"] == "reverse_design" and "candidates" in log["result"]:
                        render_reverse_card(log["result"])
                    elif log["tool"] == "query_formulation_database":
                        render_query_card(log["result"])
                    elif log["tool"] == "aggregate_formulation_database":
                        render_aggregate_card(log["result"])

                with st.expander("查看工具调用详情"):
                    st.json(msg["tool_logs"])

    with st.form("copilot_form", clear_on_submit=True):
        user_text = st.text_area(
            "输入问题",
            placeholder="例如：按 apo_type 统计平均粒径；或者帮我筛选 phos_1_type=DMPC 且 PDI<0.2 的样本",
            height=90,
            label_visibility="collapsed"
        )
        f1, f2 = st.columns(2)
        submitted = f1.form_submit_button("发送", use_container_width=True)
        clear_chat = f2.form_submit_button("清空对话", use_container_width=True)

    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    prompt = None
    if submitted and user_text.strip():
        prompt = user_text.strip()
    elif st.session_state.pending_prompt:
        prompt = st.session_state.pending_prompt

    if prompt:
        st.session_state.pending_prompt = None
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user", avatar="🧑‍🔬"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.status("正在分析并决定调用哪些工具……", expanded=True) as status:
                result = run_agent(prompt)
                answer = result["answer"]
                tool_logs = result["tool_logs"]
                status.update(label="处理完成", state="complete", expanded=False)

            st.markdown(answer)

            for log in tool_logs:
                if log["tool"] == "search_knowledge" and "hits" in log["result"]:
                    render_knowledge_hits(log["result"]["hits"])
                elif log["tool"] == "predict_formulation" and "预测包封率(%)" in log["result"]:
                    render_prediction_card(log["result"])
                elif log["tool"] == "reverse_design" and "candidates" in log["result"]:
                    render_reverse_card(log["result"])
                elif log["tool"] == "query_formulation_database":
                    render_query_card(log["result"])
                elif log["tool"] == "aggregate_formulation_database":
                    render_aggregate_card(log["result"])

            with st.expander("查看工具调用详情"):
                st.json(tool_logs)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "tool_logs": tool_logs,
        })

    st.markdown('</div>', unsafe_allow_html=True)
