import os
import io
import json
import random
import re
import html
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 基础配置
# =========================
load_dotenv()

st.set_page_config(
    page_title="仿生rHDL纳米制剂开发助手",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


def get_secret(name: str, default: str = "") -> str:
    """优先读取 Streamlit secrets，其次读取 .env / 环境变量。"""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, default)


API_KEY = get_secret("API_KEY", "")
API_BASE = get_secret("API_BASE", "")
MODEL = get_secret("MODEL", "")

# 按你的要求：下载密码直接写死在程序里。
DOWNLOAD_PASSWORD = "2026"

client: Optional[OpenAI] = None
if API_KEY and API_BASE and MODEL:
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE)
    except Exception:
        client = None

DEFAULT_KNOWLEDGE = """\
rHDL 制剂在胆固醇比例过高时，可能出现沉淀或体系不稳定。
超声过程温度过低时，均质效率可能下降，导致颗粒分散不充分。
适中的脂质/蛋白比例通常更有利于颗粒稳定性。
包封率通常受脂质比例、蛋白比例、温度和处理时间共同影响。
若体系出现明显白色沉淀，应优先排查胆固醇比例、温度以及原料溶解状态。
PDI 越低通常表示粒径分布越集中，体系均一性更好。
Zeta 电位可用于辅助判断体系的分散稳定性。
"""

DEFAULT_DB_PATHS = [
    "data/rHDL脂质组成表最新版.xlsx",
    "rHDL脂质组成表最新版.xlsx",
]

FIELD_EXPLANATIONS = {
    "ref_id": "文献或来源编号，用于区分数据来源。",
    "formulation_name": "配方名称，用于快速识别具体样本。",
    "phos_1_type": "主磷脂类型，是配方的核心脂质成分。",
    "phos_1_ratio": "主磷脂对应比例。",
    "phos_2_type": "第二磷脂类型。",
    "phos_2_ratio": "第二磷脂比例。",
    "chol_ratio": "胆固醇比例，常影响膜稳定性、流动性和沉淀风险。",
    "apo_type": "载脂蛋白或肽类型。",
    "apo_ratio": "载脂蛋白或肽比例。",
    "method_assembly": "组装方法。",
    "buffer_assembly_type": "组装所用缓冲液类型。",
    "ph_assembly": "组装体系 pH。",
    "Shape_Observed": "观察到的形貌。",
    "Size_Mean_nm": "平均粒径，单位 nm。",
    "PDI": "多分散指数，越小一般说明分布越集中。",
    "Zeta_mV": "Zeta 电位，反映表面电荷特性与分散稳定性。",
    "EE_Percent": "包封率百分比。",
    "DL_Percent": "载药量百分比。",
    "Indication": "适应症或应用方向。",
}

THEME_CONFIG = {
    "浅色": {
        "bg": "#f6f7fb",
        "bg_grad_1": "#f8f9fc",
        "bg_grad_2": "#f4f6fb",
        "card": "#ffffff",
        "card_alt": "#f8fafc",
        "line": "#e7ebf3",
        "text": "#1f2937",
        "muted": "#667085",
        "sidebar_1": "#fbfcff",
        "sidebar_2": "#f3f6fd",
        "hero_1": "#ffffff",
        "hero_2": "#f3f7ff",
        "chat_1": "#faf7f3",
        "chat_2": "#f4efe9",
        "chat_line": "#e8dfd7",
        "chip_bg": "#eee7df",
        "chip_line": "#ddd3ca",
        "chip_text": "#4b5563",
        "accent": "#2563eb",
        "accent_soft": "rgba(37,99,235,0.08)",
        "input_bg": "#ffffff",
        "grid": "#d9e2ef",
    },
    "深色": {
        "bg": "#0b1220",
        "bg_grad_1": "#0b1220",
        "bg_grad_2": "#111827",
        "card": "#111827",
        "card_alt": "#172033",
        "line": "#263244",
        "text": "#e5e7eb",
        "muted": "#94a3b8",
        "sidebar_1": "#0b1220",
        "sidebar_2": "#111827",
        "hero_1": "#111827",
        "hero_2": "#172033",
        "chat_1": "#131a27",
        "chat_2": "#0f172a",
        "chat_line": "#2a3447",
        "chip_bg": "#1f2937",
        "chip_line": "#334155",
        "chip_text": "#cbd5e1",
        "accent": "#60a5fa",
        "accent_soft": "rgba(96,165,250,0.10)",
        "input_bg": "#0f172a",
        "grid": "#334155",
    },
}

# =========================
# Session State
# =========================
def init_session_state() -> None:
    defaults = {
        "messages": [],
        "knowledge_text": DEFAULT_KNOWLEDGE,
        "project_name": "仿生rHDL纳米制剂开发助手",
        "project_type": "处方开发",
        "project_desc": "围绕仿生rHDL纳米制剂，进行文献知识检索、数据库筛选、理化参数预测和处方工艺逆向推荐。",
        "pending_prompt": None,
        "db_file_bytes": None,
        "db_file_name": None,
        "db_sheet_name": None,
        "active_df": None,
        "active_db_name": "未加载",
        "theme_mode": "浅色",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # 旧版本会话兼容：如果浏览器里还保留旧项目名/简介，自动迁移到新版文案。
    if st.session_state.get("project_name") in ["靶向 rHDL 项目", "靶向tHDL项目", "靶向 tHDL 项目"]:
        st.session_state.project_name = "仿生rHDL纳米制剂开发助手"
    if st.session_state.get("project_desc") == "围绕 rHDL / 纳米制剂体系，进行文献知识检索、数据库筛选、参数预测和逆向推荐。":
        st.session_state.project_desc = "围绕仿生rHDL纳米制剂，进行文献知识检索、数据库筛选、理化参数预测和处方工艺逆向推荐。"


# =========================
# CSS 与图表样式
# =========================
def apply_theme_css(theme_name: str) -> None:
    cfg = THEME_CONFIG.get(theme_name, THEME_CONFIG["浅色"])
    chat_title = "#1f1f1f" if theme_name == "浅色" else "#e5e7eb"

    st.markdown(
        f"""
<style>
:root {{
    --bg: {cfg["bg"]};
    --card: {cfg["card"]};
    --card-alt: {cfg["card_alt"]};
    --line: {cfg["line"]};
    --text: {cfg["text"]};
    --muted: {cfg["muted"]};
    --accent: {cfg["accent"]};
    --accent-soft: {cfg["accent_soft"]};
    --input-bg: {cfg["input_bg"]};
}}

html, body, [class*="css"] {{
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
}}

.stApp {{
    background: linear-gradient(180deg, {cfg["bg_grad_1"]} 0%, {cfg["bg_grad_2"]} 100%);
    color: var(--text);
}}

.main .block-container {{
    padding-top: 0.75rem;
    padding-bottom: 1rem;
    max-width: 1820px;
}}

section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {cfg["sidebar_1"]} 0%, {cfg["sidebar_2"]} 100%);
    border-right: 1px solid var(--line);
}}

.sidebar-card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 14px 10px 14px;
    margin-bottom: 12px;
    box-shadow: 0 3px 12px rgba(15,23,42,0.06);
}}

.sidebar-title {{
    font-size: 15px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 8px;
}}

.sidebar-desc {{
    font-size: 12px;
    color: var(--muted);
    line-height: 1.7;
    margin-bottom: 8px;
}}

.hero {{
    background: linear-gradient(135deg, {cfg["hero_1"]} 0%, {cfg["hero_2"]} 100%);
    border: 1px solid var(--line);
    border-radius: 24px;
    padding: 20px 22px;
    box-shadow: 0 6px 20px rgba(15,23,42,0.06);
    margin-bottom: 14px;
}}

.hero-title {{
    font-size: 30px;
    font-weight: 900;
    color: var(--text);
    margin-bottom: 6px;
}}

.hero-sub {{
    font-size: 14px;
    color: var(--muted);
    line-height: 1.8;
}}

.soft-card {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 22px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(15,23,42,0.06);
    margin-bottom: 12px;
}}

.soft-title {{
    font-size: 16px;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 8px;
}}

.quick-card {{
    background: linear-gradient(180deg, var(--card) 0%, var(--card-alt) 100%);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 12px 14px;
    margin-bottom: 12px;
}}

.section-head {{ margin-bottom: 14px; }}
.section-title {{ font-size: 18px; font-weight: 900; color: var(--text); margin-bottom: 4px; }}
.section-sub {{ font-size: 13px; color: var(--muted); line-height: 1.8; }}

.soft-panel {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 16px 16px 14px 16px;
    box-shadow: 0 4px 16px rgba(15,23,42,0.05);
    margin-bottom: 12px;
}}

.panel-title {{ font-size: 15px; font-weight: 800; color: var(--text); margin-bottom: 10px; }}
.panel-note {{ font-size: 12px; color: var(--muted); line-height: 1.7; margin-bottom: 8px; }}

.capture-box {{
    background: var(--card);
    border: 1px dashed var(--line);
    border-radius: 16px;
    padding: 12px;
}}
.capture-title {{ font-size: 13px; font-weight: 800; color: var(--text); margin-bottom: 6px; }}
.capture-desc {{ font-size: 12px; color: var(--muted); line-height: 1.7; }}

div[data-testid="stMetric"] {{
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 8px 10px;
}}
div[data-testid="stMetricLabel"] {{ color: var(--muted); }}
div[data-testid="stMetricValue"] {{ color: var(--text); }}

[data-baseweb="tab-list"] {{
    gap: 8px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 14px;
}}
button[data-baseweb="tab"] {{
    height: 50px;
    background: var(--card);
    border: 1px solid var(--line);
    border-radius: 14px 14px 0 0;
    padding: 0 16px;
    font-weight: 800;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: var(--accent);
    border-bottom: 2px solid var(--accent);
    background: linear-gradient(180deg, var(--card) 0%, var(--accent-soft) 100%);
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: transparent;
    border-color: var(--line) !important;
    border-radius: 20px !important;
}}

div[data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
div[data-testid="stChatMessage"] {{ border-radius: 16px; }}
div[data-testid="stChatMessageContent"] {{ border-radius: 16px; }}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
label,
.stCaption,
.stText {{ color: var(--text); }}

.stTextArea textarea,
.stTextInput input,
.stNumberInput input {{
    border-radius: 16px !important;
    background: var(--input-bg) !important;
    color: var(--text) !important;
}}

.stTextArea textarea {{ min-height: 86px !important; }}
[data-testid="stForm"] {{ margin-bottom: 0 !important; }}

.stButton > button,
.stDownloadButton > button {{
    border-radius: 14px !important;
    font-weight: 700 !important;
    background: var(--card);
    color: var(--text);
    border: 1px solid var(--line);
}}
.stButton > button:hover,
.stDownloadButton > button:hover {{
    border-color: var(--accent);
    color: var(--accent);
}}

.streamlit-expanderHeader {{ font-weight: 700; }}
hr {{ border-color: var(--line); }}

.copilot-shell {{ position: sticky; top: 0.75rem; }}
.copilot-card {{
    background: linear-gradient(180deg, {cfg["chat_1"]} 0%, {cfg["chat_2"]} 100%);
    border: 1px solid {cfg["chat_line"]};
    border-radius: 30px;
    box-shadow: 0 12px 34px rgba(15,23,42,0.10);
    padding: 12px;
}}
.copilot-topbar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 6px 8px 6px 8px;
}}
.copilot-chip {{
    display: inline-block;
    font-size: 11px;
    color: {cfg["chip_text"]};
    background: {cfg["chip_bg"]};
    border: 1px solid {cfg["chip_line"]};
    padding: 5px 10px;
    border-radius: 999px;
    margin-right: 6px;
}}
.copilot-title {{
    font-size: 32px;
    font-weight: 900;
    color: {chat_title};
    line-height: 1.24;
    text-align: center;
    margin-top: 10px;
    margin-bottom: 10px;
}}
.copilot-sub {{
    color: var(--muted);
    font-size: 13px;
    line-height: 1.85;
    text-align: center;
    max-width: 420px;
    margin: 0 auto 16px auto;
}}
.copilot-hint {{
    color: var(--muted);
    font-size: 12px;
    line-height: 1.85;
    max-width: 460px;
    margin: 12px auto 0 auto;
}}
.copilot-divider {{
    height: 1px;
    background: {cfg["chat_line"]};
    margin: 10px 0 10px 0;
}}
.copilot-shell [data-testid="stFormSubmitButton"] button {{
    border-radius: 18px !important;
    height: 44px !important;
}}
.copilot-shell .stTextArea textarea {{
    border-radius: 24px !important;
    padding: 14px 16px !important;
}}
</style>
""",
        unsafe_allow_html=True,
    )


def get_theme_cfg() -> Dict[str, str]:
    return THEME_CONFIG.get(st.session_state.theme_mode, THEME_CONFIG["浅色"])


def style_plotly(fig):
    cfg = get_theme_cfg()
    fig.update_layout(
        template="plotly_dark" if st.session_state.theme_mode == "深色" else "plotly_white",
        paper_bgcolor=cfg["card"],
        plot_bgcolor=cfg["card"],
        font=dict(color=cfg["text"]),
        title=dict(font=dict(color=cfg["text"], size=18)),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(color=cfg["text"])),
        margin=dict(l=22, r=18, t=56, b=28),
        hoverlabel=dict(font=dict(color=cfg["text"])),
    )
    fig.update_xaxes(
        gridcolor=cfg["grid"],
        linecolor=cfg["line"],
        zerolinecolor=cfg["grid"],
        tickfont=dict(color=cfg["text"]),
        title_font=dict(color=cfg["text"]),
    )
    fig.update_yaxes(
        gridcolor=cfg["grid"],
        linecolor=cfg["line"],
        zerolinecolor=cfg["grid"],
        tickfont=dict(color=cfg["text"]),
        title_font=dict(color=cfg["text"]),
    )
    return fig


# =========================
# 通用工具函数
# =========================
def queue_prompt(text: str) -> None:
    st.session_state.pending_prompt = text
    st.rerun()


def make_arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    safe_df.columns = [str(c) for c in safe_df.columns]
    safe_df = safe_df.loc[:, ~pd.Index(safe_df.columns).duplicated()]

    for col in safe_df.columns:
        if safe_df[col].dtype == "object":
            def _convert(v):
                try:
                    if pd.isna(v):
                        return None
                except Exception:
                    pass
                if isinstance(v, (dict, list, tuple, set)):
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                return str(v)
            safe_df[col] = safe_df[col].map(_convert)

    return safe_df


def safe_dataframe(df: pd.DataFrame, **kwargs) -> None:
    st.dataframe(make_arrow_safe(df), **kwargs)


def find_local_db_path() -> Optional[str]:
    for path in DEFAULT_DB_PATHS:
        if os.path.exists(path):
            return path
    return None


def choose_final_sheet(sheet_names: List[str]) -> str:
    if not sheet_names:
        return ""
    priorities = ["最终版", "最终", "final", "Final", "FINAL"]
    for priority in priorities:
        for sheet in sheet_names:
            if priority in sheet:
                return sheet
    return sheet_names[-1]


def get_excel_sheet_names_from_bytes(file_bytes: bytes) -> List[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def get_excel_sheet_names_from_path(path: str) -> List[str]:
    return pd.ExcelFile(path).sheet_names


def _normalize_column_name(name: Any) -> str:
    """把 Excel 表头统一成程序内部字段名，避免空格、换行、中文表头导致字段识别失败。"""
    raw = "" if name is None else str(name)
    raw = raw.replace("\ufeff", "").replace("\u3000", " ").strip()
    raw = re.sub(r"[\r\n\t]+", "_", raw)
    raw = re.sub(r"\s+", "_", raw)
    raw = raw.replace("-", "_")
    raw = re.sub(r"_+", "_", raw).strip("_")

    key = raw.lower()
    key_compact = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", key)

    aliases = {
        "refid": "ref_id",
        "编号": "ref_id",
        "文献编号": "ref_id",
        "处方名称": "formulation_name",
        "配方名称": "formulation_name",
        "formulationname": "formulation_name",
        "phos1type": "phos_1_type",
        "phos_1_type": "phos_1_type",
        "phospholipid1type": "phos_1_type",
        "主磷脂类型": "phos_1_type",
        "磷脂1类型": "phos_1_type",
        "磷脂类型": "phos_1_type",
        "phos1ratio": "phos_1_ratio",
        "phos_1_ratio": "phos_1_ratio",
        "主磷脂比例": "phos_1_ratio",
        "磷脂1比例": "phos_1_ratio",
        "phos2type": "phos_2_type",
        "phos_2_type": "phos_2_type",
        "第二磷脂类型": "phos_2_type",
        "phos2ratio": "phos_2_ratio",
        "phos_2_ratio": "phos_2_ratio",
        "第二磷脂比例": "phos_2_ratio",
        "cholratio": "chol_ratio",
        "cholesterolratio": "chol_ratio",
        "胆固醇比例": "chol_ratio",
        "apotype": "apo_type",
        "apo_type": "apo_type",
        "载脂蛋白类型": "apo_type",
        "载脂蛋白": "apo_type",
        "aporatio": "apo_ratio",
        "apo_ratio": "apo_ratio",
        "载脂蛋白比例": "apo_ratio",
        "methodassembly": "method_assembly",
        "method_assembly": "method_assembly",
        "组装方法": "method_assembly",
        "shapeobserved": "Shape_Observed",
        "shape_observed": "Shape_Observed",
        "形貌": "Shape_Observed",
        "观察形貌": "Shape_Observed",
        "sizemeannm": "Size_Mean_nm",
        "size_mean_nm": "Size_Mean_nm",
        "平均粒径nm": "Size_Mean_nm",
        "粒径nm": "Size_Mean_nm",
        "粒径": "Size_Mean_nm",
        "pdi": "PDI",
        "zetamv": "Zeta_mV",
        "zeta_mv": "Zeta_mV",
        "zeta电位": "Zeta_mV",
        "eepercent": "EE_Percent",
        "ee_percent": "EE_Percent",
        "ee": "EE_Percent",
        "ee%": "EE_Percent",
        "包封率": "EE_Percent",
        "包封率%": "EE_Percent",
        "dlpercent": "DL_Percent",
        "dl_percent": "DL_Percent",
        "载药量": "DL_Percent",
        "载药量%": "DL_Percent",
        "indication": "Indication",
        "适应症": "Indication",
        "应用方向": "Indication",
    }

    if key in aliases:
        return aliases[key]
    if key_compact in aliases:
        return aliases[key_compact]
    return raw


def _deduplicate_columns(columns: List[str]) -> List[str]:
    counts: Dict[str, int] = {}
    out: List[str] = []
    for col in columns:
        base = str(col).strip() or "column"
        if base not in counts:
            counts[base] = 0
            out.append(base)
        else:
            counts[base] += 1
            out.append(f"{base}_{counts[base]}")
    return out


def clean_categorical_series(series: pd.Series) -> pd.Series:
    """清理分类字段，过滤 none/null/空字符串，避免统计和图表被脏值干扰。"""
    s = series.copy()
    s = s.where(pd.notnull(s), None)
    s = s.astype(str).str.strip()
    s = s.replace({
        "": None,
        "nan": None,
        "NaN": None,
        "None": None,
        "none": None,
        "NULL": None,
        "null": None,
        "NA": None,
        "N/A": None,
        "na": None,
        "无": None,
        "无具体信息": None,
        "0": None,
        "0.0": None,
    })
    return s.dropna()


def nonempty_nunique(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    return int(clean_categorical_series(df[col]).nunique())


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 如果 Excel 被错误读取成“第一行是字段名、真正表头在数据行”，自动重新识别表头。
    expected = {"phos_1_type", "apo_type", "Size_Mean_nm", "PDI", "EE_Percent"}
    normalized_cols = [_normalize_column_name(c) for c in df.columns]
    if len(expected.intersection(set(normalized_cols))) < 2 and len(df) > 0:
        max_scan_rows = min(8, len(df))
        for i in range(max_scan_rows):
            candidate = [_normalize_column_name(v) for v in df.iloc[i].tolist()]
            if len(expected.intersection(set(candidate))) >= 2:
                df = df.iloc[i + 1:].reset_index(drop=True)
                df.columns = candidate
                break

    df.columns = _deduplicate_columns([_normalize_column_name(c) for c in df.columns])
    df = df.loc[:, ~pd.Index(df.columns).astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({
                "": None,
                "nan": None,
                "NaN": None,
                "None": None,
                "none": None,
                "NULL": None,
                "null": None,
                "NA": None,
                "N/A": None,
            })

    numeric_hint_cols = []
    for col in df.columns:
        lc = str(col).lower()
        if any(k in lc for k in ["ratio", "percent", "size", "pdi", "zeta", "ph_", "ph", "time", "temp", "ee", "dl"]):
            numeric_hint_cols.append(col)

    for col in set(numeric_hint_cols):
        try:
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("−", "-", regex=False)
                    .str.replace("–", "-", regex=False)
                    .str.replace("%", "", regex=False)
                    .str.extract(r"([-+]?\d*\.?\d+)", expand=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass

    return df


def load_dataframe_from_bytes(file_bytes: bytes, file_name: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if file_name.lower().endswith(".csv"):
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)
    return clean_dataframe(df)


def load_dataframe_from_path(path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name)
    return clean_dataframe(df)


def get_active_df() -> Optional[pd.DataFrame]:
    df = st.session_state.get("active_df")
    return df if isinstance(df, pd.DataFrame) else None


def decode_uploaded_text(file) -> str:
    raw = file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "gbk", "gb2312"]:
        try:
            return raw.decode(enc)
        except Exception:
            continue
    return raw.decode("utf-8", errors="ignore")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        line = re.sub(r"^\s*#+\s*", "", line)
        line = re.sub(r"^\s*[-*+]\s*", "", line)
        line = re.sub(r"^\s*\d+\.\s*", "", line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def extract_terms(text: str) -> List[str]:
    text = text.lower().strip()
    zh_terms = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    en_terms = re.findall(r"[a-zA-Z0-9_]+", text)
    out = []
    seen = set()
    for term in zh_terms + en_terms:
        if term and term not in seen:
            out.append(term)
            seen.add(term)
    return out


def attach_record_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index(drop=False).rename(columns={"index": "_row_id"}).copy()
    labels = []
    for _, row in out.iterrows():
        if "formulation_name" in out.columns and pd.notna(row.get("formulation_name")):
            base = str(row.get("formulation_name"))
        elif "ref_id" in out.columns and pd.notna(row.get("ref_id")):
            base = f"ref_{row.get('ref_id')}"
        elif "apo_type" in out.columns and pd.notna(row.get("apo_type")):
            base = f"apo_{row.get('apo_type')}"
        else:
            base = f"record_{row.get('_row_id')}"
        labels.append(f"{base} | row {row.get('_row_id')}")
    out["_record_label"] = labels
    return out


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
        out = out[out["phos_1_type"].astype(str) == str(phos_1_type)]
    if apo_type and "apo_type" in out.columns:
        out = out[out["apo_type"].astype(str) == str(apo_type)]
    if indication and "Indication" in out.columns:
        out = out[out["Indication"].astype(str) == str(indication)]
    if method_assembly and "method_assembly" in out.columns:
        out = out[out["method_assembly"].astype(str) == str(method_assembly)]
    if shape_observed and "Shape_Observed" in out.columns:
        out = out[out["Shape_Observed"].astype(str) == str(shape_observed)]
    if max_size_nm is not None and "Size_Mean_nm" in out.columns:
        out = out[out["Size_Mean_nm"].fillna(1e9) <= max_size_nm]
    if max_pdi is not None and "PDI" in out.columns:
        out = out[out["PDI"].fillna(1e9) <= max_pdi]
    if min_ee_percent is not None and "EE_Percent" in out.columns:
        out = out[out["EE_Percent"].fillna(-1e9) >= min_ee_percent]

    return out


def render_capture_button() -> None:
    components.html(
        """
        <div style="margin:0;padding:0;">
          <button id="smu_capture_btn"
            style="
                width:100%;
                height:40px;
                border-radius:12px;
                border:1px solid #d0d7e2;
                background:#ffffff;
                cursor:pointer;
                font-weight:700;
            ">
            📸 截屏下载当前页面
          </button>
        </div>

        <script>
        (function() {
            const btn = document.getElementById("smu_capture_btn");

            function ensureLib(callback){
                const parentDoc = window.parent.document;
                if (window.parent.html2canvas) {
                    callback();
                    return;
                }
                let oldScript = parentDoc.getElementById("smu_html2canvas_lib");
                if (oldScript) {
                    oldScript.addEventListener("load", callback);
                    return;
                }
                let script = parentDoc.createElement("script");
                script.id = "smu_html2canvas_lib";
                script.src = "https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js";
                script.onload = callback;
                parentDoc.head.appendChild(script);
            }

            btn.onclick = function() {
                ensureLib(async function() {
                    try {
                        const parentDoc = window.parent.document;
                        const target = parentDoc.querySelector(".stApp");
                        if (!target) {
                            alert("未找到页面区域，无法截屏。");
                            return;
                        }
                        const canvas = await window.parent.html2canvas(target, {
                            useCORS: true,
                            backgroundColor: null,
                            scale: 2
                        });

                        const link = parentDoc.createElement("a");
                        const now = new Date();
                        const stamp = now.getFullYear().toString() +
                                      String(now.getMonth()+1).padStart(2,"0") +
                                      String(now.getDate()).padStart(2,"0") + "_" +
                                      String(now.getHours()).padStart(2,"0") +
                                      String(now.getMinutes()).padStart(2,"0") +
                                      String(now.getSeconds()).padStart(2,"0");
                        link.download = "smu_agent_capture_" + stamp + ".png";
                        link.href = canvas.toDataURL("image/png");
                        link.click();
                    } catch (e) {
                        alert("截屏失败，可能是当前部署环境限制。");
                    }
                });
            };
        })();
        </script>
        """,
        height=52,
    )


def auto_scroll_chat() -> None:
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            function scrollToBottom() {
                const anchor = doc.getElementById("smu-chat-bottom-anchor");
                if (!anchor) return;
                try {
                    anchor.scrollIntoView({behavior: "smooth", block: "end"});
                } catch (e) {
                    anchor.scrollIntoView();
                }
            }
            setTimeout(scrollToBottom, 120);
            setTimeout(scrollToBottom, 350);
            setTimeout(scrollToBottom, 700);
        })();
        </script>
        """,
        height=0,
    )


# =========================
# Demo 公式 / 工具逻辑
# =========================
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
    pdi = 0.26 + 0.015 * abs(protein_ratio - 1.3) + 0.005 * abs(lipid_ratio - 4.8) - 0.0015 * time_min

    ee = max(20.0, min(95.0, ee))
    size = max(45.0, min(300.0, size))
    pdi = max(0.05, min(0.60, pdi))

    return {
        "predicted_ee": round(ee, 2),
        "predicted_size_nm": round(size, 2),
        "predicted_pdi": round(pdi, 3),
    }


def search_knowledge_impl(question: str, top_k: int = 5) -> Dict[str, Any]:
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
        if any(k in text for k in ["沉淀", "包封率", "温度", "粒径", "pdi", "zeta", "脂质", "蛋白"]):
            score += 1
        if score > 0:
            scored.append(
                {
                    "chunk_id": f"local_{idx + 1}",
                    "score": score,
                    "text": line,
                    "source": "本地知识库",
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"hits": scored[:top_k]}


def predict_formulation_impl(lipid_ratio: float, protein_ratio: float, temperature: float, time_min: float) -> Dict[str, Any]:
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
        "说明": "当前为演示公式，后续可替换为真实 sklearn/xgboost 模型。",
    }


def reverse_design_impl(target_ee_min: float = 80.0, target_size_max: float = 120.0, top_k: int = 5) -> Dict[str, Any]:
    candidates = []

    for _ in range(500):
        params = {
            "lipid_ratio": round(random.uniform(1.0, 8.0), 2),
            "protein_ratio": round(random.uniform(0.4, 2.5), 2),
            "temperature": round(random.uniform(20.0, 45.0), 1),
            "time_min": round(random.uniform(5.0, 40.0), 1),
        }

        pred = _toy_formula(**params)

        if pred["predicted_ee"] >= target_ee_min and pred["predicted_size_nm"] <= target_size_max:
            score = pred["predicted_ee"] - 0.06 * pred["predicted_size_nm"] - 18.0 * pred["predicted_pdi"]
            candidates.append(
                {
                    "脂质比例": params["lipid_ratio"],
                    "蛋白比例": params["protein_ratio"],
                    "温度": params["temperature"],
                    "时间(min)": params["time_min"],
                    "预测包封率(%)": pred["predicted_ee"],
                    "预测粒径(nm)": pred["predicted_size_nm"],
                    "预测PDI": pred["predicted_pdi"],
                    "综合评分": round(score, 3),
                }
            )

    candidates.sort(key=lambda x: x["综合评分"], reverse=True)
    return {
        "目标最低包封率(%)": target_ee_min,
        "目标粒径上限(nm)": target_size_max,
        "candidates": candidates[:top_k],
        "说明": "当前为随机采样 + 演示评分，后续可替换为真实优化算法。",
    }


def query_formulation_database_impl(
    phos_1_type=None,
    apo_type=None,
    indication=None,
    method_assembly=None,
    shape_observed=None,
    max_size_nm=None,
    max_pdi=None,
    min_ee_percent=None,
    top_k: int = 10,
) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。请先在左侧上传 Excel/CSV 数据表。"}

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

    if "EE_Percent" in filtered.columns:
        filtered = filtered.sort_values(by="EE_Percent", ascending=False, na_position="last")

    preferred_cols = [
        "ref_id", "formulation_name", "phos_1_type", "phos_1_ratio", "chol_ratio",
        "apo_type", "apo_ratio", "method_assembly", "Shape_Observed", "Size_Mean_nm",
        "PDI", "Zeta_mV", "EE_Percent", "DL_Percent", "Indication",
    ]
    cols = [c for c in preferred_cols if c in filtered.columns] or list(filtered.columns[:12])
    preview_df = filtered[cols].head(top_k).copy()

    return {
        "matched_count": int(len(filtered)),
        "preview_count": int(len(preview_df)),
        "records": preview_df.where(pd.notnull(preview_df), None).to_dict(orient="records"),
    }


def aggregate_formulation_database_impl(group_by: str, metric: str, agg: str = "mean", top_k: int = 10) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。"}
    if group_by not in df.columns:
        return {"error": f"group_by 字段不存在：{group_by}"}
    if metric not in df.columns:
        return {"error": f"metric 字段不存在：{metric}"}

    work = df[[group_by, metric]].copy().dropna()
    if work.empty:
        return {"error": "可用于聚合的数据为空。"}

    if agg not in ["mean", "median", "max", "min", "count"]:
        agg = "mean"

    if agg == "count":
        result = work.groupby(group_by)[metric].count().reset_index(name=f"{metric}_{agg}")
    else:
        result = work.groupby(group_by)[metric].agg(agg).reset_index(name=f"{metric}_{agg}")

    result = result.sort_values(by=f"{metric}_{agg}", ascending=False).head(top_k)
    return {
        "group_by": group_by,
        "metric": metric,
        "agg": agg,
        "table": result.where(pd.notnull(result), None).to_dict(orient="records"),
    }


def explain_database_field_impl(field_name: str) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。"}
    if field_name not in df.columns:
        return {"error": f"字段不存在：{field_name}"}

    series = df[field_name]
    explanation = FIELD_EXPLANATIONS.get(
        field_name,
        "这是当前数据库中的一个字段。你可以结合样本筛选、统计分析和可视化进一步理解它的意义。",
    )
    result = {
        "field_name": field_name,
        "explanation": explanation,
        "dtype": str(series.dtype),
        "non_null_count": int(series.notna().sum()),
    }

    if pd.api.types.is_numeric_dtype(series):
        s = series.dropna()
        result["min"] = None if s.empty else float(s.min())
        result["max"] = None if s.empty else float(s.max())
        result["mean"] = None if s.empty else float(s.mean())
    else:
        result["top_values"] = series.dropna().astype(str).value_counts().head(10).to_dict()

    return result


TOOL_IMPL = {
    "search_knowledge": search_knowledge_impl,
    "predict_formulation": predict_formulation_impl,
    "reverse_design": reverse_design_impl,
    "query_formulation_database": query_formulation_database_impl,
    "aggregate_formulation_database": aggregate_formulation_database_impl,
    "explain_database_field": explain_database_field_impl,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "检索本地知识库中的实验经验、问题诊断说明和知识片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_formulation",
            "description": "根据输入参数预测包封率、粒径和PDI。",
            "parameters": {
                "type": "object",
                "properties": {
                    "lipid_ratio": {"type": "number"},
                    "protein_ratio": {"type": "number"},
                    "temperature": {"type": "number"},
                    "time_min": {"type": "number"},
                },
                "required": ["lipid_ratio", "protein_ratio", "temperature", "time_min"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reverse_design",
            "description": "根据目标包封率和目标粒径上限，逆向给出若干推荐参数组合。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_ee_min": {"type": "number", "default": 80.0},
                    "target_size_max": {"type": "number", "default": 120.0},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_formulation_database",
            "description": "根据条件筛选处方数据库。",
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
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aggregate_formulation_database",
            "description": "对数据库按某字段分组，并统计某个指标的聚合值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {"type": "string"},
                    "metric": {"type": "string"},
                    "agg": {"type": "string", "default": "mean"},
                    "top_k": {"type": "integer", "default": 10},
                },
                "required": ["group_by", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_database_field",
            "description": "解释一个数据库字段的含义。",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string"},
                },
                "required": ["field_name"],
            },
        },
    },
]

BASE_SYSTEM_PROMPT = """
你是一个中文科研智能体助手，名字叫 纳米制剂开发助手。
你服务于一个中药药剂学实验平台。

规则：
1. 用户涉及知识检索、经验总结、故障诊断时，优先调用 search_knowledge。
2. 用户提到预测参数表现时，优先调用 predict_formulation。
3. 用户提到逆向推荐、目标包封率、目标粒径时，优先调用 reverse_design。
4. 用户提到筛选数据库、寻找样本时，优先调用 query_formulation_database。
5. 用户提到分组统计、平均值、比较时，优先调用 aggregate_formulation_database。
6. 用户提到“字段是什么意思”时，优先调用 explain_database_field。
7. 默认使用中文回答。
8. 如果结果来自演示公式，一定要明确说明它是 demo。
9. 输出尽量简洁清晰，有条理。
"""


def build_system_prompt() -> str:
    prompt = BASE_SYSTEM_PROMPT
    df = get_active_df()
    if df is not None and not df.empty:
        prompt += "\n当前已加载结构化数据库。"
        prompt += f"\n当前数据库记录数：{len(df)}。"
        prompt += f"\n可用字段：{', '.join(list(df.columns)[:50])}。"
    return prompt


def run_agent(user_text: str) -> Dict[str, Any]:
    if client is None:
        return {
            "answer": "当前没有可用的大模型连接。请检查 API_KEY / API_BASE / MODEL 配置。",
            "tool_logs": [],
        }

    tool_logs: List[Dict[str, Any]] = []
    messages = [{"role": "system", "content": build_system_prompt()}]

    for msg in st.session_state.messages[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_text})

    try:
        for _ in range(6):
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            msg = response.choices[0].message

            if not getattr(msg, "tool_calls", None):
                return {"answer": msg.content or "", "tool_logs": tool_logs}

            assistant_message = {"role": "assistant", "content": msg.content or "", "tool_calls": []}
            tool_results_by_id: Dict[str, Dict[str, Any]] = {}

            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                raw_args = tool_call.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except Exception:
                    args = {}

                if tool_name not in TOOL_IMPL:
                    result = {"error": f"未知工具：{tool_name}"}
                else:
                    try:
                        result = TOOL_IMPL[tool_name](**args)
                    except Exception as e:
                        result = {"error": f"{tool_name} 执行失败：{str(e)}"}

                tool_logs.append({"tool": tool_name, "arguments": args, "result": result})
                tool_results_by_id[tool_call.id] = result

                assistant_message["tool_calls"].append(
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(args, ensure_ascii=False),
                        },
                    }
                )

            messages.append(assistant_message)

            for tool_call in msg.tool_calls:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(tool_results_by_id.get(tool_call.id, {}), ensure_ascii=False),
                    }
                )

        return {"answer": "工具调用达到上限，本轮停止。", "tool_logs": tool_logs}
    except Exception as e:
        return {"answer": f"调用大模型失败：{str(e)}", "tool_logs": tool_logs}


# =========================
# 渲染函数
# =========================
def render_chat_message(role: str, content: str) -> None:
    avatar = "🧑‍🔬" if role == "user" else "🤖"
    with st.chat_message(role, avatar=avatar):
        st.markdown(content)


def render_tool_result(tool_name: str, result: Dict[str, Any]) -> None:
    if tool_name == "search_knowledge":
        hits = result.get("hits", [])
        if hits:
            st.markdown("**本地知识命中**")
            for hit in hits:
                st.markdown(
                    f"""
                    <div style="background:rgba(255,255,255,0.03);border:1px solid var(--line);border-radius:14px;padding:10px 12px;margin-top:8px;">
                        <div style="font-weight:700;font-size:12px;margin-bottom:4px;">{hit.get('chunk_id','-')} · 相关度 {hit.get('score','-')}</div>
                        <div style="font-size:13px;color:var(--text);line-height:1.7;">{html.escape(str(hit.get('text','')))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    elif tool_name == "predict_formulation":
        st.markdown("**参数预测结果**")
        pred_df = pd.DataFrame(
            [
                {
                    "预测包封率(%)": result.get("预测包封率(%)"),
                    "预测粒径(nm)": result.get("预测粒径(nm)"),
                    "预测PDI": result.get("预测PDI"),
                    "说明": result.get("说明", ""),
                }
            ]
        )
        safe_dataframe(pred_df, use_container_width=True)

    elif tool_name == "reverse_design":
        st.markdown(
            f"**逆向推荐结果**（目标 EE ≥ {result.get('目标最低包封率(%)','-')}；目标粒径 ≤ {result.get('目标粒径上限(nm)','-')} nm）"
        )
        candidates = result.get("candidates", [])
        if candidates:
            safe_dataframe(pd.DataFrame(candidates), use_container_width=True)
        else:
            st.info("当前条件下没有找到满足要求的候选方案。可以适当放宽目标粒径或目标包封率。")

    elif tool_name == "query_formulation_database":
        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown(f"**数据库筛选结果**：匹配 {result.get('matched_count', 0)} 条，展示 {result.get('preview_count', 0)} 条")
            records = result.get("records", [])
            if records:
                safe_dataframe(pd.DataFrame(records), use_container_width=True)

    elif tool_name == "aggregate_formulation_database":
        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown(f"**分组统计结果**：{result.get('group_by','-')} / {result.get('metric','-')} / {result.get('agg','-')}")
            table = result.get("table", [])
            if table:
                safe_dataframe(pd.DataFrame(table), use_container_width=True)

    elif tool_name == "explain_database_field":
        if "error" in result:
            st.error(result["error"])
        else:
            st.markdown(f"**字段说明：{result.get('field_name','-')}**")
            st.write(result.get("explanation", ""))
            aux = {k: v for k, v in result.items() if k not in ["field_name", "explanation"]}
            st.json(aux)


# =========================
# 页面模块
# =========================
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="font-size:22px;font-weight:900;margin-bottom:4px;">纳米制剂开发助手</div>
            <div style="font-size:13px;color:var(--muted);line-height:1.7;margin-bottom:12px;">
            面向中药药剂学实验场景的智能体平台原型
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🎨 主题设置</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-desc">可在浅色与深色主题之间自由切换。</div>', unsafe_allow_html=True)
        theme_value = st.radio(
            "选择主题",
            ["浅色", "深色"],
            index=0 if st.session_state.theme_mode == "浅色" else 1,
            horizontal=True,
            key="sidebar_theme_radio",
            label_visibility="collapsed",
        )
        if theme_value != st.session_state.theme_mode:
            st.session_state.theme_mode = theme_value
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📁 项目区</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-desc">管理当前项目、加载数据库、查看连接状态。</div>', unsafe_allow_html=True)

        st.session_state.project_name = st.text_input(
            "项目名称",
            value=st.session_state.project_name,
            key="sidebar_project_name_input",
        )

        project_type_options = ["处方开发", "知识问答", "数据库分析", "实验设计", "综合智能体"]
        st.session_state.project_type = st.selectbox(
            "项目类型",
            project_type_options,
            index=project_type_options.index(st.session_state.project_type)
            if st.session_state.project_type in project_type_options
            else 0,
            key="sidebar_project_type_select",
        )

        st.session_state.project_desc = st.text_area(
            "项目简介",
            value=st.session_state.project_desc,
            height=90,
            key="sidebar_project_desc_textarea",
        )

        st.markdown("---")

        db_file = st.file_uploader(
            "上传 Excel / CSV 数据表",
            type=["xlsx", "csv"],
            accept_multiple_files=False,
            key="sidebar_db_file_uploader",
        )
        if db_file is not None:
            st.session_state.db_file_bytes = db_file.getvalue()
            st.session_state.db_file_name = db_file.name

        if st.button("清除已上传数据库", use_container_width=True, key="sidebar_clear_uploaded_db_btn"):
            st.session_state.db_file_bytes = None
            st.session_state.db_file_name = None
            st.session_state.db_sheet_name = None
            st.rerun()

        if st.session_state.db_sheet_name:
            st.caption(f"当前工作表：{st.session_state.db_sheet_name}")

        if client is not None:
            st.success("大模型连接：已配置")
        else:
            st.warning("大模型连接：未配置")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">📚 知识库区</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-desc">可上传 txt / md 片段，追加到本地知识库。</div>', unsafe_allow_html=True)

        kb_files = st.file_uploader(
            "上传 txt / md 文件",
            type=["txt", "md"],
            accept_multiple_files=True,
            key="sidebar_kb_file_uploader",
        )

        if st.button("导入知识片段", use_container_width=True, key="sidebar_import_kb_btn"):
            added_lines = []
            if kb_files:
                for f in kb_files:
                    text = decode_uploaded_text(f)
                    text = normalize_text(text)
                    if text:
                        added_lines.append(text)

            if added_lines:
                current = st.session_state.knowledge_text.strip()
                merged = current + "\n" + "\n".join(added_lines) if current else "\n".join(added_lines)
                st.session_state.knowledge_text = merged.strip()
                st.success("知识片段导入成功。")
            else:
                st.info("没有可导入的 txt / md 内容。")

        st.session_state.knowledge_text = st.text_area(
            "知识库内容",
            value=st.session_state.knowledge_text,
            height=180,
            key="sidebar_knowledge_textarea",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="sidebar-card">', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-title">🧪 测试区</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-desc">快速向右侧聊天面板发送测试问题。</div>', unsafe_allow_html=True)

        if st.button("测试：沉淀原因", use_container_width=True, key="sidebar_test_precipitation_btn"):
            queue_prompt("为什么超声后出现白色沉淀？")
        if st.button("测试：参数预测", use_container_width=True, key="sidebar_test_predict_btn"):
            queue_prompt("预测一下：lipid_ratio=4.5, protein_ratio=1.2, temperature=37, time_min=20")
        if st.button("测试：高包封率推荐", use_container_width=True, key="sidebar_test_reverse_btn"):
            queue_prompt("帮我设计几组包封率大于80且粒径尽量小的参数")
        if st.button("测试：字段解释", use_container_width=True, key="sidebar_test_explain_btn"):
            queue_prompt("解释一下 EE_Percent 这个字段")
        if st.button("清空对话记录", use_container_width=True, key="sidebar_clear_chat_btn"):
            st.session_state.messages = []
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)


def load_active_database() -> Tuple[Optional[pd.DataFrame], str]:
    df_main = None
    current_db_name = "未加载"

    try:
        if st.session_state.db_file_bytes is not None and st.session_state.db_file_name:
            db_name = st.session_state.db_file_name
            current_db_name = db_name

            if db_name.lower().endswith(".xlsx"):
                sheet_names = get_excel_sheet_names_from_bytes(st.session_state.db_file_bytes)
                chosen_sheet = choose_final_sheet(sheet_names)
                st.session_state.db_sheet_name = chosen_sheet
                df_main = load_dataframe_from_bytes(st.session_state.db_file_bytes, st.session_state.db_file_name, sheet_name=chosen_sheet)
            else:
                st.session_state.db_sheet_name = "最终版"
                df_main = load_dataframe_from_bytes(st.session_state.db_file_bytes, st.session_state.db_file_name)
        else:
            local_path = find_local_db_path()
            if local_path:
                current_db_name = os.path.basename(local_path)
                if local_path.lower().endswith(".xlsx"):
                    sheet_names = get_excel_sheet_names_from_path(local_path)
                    chosen_sheet = choose_final_sheet(sheet_names)
                    st.session_state.db_sheet_name = chosen_sheet
                    df_main = load_dataframe_from_path(local_path, sheet_name=chosen_sheet)
                else:
                    st.session_state.db_sheet_name = "最终版"
                    df_main = load_dataframe_from_path(local_path)
    except Exception as e:
        st.warning(f"数据库加载失败：{e}")
        df_main = None

    st.session_state.active_df = df_main
    st.session_state.active_db_name = current_db_name
    return df_main, current_db_name


def render_header(df_main: Optional[pd.DataFrame]) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-title">{html.escape(st.session_state.project_name)}</div>
            <div class="hero-sub">
                当前项目类型：<b>{html.escape(st.session_state.project_type)}</b><br>
                {html.escape(st.session_state.project_desc)}<br>
                当前数据库：<b>{html.escape(st.session_state.active_db_name)}</b>
                {" · 工作表：" + html.escape(st.session_state.db_sheet_name) if st.session_state.db_sheet_name else ""}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="quick-card"><div class="soft-title">⚡ 快捷操作</div></div>', unsafe_allow_html=True)
    q1, q2, q3, q4, q5 = st.columns(5)
    if q1.button("沉淀分析", use_container_width=True, key="main_quick_precipitation_btn"):
        queue_prompt("为什么超声后出现白色沉淀？")
    if q2.button("筛选处方", use_container_width=True, key="main_quick_filter_btn"):
        queue_prompt("帮我筛选 apo_type=22A 且 Size_Mean_nm 小于 100 的处方")
    if q3.button("平均粒径", use_container_width=True, key="main_quick_aggregate_btn"):
        queue_prompt("按 apo_type 统计平均粒径")
    if q4.button("高包封率", use_container_width=True, key="main_quick_reverse_btn"):
        queue_prompt("帮我设计几组包封率大于80且粒径尽量小的参数")
    if q5.button("解释字段", use_container_width=True, key="main_quick_explain_btn"):
        queue_prompt("解释一下 PDI 这个字段")

    st.markdown('<div class="soft-card"><div class="soft-title">📊 项目总览</div></div>', unsafe_allow_html=True)
    if df_main is None or df_main.empty:
        st.info("当前还没有可用的结构化数据库。你可以把 Excel 放进 GitHub 仓库的 data/ 目录，或者直接在左侧上传。")
    else:
        metric_cols = st.columns(6)
        metric_items = [
            ("记录数", len(df_main)),
            ("字段数", len(df_main.columns)),
            ("有效粒径数", int(df_main["Size_Mean_nm"].notna().sum()) if "Size_Mean_nm" in df_main.columns else 0),
            ("有效EE数", int(df_main["EE_Percent"].notna().sum()) if "EE_Percent" in df_main.columns else 0),
            ("磷脂类型数", nonempty_nunique(df_main, "phos_1_type")),
            ("Apo类型数", nonempty_nunique(df_main, "apo_type")),
        ]
        for col, (label, value) in zip(metric_cols, metric_items):
            col.metric(label, value)


def render_overview_tab(df_main: Optional[pd.DataFrame]) -> None:
    if df_main is None or df_main.empty:
        st.info("请先加载数据库。")
        return

    r1, r2 = st.columns(2)
    if "phos_1_type" in df_main.columns:
        top_phos = clean_categorical_series(df_main["phos_1_type"]).value_counts().head(10).reset_index()
        top_phos.columns = ["phos_1_type", "count"]
        fig = px.bar(top_phos, x="phos_1_type", y="count", title="Top 10 磷脂类型分布")
        r1.plotly_chart(style_plotly(fig), use_container_width=True)

    if "apo_type" in df_main.columns:
        top_apo = clean_categorical_series(df_main["apo_type"]).value_counts().head(10).reset_index()
        top_apo.columns = ["apo_type", "count"]
        fig = px.bar(top_apo, x="apo_type", y="count", title="Top 10 Apo 类型分布")
        r2.plotly_chart(style_plotly(fig), use_container_width=True)

    r3, r4 = st.columns(2)
    if "Size_Mean_nm" in df_main.columns and df_main["Size_Mean_nm"].notna().sum() > 0:
        fig = px.histogram(df_main.dropna(subset=["Size_Mean_nm"]), x="Size_Mean_nm", nbins=30, title="粒径分布")
        r3.plotly_chart(style_plotly(fig), use_container_width=True)

    if "EE_Percent" in df_main.columns and df_main["EE_Percent"].notna().sum() > 0:
        fig = px.histogram(df_main.dropna(subset=["EE_Percent"]), x="EE_Percent", nbins=30, title="包封率分布")
        r4.plotly_chart(style_plotly(fig), use_container_width=True)

    if all(col in df_main.columns for col in ["Size_Mean_nm", "PDI"]):
        plot_df = df_main.dropna(subset=["Size_Mean_nm", "PDI"]).copy()
        if not plot_df.empty:
            color_col = "apo_type" if "apo_type" in plot_df.columns else None
            fig = px.scatter(
                plot_df.head(500),
                x="Size_Mean_nm",
                y="PDI",
                color=color_col,
                hover_data=[c for c in ["phos_1_type", "EE_Percent", "method_assembly"] if c in plot_df.columns],
                title="粒径 - PDI 散点图",
            )
            st.plotly_chart(style_plotly(fig), use_container_width=True)


def render_datawork_tab(df_main: Optional[pd.DataFrame]) -> None:
    if df_main is None or df_main.empty:
        st.info("请先加载数据库。")
        return

    st.markdown("#### 条件筛选")
    phos_options = ["全部"] + sorted(clean_categorical_series(df_main["phos_1_type"]).unique().tolist()) if "phos_1_type" in df_main.columns else ["全部"]
    apo_options = ["全部"] + sorted(clean_categorical_series(df_main["apo_type"]).unique().tolist()) if "apo_type" in df_main.columns else ["全部"]
    ind_options = ["全部"] + sorted(clean_categorical_series(df_main["Indication"]).unique().tolist()) if "Indication" in df_main.columns else ["全部"]
    method_options = ["全部"] + sorted(clean_categorical_series(df_main["method_assembly"]).unique().tolist()) if "method_assembly" in df_main.columns else ["全部"]

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_phos = st.selectbox("phos_1_type", phos_options, key="datawork_phos_select")
        sel_apo = st.selectbox("apo_type", apo_options, key="datawork_apo_select")
    with f2:
        sel_ind = st.selectbox("Indication", ind_options, key="datawork_indication_select")
        sel_method = st.selectbox("method_assembly", method_options, key="datawork_method_select")
    with f3:
        max_size = st.number_input("最大粒径 (nm)", min_value=0.0, value=150.0, step=5.0, key="datawork_max_size_input")
        max_pdi = st.number_input("最大 PDI", min_value=0.0, value=0.30, step=0.01, format="%.2f", key="datawork_max_pdi_input")
    with f4:
        min_ee = st.number_input("最小 EE_Percent", min_value=0.0, value=0.0, step=1.0, key="datawork_min_ee_input")
        show_rows = st.slider("展示行数", 5, 80, 20, 1, key="datawork_show_rows_slider")

    filtered_df = apply_dashboard_filters(
        df_main,
        phos_1_type=None if sel_phos == "全部" else sel_phos,
        apo_type=None if sel_apo == "全部" else sel_apo,
        indication=None if sel_ind == "全部" else sel_ind,
        method_assembly=None if sel_method == "全部" else sel_method,
        max_size_nm=max_size if "Size_Mean_nm" in df_main.columns else None,
        max_pdi=max_pdi if "PDI" in df_main.columns else None,
        min_ee_percent=min_ee if ("EE_Percent" in df_main.columns and min_ee > 0) else None,
    )

    st.success(f"筛选后共有 {len(filtered_df)} 条记录。")

    preferred_cols = [
        "ref_id", "formulation_name", "phos_1_type", "phos_1_ratio", "chol_ratio",
        "apo_type", "apo_ratio", "method_assembly", "Shape_Observed", "Size_Mean_nm",
        "PDI", "Zeta_mV", "EE_Percent", "DL_Percent", "Indication",
    ]
    show_cols = [c for c in preferred_cols if c in filtered_df.columns] or list(filtered_df.columns[:15])
    safe_dataframe(filtered_df[show_cols].head(show_rows), use_container_width=True, height=340)

    if not filtered_df.empty:
        st.markdown("#### 导出权限验证")
        download_pwd = st.text_input(
            "请输入下载密码",
            type="password",
            key="datawork_download_password_input",
            placeholder="输入密码后才能下载当前筛选结果",
        )
        if download_pwd:
            if download_pwd == DOWNLOAD_PASSWORD:
                st.success("密码正确，已解锁下载。")
                st.download_button(
                    "下载当前筛选结果 CSV",
                    data=filtered_df[show_cols].to_csv(index=False).encode("utf-8-sig"),
                    file_name="filtered_formulations.csv",
                    mime="text/csv",
                    key="datawork_download_csv_btn",
                    use_container_width=True,
                )
            else:
                st.error("密码错误，无法下载数据。")


def render_compare_tab(df_main: Optional[pd.DataFrame]) -> None:
    if df_main is None or df_main.empty:
        st.info("请先加载数据库。")
        return

    compare_df = attach_record_labels(df_main)
    options = compare_df["_record_label"].tolist()
    selected_labels = st.multiselect(
        "选择 2~4 个样本进行对比",
        options=options,
        default=options[:2] if len(options) >= 2 else options[:1],
        key="compare_multiselect",
    )
    picked = compare_df[compare_df["_record_label"].isin(selected_labels)].copy()
    if picked.empty:
        st.info("请先选择样本。")
        return

    base_cols = [c for c in ["formulation_name", "phos_1_type", "apo_type", "method_assembly", "Shape_Observed"] if c in picked.columns]
    numeric_cols = [c for c in ["Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent"] if c in picked.columns]
    st.markdown("#### 样本信息对比")
    safe_dataframe(picked[["_record_label"] + base_cols + numeric_cols], use_container_width=True, height=320)

    if numeric_cols:
        metric_to_compare = st.selectbox("选择一个指标绘制对比图", numeric_cols, key="compare_metric_select")
        plot_df = picked[["_record_label", metric_to_compare]].dropna().copy()
        if not plot_df.empty:
            fig = px.bar(plot_df, x="_record_label", y=metric_to_compare, title=f"{metric_to_compare} 样本对比")
            st.plotly_chart(style_plotly(fig), use_container_width=True)


def render_ranking_tab(df_main: Optional[pd.DataFrame]) -> None:
    if df_main is None or df_main.empty:
        st.info("请先加载数据库。")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 指标排名")
        rank_numeric_cols = [c for c in ["Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent"] if c in df_main.columns]
        if rank_numeric_cols:
            rank_metric = st.selectbox("排序指标", rank_numeric_cols, key="rank_metric_select")
            rank_order = st.radio("排序方式", ["降序", "升序"], horizontal=True, key="rank_order_radio")
            rank_n = st.slider("展示 Top N", 5, 30, 10, 1, key="rank_topn_slider")

            rank_df = df_main.copy().dropna(subset=[rank_metric])
            ascending = rank_order == "升序"
            rank_df = rank_df.sort_values(by=rank_metric, ascending=ascending).head(rank_n)
            label_col = "formulation_name" if "formulation_name" in rank_df.columns else ("apo_type" if "apo_type" in rank_df.columns else None)
            if label_col:
                fig = px.bar(rank_df, x=label_col, y=rank_metric, title=f"{rank_metric} 排名")
                st.plotly_chart(style_plotly(fig), use_container_width=True)
            safe_dataframe(rank_df.head(rank_n), use_container_width=True, height=260)

    with c2:
        st.markdown("#### 分组统计")
        group_cols = [c for c in ["phos_1_type", "apo_type", "method_assembly", "Shape_Observed", "Indication"] if c in df_main.columns]
        stat_numeric_cols = [c for c in ["Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent"] if c in df_main.columns]
        if group_cols and stat_numeric_cols:
            group_by = st.selectbox("分组字段", group_cols, key="group_tab_groupby_select")
            metric = st.selectbox("统计指标", stat_numeric_cols, key="group_tab_metric_select")
            agg = st.selectbox("聚合方式", ["mean", "median", "max", "min", "count"], key="group_tab_agg_select")
            agg_result = aggregate_formulation_database_impl(group_by, metric, agg=agg, top_k=20)
            table = agg_result.get("table", [])
            if table:
                agg_df = pd.DataFrame(table)
                value_col = f"{metric}_{agg}"
                fig = px.bar(agg_df, x=group_by, y=value_col, title=f"{group_by} - {metric} ({agg})")
                st.plotly_chart(style_plotly(fig), use_container_width=True)
                safe_dataframe(agg_df, use_container_width=True, height=260)


def render_experiment_tab() -> None:
    st.markdown(
        """
        <div class="section-head">
            <div class="section-title">参数实验设计器</div>
            <div class="section-sub">
                这里目前接的是 demo 预测公式。后续接入真实模型后，可直接升级为实验推荐器与参数优化器。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    top_tool_1, top_tool_2 = st.columns([1.2, 4.8], gap="large")
    with top_tool_1:
        st.markdown(
            """
            <div class="capture-box">
                <div class="capture-title">页面工具</div>
                <div class="capture-desc">可一键截取当前页面，方便汇报或记录实验设计结果。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_capture_button()

    with top_tool_2:
        st.markdown(
            """
            <div class="soft-panel">
                <div class="panel-title">设计思路</div>
                <div class="panel-note">
                    左侧用于设置输入参数，右侧用于设置逆向推荐目标。
                    目标推荐会同时考虑：<b>包封率要高</b>、<b>粒径要小</b>、<b>PDI 要尽可能低</b>。
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    design_left, design_right = st.columns([1.25, 1.0], gap="large")
    with design_left:
        with st.container(border=True):
            st.markdown("#### 输入参数设置")
            lipid_ratio = st.slider("脂质比例 lipid_ratio", 1.0, 8.0, 4.5, 0.1, key="exp_lipid_ratio_slider")
            protein_ratio = st.slider("蛋白比例 protein_ratio", 0.4, 2.5, 1.2, 0.1, key="exp_protein_ratio_slider")
            temperature = st.slider("温度 temperature", 20.0, 45.0, 37.0, 0.5, key="exp_temperature_slider")
            time_min = st.slider("时间 time_min", 5.0, 40.0, 20.0, 1.0, key="exp_time_slider")
            pred = predict_formulation_impl(lipid_ratio, protein_ratio, temperature, time_min)

            m1, m2, m3 = st.columns(3)
            m1.metric("预测包封率(%)", pred["预测包封率(%)"])
            m2.metric("预测粒径(nm)", pred["预测粒径(nm)"])
            m3.metric("预测PDI", pred["预测PDI"])

            if st.button("让 纳米制剂开发助手 分析这组参数", use_container_width=True, key="exp_explain_params_btn"):
                queue_prompt(
                    f"请分析这组参数的表现：lipid_ratio={lipid_ratio}, protein_ratio={protein_ratio}, temperature={temperature}, time_min={time_min}"
                )

        curve_df = pd.DataFrame(
            [
                {"指标": "包封率(%)", "数值": pred["预测包封率(%)"]},
                {"指标": "粒径(nm)", "数值": pred["预测粒径(nm)"]},
                {"指标": "PDI", "数值": pred["预测PDI"]},
            ]
        )
        fig_curve = px.bar(curve_df, x="指标", y="数值", title="当前参数输出概览")
        st.plotly_chart(style_plotly(fig_curve), use_container_width=True)

    with design_right:
        with st.container(border=True):
            st.markdown("#### 当前参数快照")
            design_df = pd.DataFrame(
                [
                    {
                        "脂质比例": lipid_ratio,
                        "蛋白比例": protein_ratio,
                        "温度": temperature,
                        "时间(min)": time_min,
                        "预测包封率(%)": pred["预测包封率(%)"],
                        "预测粒径(nm)": pred["预测粒径(nm)"],
                        "预测PDI": pred["预测PDI"],
                    }
                ]
            )
            safe_dataframe(design_df, use_container_width=True, height=140)

        with st.container(border=True):
            st.markdown("#### 逆向推荐目标")
            target_ee = st.slider("目标包封率阈值", 60.0, 95.0, 80.0, 1.0, key="exp_target_ee_slider")
            target_size = st.slider("目标粒径上限 (nm)", 50.0, 220.0, 120.0, 1.0, key="exp_target_size_slider")
            top_k = st.slider("推荐候选数", 3, 10, 5, 1, key="exp_topk_slider")

            if st.button("生成候选参数方案", use_container_width=True, key="exp_generate_candidates_btn"):
                result = reverse_design_impl(target_ee_min=target_ee, target_size_max=target_size, top_k=top_k)
                render_tool_result("reverse_design", result)


def render_knowledge_tab(df_main: Optional[pd.DataFrame]) -> None:
    c1, c2 = st.columns([1.0, 1.0])
    with c1:
        st.markdown("#### 知识检索")
        q = st.text_input("输入知识问题", value="为什么超声后会出现白色沉淀？", key="knowledge_query_input")
        qa1, qa2 = st.columns(2)
        if qa1.button("检索知识库", use_container_width=True, key="knowledge_search_btn"):
            result = search_knowledge_impl(q, top_k=6)
            render_tool_result("search_knowledge", result)
        if qa2.button("交给 纳米制剂开发助手", use_container_width=True, key="knowledge_send_to_agent_btn"):
            queue_prompt(q)

        st.metric("知识条目数", len([x for x in st.session_state.knowledge_text.splitlines() if x.strip()]))
        st.text_area("知识内容预览", value=st.session_state.knowledge_text, height=280, key="knowledge_preview_textarea")

    with c2:
        st.markdown("#### 字段说明")
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            field_name = st.selectbox("选择一个字段", df_main.columns.tolist(), key="knowledge_field_select")
            result = explain_database_field_impl(field_name)
            render_tool_result("explain_database_field", result)
            if st.button("让 纳米制剂开发助手 解释这个字段", use_container_width=True, key="knowledge_explain_field_btn"):
                queue_prompt(f"请解释一下字段 {field_name} 的含义，并说明它在当前数据库里有什么作用")


def render_main_area(df_main: Optional[pd.DataFrame]) -> None:
    left_col, right_col = st.columns([3.9, 2.9], gap="large")

    with left_col:
        render_header(df_main)
        tabs = st.tabs(
            [
                "📌 总览看板",
                "🗂️ 数据工作台",
                "🆚 样本对比",
                "🏆 排名与分组",
                "🧪 实验设计器",
                "📚 知识与字段工作台",
            ]
        )
        with tabs[0]:
            render_overview_tab(df_main)
        with tabs[1]:
            render_datawork_tab(df_main)
        with tabs[2]:
            render_compare_tab(df_main)
        with tabs[3]:
            render_ranking_tab(df_main)
        with tabs[4]:
            render_experiment_tab()
        with tabs[5]:
            render_knowledge_tab(df_main)

    with right_col:
        render_chat_panel()


def render_chat_panel() -> None:
    st.markdown('<div class="copilot-shell">', unsafe_allow_html=True)
    st.markdown('<div class="copilot-card">', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="copilot-topbar">
            <div>
                <span class="copilot-chip">纳米制剂开发助手</span>
                <span class="copilot-chip">智能研究助手</span>
            </div>
            <div style="font-size:12px;color:var(--muted);">Chat</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_history = st.container(height=760)
    with chat_history:
        if not st.session_state.messages:
            st.markdown(
                """
                <div class="copilot-title">你好，今天你有什么想法？</div>
                <div class="copilot-sub">
                    这里就是 纳米制剂开发助手 的专属聊天区。你可以直接提问，
                    也可以从下面这些建议开始。
                </div>
                """,
                unsafe_allow_html=True,
            )

            s1, s2 = st.columns(2)
            if s1.button("创建此页摘要", use_container_width=True, key="right_summary_btn"):
                queue_prompt("请根据当前页面已有信息，帮我做一个简要摘要")
            if s2.button("展开本主题", use_container_width=True, key="right_expand_topic_btn"):
                queue_prompt("请围绕当前项目主题，给我进一步展开思路")

            s3, s4 = st.columns(2)
            if s3.button("生成实验建议", use_container_width=True, key="right_experiment_suggest_btn"):
                queue_prompt("请根据当前项目与数据库情况，生成几个实验建议")
            if s4.button("包封率优化", use_container_width=True, key="right_optimize_ee_btn"):
                queue_prompt("如果我想提高包封率，可以从哪些方向优化？")

            st.markdown(
                """
                <div class="copilot-hint">
                    你可以试着问：
                    <br>• 为什么我的体系会出现白色沉淀？
                    <br>• 帮我筛选粒径小于 100 nm 的样本
                    <br>• 按 apo_type 统计平均粒径
                    <br>• 解释一下 PDI / EE_Percent 字段
                    <br>• 帮我设计几组高包封率参数
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        else:
            for idx, msg in enumerate(st.session_state.messages):
                render_chat_message(msg["role"], msg["content"])
                if msg.get("tool_logs"):
                    for log in msg["tool_logs"]:
                        render_tool_result(log["tool"], log["result"])
                    with st.expander(f"查看本轮工具调用详情 #{idx + 1}", expanded=False):
                        st.json(msg["tool_logs"])

        st.markdown("<div id='smu-chat-bottom-anchor'></div>", unsafe_allow_html=True)

    st.markdown('<div class="copilot-divider"></div>', unsafe_allow_html=True)

    with st.form("smu_agent_chat_form", clear_on_submit=True):
        user_text = st.text_area(
            "输入问题",
            placeholder="向 纳米制剂开发助手 发送消息，例如：帮我筛选 apo_type=22A 且粒径小于100nm 的样本",
            label_visibility="collapsed",
            height=92,
            key="right_chat_input_textarea",
        )
        b1, b2 = st.columns([1.0, 1.0])
        send_btn = b1.form_submit_button("发送", use_container_width=True)
        clear_btn = b2.form_submit_button("清空聊天", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if clear_btn:
        st.session_state.messages = []
        st.rerun()

    prompt_to_run = None
    if send_btn and user_text.strip():
        prompt_to_run = user_text.strip()
    elif st.session_state.pending_prompt:
        prompt_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None

    if prompt_to_run:
        st.session_state.messages.append({"role": "user", "content": prompt_to_run})
        with st.status("纳米制剂开发助手 正在分析并调用工具……", expanded=True) as status:
            result = run_agent(prompt_to_run)
            answer = result.get("answer", "")
            tool_logs = result.get("tool_logs", [])
            status.update(label="处理完成", state="complete", expanded=False)

        st.session_state.messages.append({"role": "assistant", "content": answer, "tool_logs": tool_logs})
        st.rerun()

    auto_scroll_chat()


# =========================
# 主入口
# =========================
def main() -> None:
    init_session_state()
    apply_theme_css(st.session_state.theme_mode)
    render_sidebar()
    df_main, _ = load_active_database()
    render_main_area(df_main)


if __name__ == "__main__":
    main()
