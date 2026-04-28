import os
import io
import json
import random
import re
import html
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

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

client = OpenAI(api_key=API_KEY, base_url=API_BASE)

st.set_page_config(page_title="医学智能体平台", page_icon="🧪", layout="wide")

st.markdown("""
<style>
.main .block-container {padding-top: 0.9rem; padding-bottom: 1.2rem; max-width: 1580px;}
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #f7f9fc 0%, #eef3ff 100%); border-right: 1px solid #e6ebf5;}
.panel-card,.work-card,.soft-card,.smu-shell,.hero-wrap{background:#fff;border:1px solid #e7edf7;box-shadow:0 2px 14px rgba(35,55,80,.05)}
.panel-card{border-radius:18px;padding:14px 16px;margin-bottom:12px}
.work-card{border-radius:22px;padding:14px 16px;margin-bottom:12px}
.soft-card{border-radius:18px;padding:14px 16px;margin-bottom:12px;background:linear-gradient(180deg,#fff 0%,#f8fbff 100%)}
.hero-wrap{border-radius:22px;padding:18px 22px;margin-bottom:14px;background:linear-gradient(135deg,#ffffff 0%,#f4f8ff 100%)}
.panel-title,.soft-title{font-size:16px;font-weight:700;margin-bottom:8px}
.panel-desc,.small-note{color:#5f6b84;font-size:13px;line-height:1.65}
.hero-title{font-size:30px;font-weight:800;margin-bottom:4px}
.hero-subtitle{color:#58657e;font-size:14px;line-height:1.75}
.quickbar{background:linear-gradient(180deg,#fff 0%,#f8fbff 100%);border:1px solid #e2eaf6;border-radius:18px;padding:10px 12px;margin-bottom:12px}
.result-card{background:linear-gradient(180deg,#fff 0%,#f8fbff 100%);border:1px solid #e2eaf6;border-radius:16px;padding:12px 14px;margin-top:8px;margin-bottom:8px}
.result-card-title{font-size:14px;font-weight:700;margin-bottom:6px}
.result-tag{display:inline-block;background:#eef4ff;color:#335caa;border:1px solid #d8e5ff;border-radius:999px;padding:2px 10px;font-size:12px;margin-right:6px;margin-bottom:6px}
.evidence-item{background:#fbfcff;border:1px solid #e7edf8;border-radius:14px;padding:10px 12px;margin-top:8px}
.evidence-title{font-weight:700;font-size:13px;margin-bottom:4px}
.evidence-text{font-size:13px;color:#475467;line-height:1.7}
div[data-testid="stMetric"]{background:#fff;border:1px solid #e5ebf6;border-radius:14px;padding:8px 10px}
.smu-shell{background:linear-gradient(180deg,#ffffff 0%,#f8fbff 100%);border-radius:22px;padding:14px;position:sticky;top:0.8rem}
.smu-head{padding:2px 4px 10px 4px;margin-bottom:10px;border-bottom:1px solid #e8eef8}
.smu-title{font-size:21px;font-weight:800;margin-bottom:2px}
.smu-sub{color:#627089;font-size:13px;line-height:1.6}
.smu-msg{border-radius:18px;padding:12px 13px;margin-bottom:10px;word-wrap:break-word}
.smu-msg.user{background:linear-gradient(180deg,#f5faff 0%,#edf5ff 100%);border:1px solid #dbe8fb}
.smu-msg.agent{background:#fff;border:1px solid #e7edf7;box-shadow:0 2px 10px rgba(35,55,80,.04)}
.smu-role{font-size:12px;font-weight:700;color:#4b5b79;margin-bottom:6px;text-transform:uppercase;letter-spacing:.3px}
.smu-content{font-size:14px;line-height:1.8;color:#1f2937}
.smu-prompt-box{background:#fff;border:1px solid #e7edf7;border-radius:16px;padding:10px;margin-top:10px}
.tag-chip{display:inline-block;background:#f2f6ff;border:1px solid #dbe7ff;color:#365fa8;border-radius:999px;padding:4px 10px;font-size:12px;margin-right:8px;margin-bottom:6px}
</style>
""", unsafe_allow_html=True)

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
    "method_assembly": "组装方法，比如 thermal cycling 等。",
    "buffer_assembly_type": "组装所用缓冲液类型。",
    "ph_assembly": "组装体系 pH。",
    "Shape_Observed": "观察到的形貌，如 discoidal。",
    "Size_Mean_nm": "平均粒径，单位 nm。",
    "PDI": "多分散指数，越小一般说明分布越集中。",
    "Zeta_mV": "Zeta 电位，反映表面电荷特性与分散稳定性。",
    "EE_Percent": "包封率百分比。",
    "DL_Percent": "载药量百分比。",
    "Indication": "适应症或应用方向。",
    "cholesterol efflux(peptide c=10ug/mL)": "10 ug/mL 条件下的胆固醇外排表现。",
    "cholesterol efflux(peptide c=50ug/mL)": "50 ug/mL 条件下的胆固醇外排表现。",
}

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


def unique_keep_order(seq):
    seen = set()
    out = []
    for item in seq:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


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


def safe_dataframe(df: pd.DataFrame, **kwargs):
    safe_df = make_arrow_safe(df)
    st.dataframe(safe_df, **kwargs)


def find_local_db_path() -> Optional[str]:
    for p in DEFAULT_DB_PATHS:
        if os.path.exists(p):
            return p
    return None


@st.cache_data(show_spinner=False)
def get_excel_sheet_names_from_path(path: str) -> List[str]:
    return pd.ExcelFile(path).sheet_names


@st.cache_data(show_spinner=False)
def get_excel_sheet_names_from_bytes(file_bytes: bytes) -> List[str]:
    return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.dropna(axis=1, how="all")

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace({"nan": None, "None": None, "": None})

    numeric_candidates = []
    for col in df.columns:
        lc = str(col).lower()
        if any(k in lc for k in ["ratio", "percent", "size", "pdi", "zeta", "temp", "time", "ph_", "efflux", "dl", "ee", "cycles"]):
            numeric_candidates.append(col)

    for col in sorted(set(numeric_candidates)):
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
    return df if isinstance(df, pd.DataFrame) else None


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
        out = out[out["Size_Mean_nm"].fillna(1e9) <= max_size_nm]
    if max_pdi is not None and "PDI" in out.columns:
        out = out[out["PDI"].fillna(1e9) <= max_pdi]
    if min_ee_percent is not None and "EE_Percent" in out.columns:
        out = out[out["EE_Percent"].fillna(-1e9) >= min_ee_percent]
    return out


def attach_record_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.reset_index().rename(columns={"index": "_row_id"}).copy()
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


def extract_terms(text: str) -> List[str]:
    text = text.lower().strip()
    zh_terms = re.findall(r'[\u4e00-\u9fff]{2,}', text)
    en_terms = re.findall(r'[a-zA-Z0-9_]+', text)
    seen, uniq = set(), []
    for t in zh_terms + en_terms:
        t = t.strip()
        if t and t not in seen:
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
            scored.append({"chunk_id": f"local_{idx+1}", "score": score, "text": line, "source": "本地知识库"})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return {"hits": scored[:top_k]}


def _toy_formula(lipid_ratio: float, protein_ratio: float, temperature: float, time_min: float) -> Dict[str, float]:
    ee = 55 + 5.0 * lipid_ratio + 6.0 * protein_ratio + 0.18 * temperature + 0.25 * time_min - 0.45 * (lipid_ratio - 5.0) ** 2 - 0.90 * (protein_ratio - 1.5) ** 2
    size = 160 - 7.0 * lipid_ratio + 6.5 * protein_ratio - 0.7 * temperature - 0.4 * time_min + 0.30 * (lipid_ratio - 4.0) ** 2
    pdi = 0.26 + 0.015 * abs(protein_ratio - 1.3) + 0.005 * abs(lipid_ratio - 4.8) - 0.0015 * time_min
    ee = max(20.0, min(95.0, ee))
    size = max(45.0, min(300.0, size))
    pdi = max(0.05, min(0.60, pdi))
    return {"predicted_ee": round(ee, 2), "predicted_size_nm": round(size, 2), "predicted_pdi": round(pdi, 3)}


def predict_formulation_impl(lipid_ratio: float, protein_ratio: float, temperature: float, time_min: float) -> Dict[str, Any]:
    pred = _toy_formula(lipid_ratio, protein_ratio, temperature, time_min)
    return {
        "输入": {"脂质比例": lipid_ratio, "蛋白比例": protein_ratio, "温度": temperature, "时间(min)": time_min},
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
            "time_min": round(random.uniform(5.0, 40.0), 1)
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
                "综合评分": round(score, 3)
            })
    candidates.sort(key=lambda x: x["综合评分"], reverse=True)
    return {"目标最低包封率(%)": target_ee_min, "candidates": candidates[:top_k], "说明": "当前为随机采样 + 演示评分，后续可替换为真实优化算法。"}


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
    sort_by: Optional[str] = None,
    ascending: bool = True
) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。请先上传或放置 Excel 数据表。"}
    filtered = apply_dashboard_filters(df, phos_1_type, apo_type, indication, method_assembly, shape_observed, max_size_nm, max_pdi, min_ee_percent)
    if sort_by and sort_by in filtered.columns:
        filtered = filtered.sort_values(by=sort_by, ascending=ascending)
    else:
        if "EE_Percent" in filtered.columns:
            filtered = filtered.sort_values(by="EE_Percent", ascending=False, na_position="last")
        elif "Size_Mean_nm" in filtered.columns:
            filtered = filtered.sort_values(by="Size_Mean_nm", ascending=True, na_position="last")
    preferred_cols = ["ref_id", "formulation_name", "phos_1_type", "phos_1_ratio", "chol_ratio", "apo_type", "apo_ratio", "method_assembly", "Shape_Observed", "Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent", "Indication"]
    cols = [c for c in preferred_cols if c in filtered.columns] or list(filtered.columns[:12])
    preview_df = filtered[cols].head(top_k).copy()
    preview_df = preview_df.where(pd.notnull(preview_df), None)
    return {"matched_count": int(len(filtered)), "preview_count": int(len(preview_df)), "records": preview_df.to_dict(orient="records"), "used_columns": cols}


def aggregate_formulation_database_impl(group_by: str, metric: str, agg: str = "mean", top_k: int = 10, ascending: bool = False) -> Dict[str, Any]:
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
    result = result.sort_values(by=f"{metric}_{agg}", ascending=ascending).head(top_k)
    result = result.where(pd.notnull(result), None)
    return {"group_by": group_by, "metric": metric, "agg": agg, "table": result.to_dict(orient="records")}


def recommend_similar_formulations_impl(anchor_label: str, top_k: int = 5) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。"}
    labeled_df = attach_record_labels(df)
    if "_record_label" not in labeled_df.columns or anchor_label not in labeled_df["_record_label"].tolist():
        return {"error": "未找到指定的锚点样本。"}
    numeric_cols = [c for c in ["phos_1_ratio", "phos_2_ratio", "chol_ratio", "apo_ratio", "Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent"] if c in labeled_df.columns]
    if len(numeric_cols) < 2:
        return {"error": "可用于相似度计算的数值字段不足。"}
    work = labeled_df[["_record_label"] + numeric_cols].copy()
    for col in numeric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        work[col] = work[col].fillna(work[col].median())
    anchor = work[work["_record_label"] == anchor_label].iloc[0]
    calc = work.copy()
    dist = 0.0
    for col in numeric_cols:
        std = calc[col].std()
        std = 1.0 if pd.isna(std) or std == 0 else std
        dist = dist + ((calc[col] - anchor[col]) / std) ** 2
    calc["distance"] = dist ** 0.5
    calc = calc[calc["_record_label"] != anchor_label].sort_values(by="distance").head(top_k)
    merged = calc.merge(labeled_df, on="_record_label", how="left", suffixes=("", "_orig"))
    keep_cols = [c for c in ["_record_label", "distance", "phos_1_type", "apo_type", "Size_Mean_nm", "PDI", "EE_Percent"] if c in merged.columns]
    out = merged[keep_cols].copy()
    out = out.where(pd.notnull(out), None)
    return {"anchor": anchor_label, "records": out.to_dict(orient="records")}


def explain_database_field_impl(field_name: str) -> Dict[str, Any]:
    df = get_active_df()
    if df is None or df.empty:
        return {"error": "当前没有加载结构化数据库。"}
    if field_name not in df.columns:
        return {"error": f"字段不存在：{field_name}"}
    series = df[field_name]
    explanation = FIELD_EXPLANATIONS.get(field_name, "这是当前数据库中的一个字段。你可以结合样本筛选、统计分析和可视化进一步理解它的意义。")
    result = {"field_name": field_name, "explanation": explanation, "dtype": str(series.dtype), "non_null_count": int(series.notna().sum())}
    if pd.api.types.is_numeric_dtype(series):
        result.update({
            "min": None if series.dropna().empty else float(series.min()),
            "max": None if series.dropna().empty else float(series.max()),
            "mean": None if series.dropna().empty else float(series.mean()),
        })
    else:
        uniq = series.dropna().astype(str).value_counts().head(10)
        result["top_values"] = uniq.to_dict()
    return result


TOOL_IMPL = {
    "search_knowledge": search_knowledge_impl,
    "predict_formulation": predict_formulation_impl,
    "reverse_design": reverse_design_impl,
    "query_formulation_database": query_formulation_database_impl,
    "aggregate_formulation_database": aggregate_formulation_database_impl,
    "recommend_similar_formulations": recommend_similar_formulations_impl,
    "explain_database_field": explain_database_field_impl,
}

TOOLS = [
    {"type": "function", "function": {"name": "search_knowledge", "description": "检索本地知识库中的实验经验、文献摘要、故障诊断说明。", "parameters": {"type": "object", "properties": {"question": {"type": "string"}, "top_k": {"type": "integer", "default": 4}}, "required": ["question"]}}},
    {"type": "function", "function": {"name": "predict_formulation", "description": "根据数值工艺参数预测处方表现。", "parameters": {"type": "object", "properties": {"lipid_ratio": {"type": "number"}, "protein_ratio": {"type": "number"}, "temperature": {"type": "number"}, "time_min": {"type": "number"}}, "required": ["lipid_ratio", "protein_ratio", "temperature", "time_min"]}}},
    {"type": "function", "function": {"name": "reverse_design", "description": "根据目标包封率推荐若干候选处方参数。", "parameters": {"type": "object", "properties": {"target_ee_min": {"type": "number", "default": 80.0}, "top_k": {"type": "integer", "default": 5}}, "required": []}}},
    {"type": "function", "function": {"name": "query_formulation_database", "description": "按条件筛选结构化处方数据库，返回符合条件的样本记录。", "parameters": {"type": "object", "properties": {"phos_1_type": {"type": "string"}, "apo_type": {"type": "string"}, "indication": {"type": "string"}, "method_assembly": {"type": "string"}, "shape_observed": {"type": "string"}, "max_size_nm": {"type": "number"}, "max_pdi": {"type": "number"}, "min_ee_percent": {"type": "number"}, "top_k": {"type": "integer", "default": 10}, "sort_by": {"type": "string"}, "ascending": {"type": "boolean", "default": True}}, "required": []}}},
    {"type": "function", "function": {"name": "aggregate_formulation_database", "description": "对结构化数据库按某字段分组，并对某个指标做聚合统计，例如平均粒径、平均包封率等。", "parameters": {"type": "object", "properties": {"group_by": {"type": "string"}, "metric": {"type": "string"}, "agg": {"type": "string", "default": "mean"}, "top_k": {"type": "integer", "default": 10}, "ascending": {"type": "boolean", "default": False}}, "required": ["group_by", "metric"]}}},
    {"type": "function", "function": {"name": "recommend_similar_formulations", "description": "根据一个样本，推荐数值特征相近的相似处方。", "parameters": {"type": "object", "properties": {"anchor_label": {"type": "string"}, "top_k": {"type": "integer", "default": 5}}, "required": ["anchor_label"]}}},
    {"type": "function", "function": {"name": "explain_database_field", "description": "解释数据库字段的含义，并给出该字段的基本统计。", "parameters": {"type": "object", "properties": {"field_name": {"type": "string"}}, "required": ["field_name"]}}},
]

BASE_SYSTEM_PROMPT = """
你是一个医学/药剂学科研智能体原型。

规则：
1. 用户涉及文献经验检索、数据库筛选、统计分析、预测、逆向推荐时，优先调用工具。
2. 不要伪造数值结果。
3. 用户提到异常现象、原因分析、经验总结时，优先调用 search_knowledge。
4. 用户提到筛选处方、比较样本、按字段统计时，优先调用 query_formulation_database 或 aggregate_formulation_database。
5. 用户提到“相似样本”“相似处方”时，优先调用 recommend_similar_formulations。
6. 用户提到“这个字段是什么意思”“解释字段”时，优先调用 explain_database_field。
7. 默认使用中文回答。
8. 如果结果来自演示公式，必须明确说明它只是 demo。
9. 输出风格尽量清晰、结构化、简洁。
"""


def build_system_prompt() -> str:
    prompt = BASE_SYSTEM_PROMPT
    df = get_active_df()
    if df is not None and not df.empty:
        prompt += f"\n当前已加载结构化数据库，共 {len(df)} 条记录。可用字段包括：{', '.join(list(df.columns)[:45])}。"
    return prompt


def run_agent(user_text: str) -> Dict[str, Any]:
    tool_logs: List[Dict[str, Any]] = []
    tool_result_by_call_id: Dict[str, Any] = {}
    messages = [{"role": "system", "content": build_system_prompt()}]
    for msg in st.session_state.messages[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    for _ in range(6):
        response = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
        msg = response.choices[0].message
        if not getattr(msg, "tool_calls", None):
            return {"answer": msg.content or "", "tool_logs": tool_logs}

        assistant_message = {"role": "assistant", "content": msg.content or "", "tool_calls": []}

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

            tool_logs.append({"tool": tool_name, "arguments": args, "result": result})
            tool_result_by_call_id[tool_call.id] = result
            assistant_message["tool_calls"].append({
                "id": tool_call.id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args, ensure_ascii=False)}
            })

        messages.append(assistant_message)

        for tool_call in msg.tool_calls:
            tool_result = tool_result_by_call_id.get(tool_call.id, {"error": "没有找到工具结果"})
            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(tool_result, ensure_ascii=False)})

    return {"answer": "工具调用次数超过上限，本轮已停止。", "tool_logs": tool_logs}


def render_knowledge_hits(hits: List[Dict[str, Any]]):
    if not hits:
        return
    st.markdown('<div class="result-card"><div class="result-card-title">检索到的本地证据</div>', unsafe_allow_html=True)
    for hit in hits:
        st.markdown(
            f'<div class="evidence-item"><div class="evidence-title">条目 {hit.get("chunk_id", "-")} · 相关度 {hit.get("score", "-")}</div><div class="evidence-text">{html.escape(str(hit.get("text", "")))}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)


def render_prediction_card(result: Dict[str, Any]):
    st.markdown('<div class="result-card"><div class="result-card-title">预测结果卡片</div><span class="result-tag">演示模型</span><span class="result-tag">数值预测</span></div>', unsafe_allow_html=True)
    safe_dataframe(pd.DataFrame([result]), use_container_width=True)


def render_reverse_card(result: Dict[str, Any]):
    st.markdown(f'<div class="result-card"><div class="result-card-title">逆向推荐结果</div><span class="result-tag">目标包封率 ≥ {result.get("目标最低包封率(%)", "-")}</span><span class="result-tag">候选参数</span></div>', unsafe_allow_html=True)
    if result.get("candidates"):
        safe_dataframe(pd.DataFrame(result["candidates"]), use_container_width=True)


def render_query_card(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        return
    st.markdown(f'<div class="result-card"><div class="result-card-title">数据库筛选结果</div><span class="result-tag">匹配记录 {result.get("matched_count", 0)} 条</span><span class="result-tag">展示 {result.get("preview_count", 0)} 条</span></div>', unsafe_allow_html=True)
    records = result.get("records", [])
    if records:
        safe_dataframe(pd.DataFrame(records), use_container_width=True)


def render_aggregate_card(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        return
    st.markdown(f'<div class="result-card"><div class="result-card-title">分组统计结果</div><span class="result-tag">{result.get("group_by", "-")}</span><span class="result-tag">{result.get("metric", "-")} · {result.get("agg", "-")}</span></div>', unsafe_allow_html=True)
    if result.get("table"):
        safe_dataframe(pd.DataFrame(result["table"]), use_container_width=True)


def render_similar_card(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        return
    st.markdown(f'<div class="result-card"><div class="result-card-title">相似处方推荐</div><span class="result-tag">锚点样本</span><span class="result-tag">{html.escape(str(result.get("anchor", "-")))}</span></div>', unsafe_allow_html=True)
    if result.get("records"):
        safe_dataframe(pd.DataFrame(result["records"]), use_container_width=True)


def render_field_explain_card(result: Dict[str, Any]):
    if "error" in result:
        st.error(result["error"])
        return
    st.markdown(f'<div class="result-card"><div class="result-card-title">字段说明</div><span class="result-tag">{result.get("field_name", "-")}</span><span class="result-tag">{result.get("dtype", "-")}</span></div>', unsafe_allow_html=True)
    st.write(result.get("explanation", ""))
    aux = {k: v for k, v in result.items() if k not in ["field_name", "explanation"]}
    st.json(aux)


def render_message_card(role: str, content: str):
    role_label = "你" if role == "user" else "SMU-Agent"
    css_class = "user" if role == "user" else "agent"
    safe_content = html.escape(content).replace("\n", "<br>")
    st.markdown(
        f'<div class="smu-msg {css_class}"><div class="smu-role">{role_label}</div><div class="smu-content">{safe_content}</div></div>',
        unsafe_allow_html=True
    )


def trigger_prompt(prompt_text: str):
    st.session_state.pending_prompt = prompt_text
    st.rerun()


with st.sidebar:
    st.markdown('<div class="panel-card"><div class="panel-title">🗂️ 项目区</div><div class="panel-desc">左侧只保留配置和导入功能，主要交互工作台放到中间。</div>', unsafe_allow_html=True)
    st.session_state.project_name = st.text_input("项目名称", value=st.session_state.project_name)
    st.session_state.project_type = st.selectbox(
        "项目类型",
        ["处方开发", "文献综述", "异常诊断", "工艺优化", "数据库整理", "其他"],
        index=["处方开发", "文献综述", "异常诊断", "工艺优化", "数据库整理", "其他"].index(st.session_state.project_type) if st.session_state.project_type in ["处方开发", "文献综述", "异常诊断", "工艺优化", "数据库整理", "其他"] else 0
    )
    st.session_state.project_desc = st.text_area("项目说明", value=st.session_state.project_desc, height=90)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card"><div class="panel-title">🧾 数据区</div><div class="panel-desc">优先读取仓库 data 文件夹，也支持手动上传 xlsx / csv。</div>', unsafe_allow_html=True)
    db_file = st.file_uploader("上传数据库文件", type=["xlsx", "csv"], help="可上传 xlsx 或 csv")
    if db_file is not None:
        st.session_state.db_uploaded_bytes = db_file.getvalue()
        st.session_state.db_uploaded_name = db_file.name

    source_mode = "未找到数据文件"
    sheet_options = []
    if st.session_state.db_uploaded_bytes is not None:
        source_mode = f"已上传：{st.session_state.db_uploaded_name}"
        sheet_options = get_excel_sheet_names_from_bytes(st.session_state.db_uploaded_bytes) if str(st.session_state.db_uploaded_name).lower().endswith(".xlsx") else ["CSV"]
    else:
        local_path = find_local_db_path()
        if local_path:
            source_mode = f"仓库数据文件：{local_path}"
            sheet_options = get_excel_sheet_names_from_path(local_path) if local_path.lower().endswith(".xlsx") else ["CSV"]

    st.info(f"当前数据源：{source_mode}")

    if sheet_options:
        default_idx = sheet_options.index(st.session_state.db_sheet_name) if st.session_state.db_sheet_name in sheet_options else 0
        chosen_sheet = st.selectbox("选择工作表", sheet_options, index=default_idx)
        st.session_state.db_sheet_name = chosen_sheet
        try:
            if st.session_state.db_uploaded_bytes is not None:
                if str(st.session_state.db_uploaded_name).lower().endswith(".csv"):
                    df_active = load_dataframe_from_bytes(st.session_state.db_uploaded_bytes, st.session_state.db_uploaded_name, None)
                else:
                    df_active = load_dataframe_from_bytes(st.session_state.db_uploaded_bytes, st.session_state.db_uploaded_name, chosen_sheet)
                st.session_state.active_db_name = st.session_state.db_uploaded_name
            else:
                local_path = find_local_db_path()
                if local_path:
                    df_active = load_dataframe_from_path(local_path, None if local_path.lower().endswith(".csv") else chosen_sheet)
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

    st.markdown('<div class="panel-card"><div class="panel-title">📚 知识区</div><div class="panel-desc">可导入 txt / md，也可直接粘贴。知识深度检索放到主区完成。</div>', unsafe_allow_html=True)
    kb_files = st.file_uploader("导入知识文件", type=["txt", "md"], accept_multiple_files=True)
    import_mode = st.radio("导入方式", ["追加到现有内容", "替换现有内容"])
    a, b = st.columns(2)
    if a.button("加载示例知识", use_container_width=True):
        st.session_state.knowledge_text = DEFAULT_KNOWLEDGE
        st.rerun()
    if b.button("清空知识区", use_container_width=True):
        st.session_state.knowledge_text = ""
        st.rerun()
    if st.button("导入知识文件", use_container_width=True):
        if kb_files:
            all_text = []
            for f in kb_files:
                text = normalize_uploaded_text(decode_uploaded_file(f))
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
    st.text_area("知识片段", key="knowledge_text", height=180)
    st.markdown('</div>', unsafe_allow_html=True)

df_main = get_active_df()

st.markdown(
    f'<div class="hero-wrap"><div class="hero-title">医学智能体平台 · 双升级版（稳定修复）</div><div class="hero-subtitle">当前项目：<b>{st.session_state.project_name}</b> ｜ 模型：<b>{MODEL}</b> ｜ 当前数据库：<b>{st.session_state.active_db_name}</b><br>这版在保留产品化界面与增强功能的同时，额外修复了 Streamlit Cloud 上 dataframe 的 Arrow 转换报错。</div></div>',
    unsafe_allow_html=True
)

main_left, main_right = st.columns([3.85, 1.45], gap="large")

with main_left:
    st.markdown('<div class="quickbar"><div class="soft-title">⚡ 快捷任务入口</div>', unsafe_allow_html=True)
    qa1, qa2, qa3, qa4, qa5 = st.columns(5)
    if qa1.button("沉淀分析", use_container_width=True):
        trigger_prompt("为什么超声后出现白色沉淀？")
    if qa2.button("筛选 22A", use_container_width=True):
        trigger_prompt("帮我筛选 apo_type=22A 且 Size_Mean_nm 小于 100 的处方")
    if qa3.button("平均粒径", use_container_width=True):
        trigger_prompt("按 apo_type 统计平均粒径")
    if qa4.button("高包封率推荐", use_container_width=True):
        trigger_prompt("帮我设计几组包封率大于80的参数")
    if qa5.button("解释 EE 字段", use_container_width=True):
        trigger_prompt("解释一下 EE_Percent 这个字段")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="work-card"><div class="soft-title">📊 项目总览</div>', unsafe_allow_html=True)
    if df_main is None or df_main.empty:
        st.warning("当前还没有可用的结构化数据库。请把 Excel 放到仓库 `data/` 目录，或在左侧上传 xlsx/csv。")
    else:
        cards = st.columns(6)
        stats = [
            ("记录数", len(df_main)),
            ("文献ID数", int(df_main["ref_id"].nunique()) if "ref_id" in df_main.columns else 0),
            ("磷脂类型数", int(df_main["phos_1_type"].nunique()) if "phos_1_type" in df_main.columns else 0),
            ("Apo类型数", int(df_main["apo_type"].nunique()) if "apo_type" in df_main.columns else 0),
            ("有效粒径数", int(df_main["Size_Mean_nm"].notna().sum()) if "Size_Mean_nm" in df_main.columns else 0),
            ("有效EE数", int(df_main["EE_Percent"].notna().sum()) if "EE_Percent" in df_main.columns else 0),
        ]
        for c, (label, value) in zip(cards, stats):
            c.metric(label, value)
    st.markdown('</div>', unsafe_allow_html=True)

    tabs = st.tabs(["📌 总览看板", "🗃️ 数据工作台", "🆚 样本对比", "🏆 排名与分组", "🧪 实验设计器", "📚 知识与字段工作台"])

    with tabs[0]:
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            c1, c2 = st.columns(2)
            if "phos_1_type" in df_main.columns:
                top_phos = df_main["phos_1_type"].dropna().value_counts().head(10).reset_index()
                top_phos.columns = ["phos_1_type", "count"]
                c1.plotly_chart(px.bar(top_phos, x="phos_1_type", y="count", title="Top 10 磷脂类型分布"), use_container_width=True)
            if "apo_type" in df_main.columns:
                top_apo = df_main["apo_type"].dropna().value_counts().head(10).reset_index()
                top_apo.columns = ["apo_type", "count"]
                c2.plotly_chart(px.bar(top_apo, x="apo_type", y="count", title="Top 10 Apo 类型分布"), use_container_width=True)

            d1, d2 = st.columns(2)
            if "Size_Mean_nm" in df_main.columns and df_main["Size_Mean_nm"].notna().sum() > 0:
                d1.plotly_chart(px.histogram(df_main.dropna(subset=["Size_Mean_nm"]), x="Size_Mean_nm", nbins=30, title="粒径分布"), use_container_width=True)
            if "EE_Percent" in df_main.columns and df_main["EE_Percent"].notna().sum() > 0:
                d2.plotly_chart(px.histogram(df_main.dropna(subset=["EE_Percent"]), x="EE_Percent", nbins=30, title="包封率分布"), use_container_width=True)

            if all(col in df_main.columns for col in ["Size_Mean_nm", "PDI"]) and df_main[["Size_Mean_nm", "PDI"]].dropna().shape[0] > 0:
                color_col = "phos_1_type" if "phos_1_type" in df_main.columns else None
                fig = px.scatter(
                    df_main.dropna(subset=["Size_Mean_nm", "PDI"]).head(450),
                    x="Size_Mean_nm",
                    y="PDI",
                    color=color_col,
                    title="粒径 - PDI 散点图",
                    hover_data=[c for c in ["apo_type", "EE_Percent", "method_assembly"] if c in df_main.columns]
                )
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            st.markdown("#### 条件筛选与记录详情")
            phos_options = ["全部"] + sorted([x for x in df_main["phos_1_type"].dropna().unique().tolist()]) if "phos_1_type" in df_main.columns else ["全部"]
            apo_options = ["全部"] + sorted([x for x in df_main["apo_type"].dropna().unique().tolist()]) if "apo_type" in df_main.columns else ["全部"]
            indication_options = ["全部"] + sorted([x for x in df_main["Indication"].dropna().unique().tolist()]) if "Indication" in df_main.columns else ["全部"]
            assembly_options = ["全部"] + sorted([x for x in df_main["method_assembly"].dropna().unique().tolist()]) if "method_assembly" in df_main.columns else ["全部"]
            shape_options = ["全部"] + sorted([x for x in df_main["Shape_Observed"].dropna().unique().tolist()]) if "Shape_Observed" in df_main.columns else ["全部"]

            f1, f2, f3, f4 = st.columns(4)
            with f1:
                sel_phos = st.selectbox("phos_1_type", phos_options)
                sel_apo = st.selectbox("apo_type", apo_options)
            with f2:
                sel_ind = st.selectbox("Indication", indication_options)
                sel_method = st.selectbox("method_assembly", assembly_options)
            with f3:
                sel_shape = st.selectbox("Shape_Observed", shape_options)
                max_size = st.number_input("最大粒径 (nm)", min_value=0.0, value=150.0, step=5.0)
            with f4:
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
                min_ee_percent=min_ee if "EE_Percent" in df_main.columns and min_ee > 0 else None
            )

            st.success(f"筛选后共 {len(filtered_df)} 条记录。")

            preferred_cols = ["ref_id", "formulation_name", "phos_1_type", "phos_1_ratio", "chol_ratio", "apo_type", "apo_ratio", "method_assembly", "Shape_Observed", "Size_Mean_nm", "PDI", "Zeta_mV", "EE_Percent", "DL_Percent", "Indication"]
            show_cols = [c for c in preferred_cols if c in filtered_df.columns] or list(filtered_df.columns[:15])
            safe_dataframe(filtered_df[show_cols], use_container_width=True, height=320)

            labeled_df = attach_record_labels(filtered_df)
            if not labeled_df.empty:
                selected_label = st.selectbox("查看某一条记录详情", labeled_df["_record_label"].tolist())
                selected_row = labeled_df[labeled_df["_record_label"] == selected_label].iloc[0]
                left_detail, right_detail = st.columns(2)
                detail_cols = [c for c in filtered_df.columns if not str(c).startswith("_")]
                mid = len(detail_cols) // 2 if len(detail_cols) > 1 else len(detail_cols)
                left_detail.json({col: selected_row.get(col) for col in detail_cols[:mid]})
                right_detail.json({col: selected_row.get(col) for col in detail_cols[mid:]})

            st.download_button(
                "下载当前筛选结果 CSV",
                data=filtered_df[show_cols].to_csv(index=False).encode("utf-8-sig"),
                file_name="filtered_formulations.csv",
                mime="text/csv"
            )

    with tabs[2]:
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            st.markdown("#### 样本对比 + 相似样本推荐")
            labeled_df = attach_record_labels(df_main)
            compare_options = labeled_df["_record_label"].tolist()
            default_metrics = [c for c in ["Size_Mean_nm", "PDI", "EE_Percent", "DL_Percent", "Zeta_mV"] if c in df_main.columns]
            all_numeric_cols = [c for c in df_main.columns if pd.api.types.is_numeric_dtype(df_main[c]) and c not in ["ref_id"]]

            compare_labels = st.multiselect("选择要对比的样本（建议 2~6 个）", compare_options, default=compare_options[:3] if len(compare_options) >= 3 else compare_options)
            compare_metrics = st.multiselect("选择对比指标", all_numeric_cols, default=default_metrics[:3] if default_metrics else all_numeric_cols[:3])

            if compare_labels and compare_metrics:
                compare_df = labeled_df[labeled_df["_record_label"].isin(compare_labels)].copy()
                display_cols = unique_keep_order(["_record_label"] + compare_metrics)
                safe_dataframe(compare_df[display_cols], use_container_width=True)

                long_df = compare_df[display_cols].melt(id_vars="_record_label", value_vars=compare_metrics, var_name="metric", value_name="value")
                st.plotly_chart(px.bar(long_df, x="_record_label", y="value", color="metric", barmode="group", title="样本对比图"), use_container_width=True)

            st.markdown("#### 相似处方推荐")
            anchor_label = st.selectbox("选择一个锚点样本", compare_options, key="similar_anchor")
            similar_topk = st.slider("相似样本数量", 3, 10, 5, 1)
            if st.button("计算相似样本", use_container_width=True):
                sim_result = recommend_similar_formulations_impl(anchor_label, similar_topk)
                render_similar_card(sim_result)

    with tabs[3]:
        if df_main is None or df_main.empty:
            st.info("请先加载数据库。")
        else:
            st.markdown("#### Top N 排名")
            numeric_cols = [c for c in df_main.columns if pd.api.types.is_numeric_dtype(df_main[c]) and c not in ["ref_id"]]
            cat_cols = [c for c in ["phos_1_type", "apo_type", "method_assembly", "Shape_Observed", "Indication"] if c in df_main.columns]

            r1, r2, r3 = st.columns(3)
            rank_metric = r1.selectbox("排名指标", numeric_cols, index=numeric_cols.index("EE_Percent") if "EE_Percent" in numeric_cols else 0)
            rank_order = r2.selectbox("排序方向", ["降序（越大越靠前）", "升序（越小越靠前）"])
            rank_topn = r3.slider("Top N", 3, 30, 10)

            rank_df = attach_record_labels(df_main).dropna(subset=[rank_metric]).copy()
            rank_df = rank_df.sort_values(by=rank_metric, ascending=(rank_order == "升序（越小越靠前）")).head(rank_topn)

            rank_show_cols = unique_keep_order([c for c in ["_record_label", "phos_1_type", "apo_type", rank_metric, "PDI", "Size_Mean_nm", "EE_Percent"] if c in rank_df.columns])
            rank_display_df = rank_df[rank_show_cols].copy()
            safe_dataframe(rank_display_df, use_container_width=True)

            st.plotly_chart(px.bar(rank_df, x="_record_label", y=rank_metric, title=f"{rank_metric} Top {rank_topn}"), use_container_width=True)

            st.markdown("#### 分组统计")
            if cat_cols and numeric_cols:
                g1, g2, g3 = st.columns(3)
                group_col = g1.selectbox("分组字段", cat_cols)
                metric_col = g2.selectbox("统计指标", numeric_cols, index=numeric_cols.index("Size_Mean_nm") if "Size_Mean_nm" in numeric_cols else 0)
                agg_method = g3.selectbox("聚合方式", ["mean", "median", "max", "min", "count"])
                agg_result = aggregate_formulation_database_impl(group_col, metric_col, agg_method, top_k=20, ascending=False)
                if "table" in agg_result:
                    result_df = pd.DataFrame(agg_result["table"])
                    safe_dataframe(result_df, use_container_width=True)
                    value_col = f"{metric_col}_{agg_method}"
                    if not result_df.empty and value_col in result_df.columns:
                        st.plotly_chart(px.bar(result_df, x=group_col, y=value_col, title=f"{group_col} - {metric_col} ({agg_method})"), use_container_width=True)

    with tabs[4]:
        st.markdown("#### 参数实验设计器")
        st.markdown('<div class="small-note">这部分先用演示公式做即时交互。后面你接入真实模型后，这里就会变成真正的数据驱动设计器。</div>', unsafe_allow_html=True)
        e1, e2 = st.columns([1.2, 1.0])

        with e1:
            lipid_ratio = st.slider("脂质比例 lipid_ratio", 1.0, 8.0, 4.5, 0.1)
            protein_ratio = st.slider("蛋白比例 protein_ratio", 0.4, 2.5, 1.2, 0.1)
            temperature = st.slider("温度 temperature", 20.0, 45.0, 37.0, 0.5)
            time_min = st.slider("时间 time_min", 5.0, 40.0, 20.0, 1.0)

            pred = predict_formulation_impl(lipid_ratio, protein_ratio, temperature, time_min)

            m1, m2, m3 = st.columns(3)
            m1.metric("预测包封率(%)", pred["预测包封率(%)"])
            m2.metric("预测粒径(nm)", pred["预测粒径(nm)"])
            m3.metric("预测PDI", pred["预测PDI"])

            if st.button("让 SMU-Agent 解释当前参数", use_container_width=True):
                trigger_prompt(f"请解释这组参数的表现：lipid_ratio={lipid_ratio}, protein_ratio={protein_ratio}, temperature={temperature}, time_min={time_min}")

        with e2:
            design_df = pd.DataFrame([{
                "脂质比例": lipid_ratio,
                "蛋白比例": protein_ratio,
                "温度": temperature,
                "时间(min)": time_min,
                "预测包封率(%)": pred["预测包封率(%)"],
                "预测粒径(nm)": pred["预测粒径(nm)"],
                "预测PDI": pred["预测PDI"]
            }])
            safe_dataframe(design_df, use_container_width=True)

            target_ee = st.slider("目标包封率阈值", 60.0, 95.0, 80.0, 1.0)
            top_k = st.slider("推荐候选数", 3, 10, 5, 1)
            if st.button("生成候选参数方案", use_container_width=True):
                render_reverse_card(reverse_design_impl(target_ee_min=target_ee, top_k=top_k))

    with tabs[5]:
        st.markdown("#### 知识检索 + 字段说明工作台")
        s1, s2 = st.columns([1.2, 1.0])

        with s1:
            query_text = st.text_input("输入一个知识问题", value="为什么超声后会出现白色沉淀？")
            cqa1, cqa2 = st.columns(2)
            if cqa1.button("检索知识库", use_container_width=True) and query_text.strip():
                hits = search_knowledge_impl(query_text.strip(), top_k=6).get("hits", [])
                if hits:
                    render_knowledge_hits(hits)
                else:
                    st.warning("当前知识库中没有命中结果。")
            if cqa2.button("交给 SMU-Agent 分析", use_container_width=True) and query_text.strip():
                trigger_prompt(query_text.strip())

            st.metric("知识条目数", len([x for x in st.session_state.knowledge_text.splitlines() if x.strip()]))
            st.text_area("知识内容预览", value=st.session_state.knowledge_text, height=280)

        with s2:
            if df_main is not None and not df_main.empty:
                field_name = st.selectbox("选择一个字段查看说明", df_main.columns.tolist())
                field_result = explain_database_field_impl(field_name)
                render_field_explain_card(field_result)
                if st.button("让 SMU-Agent 解释这个字段", use_container_width=True):
                    trigger_prompt(f"请解释一下字段 {field_name} 的含义，并结合当前数据库说明它有什么用")
            else:
                st.info("请先加载数据库。")

with main_right:
    st.markdown('<div class="smu-shell"><div class="smu-head"><div class="smu-title">SMU-Agent</div><div class="smu-sub">这里不是普通聊天框，而是你的右侧智能研究助手。它可以解释知识、筛选数据库、做统计、推荐参数，也能解释字段。</div></div>', unsafe_allow_html=True)

    q1, q2 = st.columns(2)
    if q1.button("沉淀原因", use_container_width=True):
        trigger_prompt("为什么超声后出现白色沉淀？")
    if q2.button("预测参数", use_container_width=True):
        trigger_prompt("预测一下：lipid_ratio=4.5, protein_ratio=1.2, temperature=37, time_min=20")

    q3, q4 = st.columns(2)
    if q3.button("筛选处方", use_container_width=True):
        trigger_prompt("帮我筛选 apo_type=22A 且 Size_Mean_nm 小于 100 的处方")
    if q4.button("解释字段", use_container_width=True):
        trigger_prompt("解释一下 PDI 这个字段")

    q5, q6 = st.columns(2)
    if q5.button("相似样本", use_container_width=True):
        if df_main is not None and not df_main.empty:
            anchor_default = attach_record_labels(df_main)["_record_label"].iloc[0]
            trigger_prompt(f"帮我找和 {anchor_default} 相似的处方")
    if q6.button("分组统计", use_container_width=True):
        trigger_prompt("按 phos_1_type 统计平均包封率")

    st.markdown("---")

    for msg in st.session_state.messages:
        render_message_card(msg["role"], msg["content"])
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
                elif log["tool"] == "recommend_similar_formulations":
                    render_similar_card(log["result"])
                elif log["tool"] == "explain_database_field":
                    render_field_explain_card(log["result"])
            with st.expander("查看工具调用详情"):
                st.json(msg["tool_logs"])

    st.markdown('<div class="smu-prompt-box">', unsafe_allow_html=True)
    with st.form("smu_agent_form", clear_on_submit=True):
        user_text = st.text_area(
            "输入问题",
            placeholder="例如：按 phos_1_type 统计平均包封率；或者帮我找 PDI<0.2 且粒径较小的样本",
            height=100,
            label_visibility="collapsed"
        )
        sf1, sf2 = st.columns(2)
        submitted = sf1.form_submit_button("发送给 SMU-Agent", use_container_width=True)
        clear_chat = sf2.form_submit_button("清空对话", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if clear_chat:
        st.session_state.messages = []
        st.rerun()

    prompt = user_text.strip() if submitted and user_text.strip() else st.session_state.pending_prompt
    if prompt:
        st.session_state.pending_prompt = None
        st.session_state.messages.append({"role": "user", "content": prompt})

        render_message_card("user", prompt)

        with st.status("SMU-Agent 正在分析并决定调用哪些工具……", expanded=True) as status:
            result = run_agent(prompt)
            answer, tool_logs = result["answer"], result["tool_logs"]
            status.update(label="处理完成", state="complete", expanded=False)

        render_message_card("assistant", answer)

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
            elif log["tool"] == "recommend_similar_formulations":
                render_similar_card(log["result"])
            elif log["tool"] == "explain_database_field":
                render_field_explain_card(log["result"])

        with st.expander("查看工具调用详情"):
            st.json(tool_logs)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "tool_logs": tool_logs
        })

    st.markdown('</div>', unsafe_allow_html=True)
