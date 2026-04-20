import os
import json
import random
import re
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# =========================
# 环境变量 / Secrets
# =========================
load_dotenv()

def get_secret(name: str, default: str = "") -> str:
    """
    优先读取 Streamlit Cloud 的 secrets；
    如果没有，再读取本地 .env / 系统环境变量。
    """
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
/* 页面整体 */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.5rem;
    max-width: 1200px;
}

/* 侧边栏 */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f7f9fc 0%, #eef3ff 100%);
    border-right: 1px solid #e6ebf5;
}

section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea,
section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stRadio > div {
    border-radius: 12px !important;
}

section[data-testid="stSidebar"] .stTextArea textarea {
    border: 1px solid #d8e1f0;
    background: #ffffff;
    line-height: 1.65;
    font-size: 14px;
}

section[data-testid="stSidebar"] .stButton button {
    border-radius: 12px;
    border: 1px solid #d5deee;
    background: white;
    height: 2.6rem;
}

section[data-testid="stSidebar"] .stFileUploader {
    background: rgba(255,255,255,0.72);
    border: 1px dashed #cbd6ea;
    padding: 10px;
    border-radius: 14px;
}

/* 卡片 */
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

/* 顶部标题 */
.hero-wrap {
    background: linear-gradient(135deg, #ffffff 0%, #f4f7ff 100%);
    border: 1px solid #e7edf7;
    border-radius: 18px;
    padding: 18px 20px;
    margin-bottom: 14px;
    box-shadow: 0 2px 14px rgba(38, 58, 90, 0.05);
}
.hero-title {
    font-size: 28px;
    font-weight: 800;
    margin-bottom: 6px;
}
.hero-subtitle {
    color: #5b6780;
    font-size: 14px;
    line-height: 1.7;
}

/* 聊天气泡 */
[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 8px 10px 2px 10px;
    margin-bottom: 10px;
}

[data-testid="stChatMessage"]:has([aria-label="user avatar"]) {
    background: linear-gradient(180deg, #f7fbff 0%, #eef6ff 100%);
    border: 1px solid #d9e8fb;
}

[data-testid="stChatMessage"]:has([aria-label="assistant avatar"]) {
    background: #ffffff;
    border: 1px solid #e8edf6;
    box-shadow: 0 2px 12px rgba(35, 55, 80, 0.04);
}

/* 结果卡片 */
.result-card {
    background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
    border: 1px solid #e2eaf6;
    border-radius: 16px;
    padding: 14px 16px;
    margin-top: 10px;
    margin-bottom: 10px;
}

.result-card-title {
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 8px;
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

/* 证据条目 */
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

/* 指标 */
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e5ebf6;
    border-radius: 14px;
    padding: 8px 10px;
}

/* 输入框 */
[data-testid="stChatInput"] {
    position: sticky;
    bottom: 0.4rem;
}
</style>
""", unsafe_allow_html=True)

DEFAULT_KNOWLEDGE = """\
rHDL 制剂在胆固醇比例过高时，可能出现沉淀或体系不稳定。
超声过程温度过低时，均质效率可能下降，导致颗粒分散不充分。
适中的脂质/蛋白比例通常更有利于颗粒稳定性。
包封率通常受脂质比例、蛋白比例、温度和处理时间共同影响。
若体系出现明显白色沉淀，应优先排查胆固醇比例、温度以及原料溶解状态。
"""

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
    st.session_state.project_desc = "围绕纳米制剂/脂蛋白体系，进行文献检索、处方预测与逆向推荐。"

if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# =========================
# 工具函数
# =========================
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
        line = re.sub(r'^\s*#+\s*', '', line)         # md 标题
        line = re.sub(r'^\s*[-*+]\s*', '', line)      # md 列表
        line = re.sub(r'^\s*\d+\.\s*', '', line)      # 编号列表
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


def search_knowledge_impl(question: str, top_k: int = 3) -> Dict[str, Any]:
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
        "inputs": {
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


def reverse_design_impl(target_ee_min: float = 80.0, top_k: int = 3) -> Dict[str, Any]:
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


TOOL_IMPL = {
    "search_knowledge": search_knowledge_impl,
    "predict_formulation": predict_formulation_impl,
    "reverse_design": reverse_design_impl,
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "检索用户本地知识库中的文献摘要、经验记录或实验说明。",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "top_k": {"type": "integer", "default": 3}
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
                    "top_k": {"type": "integer", "default": 3}
                },
                "required": []
            }
        }
    }
]

SYSTEM_PROMPT = """
你是一个医学/药剂学科研智能体原型。

规则：
1. 用户涉及检索、预测、逆向推荐时，优先调用工具。
2. 不要伪造数值预测结果。
3. 用户提到故障、异常、现象分析时，优先调用 search_knowledge。
4. 默认使用中文回答。
5. 如果结果来自演示公式，必须明确说明它只是 demo。
6. 输出风格尽量清晰、结构化、简洁。
"""

# =========================
# Agent
# =========================
def run_agent(user_text: str) -> Dict[str, Any]:
    tool_logs: List[Dict[str, Any]] = []
    tool_result_by_call_id: Dict[str, Any] = {}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in st.session_state.messages[-8:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_text})

    for _ in range(5):
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
# 辅助渲染
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

# =========================
# 顶部
# =========================
st.markdown(f"""
<div class="hero-wrap">
    <div class="hero-title">医学智能体平台 · 演示版</div>
    <div class="hero-subtitle">
        当前项目：<b>{st.session_state.project_name}</b> ｜ 模型：<b>{MODEL}</b><br>
        已支持本地知识检索、参数预测、逆向推荐。左侧可维护项目配置与知识库内容。
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# 左侧栏：项目区 / 知识库区 / 测试区
# =========================
with st.sidebar:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🗂️ 项目区</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">维护当前课题名称、课题类型和项目说明。这里先做前端状态保存。</div>', unsafe_allow_html=True)

    st.session_state.project_name = st.text_input("项目名称", value=st.session_state.project_name)
    st.session_state.project_type = st.selectbox(
        "项目类型",
        ["处方开发", "文献综述", "异常诊断", "工艺优化", "其他"],
        index=["处方开发", "文献综述", "异常诊断", "工艺优化", "其他"].index(st.session_state.project_type)
        if st.session_state.project_type in ["处方开发", "文献综述", "异常诊断", "工艺优化", "其他"] else 0
    )
    st.session_state.project_desc = st.text_area("项目说明", value=st.session_state.project_desc, height=90)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">📚 知识库区</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">支持手动编辑，也支持导入 txt / md 文件。每行尽量是一条完整知识点。</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "导入知识文件",
        type=["txt", "md"],
        accept_multiple_files=True,
        help="支持一次选择多个 txt / md 文件"
    )

    import_mode = st.radio("导入方式", ["追加到现有内容", "替换现有内容"], horizontal=False)

    col_k1, col_k2 = st.columns(2)
    with col_k1:
        if st.button("加载示例知识", use_container_width=True):
            st.session_state.knowledge_text = DEFAULT_KNOWLEDGE
            st.rerun()
    with col_k2:
        if st.button("清空知识库", use_container_width=True):
            st.session_state.knowledge_text = ""
            st.rerun()

    if st.button("导入已上传文件", use_container_width=True):
        if uploaded_files:
            file_texts = []
            for f in uploaded_files:
                decoded = decode_uploaded_file(f)
                normalized = normalize_uploaded_text(decoded)
                if normalized.strip():
                    file_texts.append(normalized)

            merged = "\n".join(file_texts).strip()

            if merged:
                if import_mode == "替换现有内容":
                    st.session_state.knowledge_text = merged
                else:
                    current = st.session_state.knowledge_text.strip()
                    if current:
                        st.session_state.knowledge_text = current + "\n" + merged
                    else:
                        st.session_state.knowledge_text = merged
                st.success("知识文件已导入。")
                st.rerun()
            else:
                st.warning("上传文件中没有可导入的有效文本。")
        else:
            st.warning("请先上传 txt 或 md 文件。")

    st.text_area(
        "知识片段内容",
        key="knowledge_text",
        height=260,
        placeholder="在这里粘贴你的文献摘要、实验经验、失败案例或 SOP 片段。"
    )

    lines_count = len([x for x in st.session_state.knowledge_text.splitlines() if x.strip()])
    char_count = len(st.session_state.knowledge_text.strip())

    mk1, mk2 = st.columns(2)
    with mk1:
        st.metric("条目数", lines_count)
    with mk2:
        st.metric("字符数", char_count)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-title">🧪 测试区</div>', unsafe_allow_html=True)
    st.markdown('<div class="panel-desc">点击下面按钮可直接把问题送入聊天区，便于快速演示。</div>', unsafe_allow_html=True)

    if st.button("测试：沉淀原因分析", use_container_width=True):
        st.session_state.pending_prompt = "为什么超声后出现白色沉淀？"
        st.rerun()

    if st.button("测试：参数预测", use_container_width=True):
        st.session_state.pending_prompt = "预测一下：lipid_ratio=4.5, protein_ratio=1.2, temperature=37, time_min=20"
        st.rerun()

    if st.button("测试：逆向推荐", use_container_width=True):
        st.session_state.pending_prompt = "帮我设计几组包封率大于80的参数"
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 历史消息
# =========================
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

            with st.expander("查看工具调用详情"):
                st.json(msg["tool_logs"])

# =========================
# 输入处理
# =========================
manual_prompt = st.chat_input("请输入你的问题，例如：帮我设计几组包封率大于 80 的处方参数")
prompt = manual_prompt or st.session_state.pending_prompt

if prompt:
    st.session_state.pending_prompt = None
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.status("正在分析问题并决定是否调用工具……", expanded=True) as status:
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

        with st.expander("查看工具调用详情"):
            st.json(tool_logs)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "tool_logs": tool_logs,
    })