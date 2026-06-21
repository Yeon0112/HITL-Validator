import streamlit as st
import anthropic
import re
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# ── 페이지 설정 ──────────────────────────────────────────
st.set_page_config(
    page_title="HITL 응답 검증기",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 HITL 응답 검증기")
st.caption("Human-in-the-Loop AI Response Validator | 10123 조성현")

# ── API 키 입력 ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input(
        "Anthropic API 키",
        type="password",
        placeholder="sk-ant-..."
    )
    st.caption("API 키는 console.anthropic.com 에서 발급받을 수 있습니다.")
    st.divider()
    st.markdown("**사용 방법**")
    st.markdown("1. API 키 입력")
    st.markdown("2. 질문 입력 후 AI 응답 받기")
    st.markdown("3. 위험 탐지 결과 확인")
    st.markdown("4. 체크리스트 검토 후 수정 방향 입력")
    st.markdown("5. HITL 응답 생성 및 비교")

# ── 위험 탐지 함수 ────────────────────────────────────────
def detect_risk(text):
    risks = []
    score = 0

    # 숫자/수치 탐지
    numbers = re.findall(r'\d+', text)
    if numbers:
        risks.append(f"숫자 포함: {len(numbers)}개 ({', '.join(numbers[:5])})")
        score += 20

    # 고유명사 탐지 (대문자로 시작하는 영단어 또는 한국어 고유명사 패턴)
    proper_nouns = re.findall(r'[A-Z][a-z]+', text)
    korean_proper = re.findall(r'[가-힣]{2,4}(?:의|이|가|은|는|을|를|에|에서|로|으로)', text)
    if proper_nouns or korean_proper:
        risks.append(f"고유명사 포함: {len(proper_nouns) + len(korean_proper)}개")
        score += 15

    # 단정 표현 탐지
    assertive = ['반드시', '항상', '절대', '완전히', '모든', '전혀', '틀림없이',
                 'always', 'never', 'definitely', 'certainly', 'absolutely']
    found_assertive = [w for w in assertive if w in text]
    if found_assertive:
        risks.append(f"단정 표현 발견: {', '.join(found_assertive)}")
        score += 25

    # 출처 없는 인용 탐지
    citation_needed = ['연구에 따르면', '보고서에 따르면', '발표에 따르면',
                       '에 의하면', 'according to', 'studies show']
    found_citation = [w for w in citation_needed if w in text]
    if found_citation:
        risks.append(f"출처 없는 인용 가능성: {', '.join(found_citation)}")
        score += 20

    # 문장 길이 체크 (너무 짧으면 핵심 누락 가능)
    sentences = [s.strip() for s in re.split(r'[.!?。]', text) if s.strip()]
    if len(sentences) < 2:
        risks.append("응답이 너무 짧아 핵심 누락 가능성 있음")
        score += 20

    return risks, min(score, 100)

# ── AI 응답 생성 함수 ─────────────────────────────────────
def get_ai_response(question, mode="basic", review_notes=""):
    client = anthropic.Anthropic(api_key=api_key)

    if mode == "basic":
        prompt = f"""아래 질문에 답하라. 설명은 덧붙이지 말고 답변만 제시하라.

[질문]
{question}"""

    elif mode == "self_review":
        prompt = f"""아래 질문에 답하라. 답변을 작성한 뒤 스스로 다시 검토하여 
사실 오류, 의미 누락, 질문 미충족, 과도한 단정 표현이 있는지 점검하고 반영하라. 
불확실한 내용은 "추가 확인 필요"라고 표시하라. 
설명은 덧붙이지 말고 최종 답변만 제시하라.

[질문]
{question}"""

    elif mode == "hitl":
        prompt = f"""아래 질문에 답하라.

[질문]
{question}

[인간 검토자의 수정 방향]
{review_notes}

위 수정 방향을 반영하여 더 정확하고 신중한 최종 답변만 제시하라."""

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text

# ── 점수 평가 함수 ────────────────────────────────────────
def evaluate_response(response, question):
    scores = {
        "정확성": 0,
        "핵심요소반영": 0,
        "엄밀성": 0,
        "신중성": 0,
        "형식충족": 0,
        "표현완성도": 0
    }

    # 정확성 (길이 기반 휴리스틱)
    if len(response) > 100:
        scores["정확성"] = 3
    elif len(response) > 50:
        scores["정확성"] = 2
    else:
        scores["정확성"] = 1

    # 핵심요소반영
    if len(response) > 150:
        scores["핵심요소반영"] = 2
    else:
        scores["핵심요소반영"] = 1

    # 엄밀성
    specific_words = ['왜냐하면', '따라서', '즉', '예를 들어', 'because',
                      'therefore', 'specifically', '구체적으로']
    if any(w in response for w in specific_words):
        scores["엄밀성"] = 2
    else:
        scores["엄밀성"] = 1

    # 신중성 (단정 표현 없으면 높은 점수)
    assertive = ['반드시', '항상', '절대', '완전히', 'always', 'never', 'definitely']
    if not any(w in response for w in assertive):
        scores["신중성"] = 1
    else:
        scores["신중성"] = 0

    # 형식충족 (문장 수)
    sentences = [s for s in re.split(r'[.!?。]', response) if s.strip()]
    if len(sentences) >= 2:
        scores["형식충족"] = 1
    else:
        scores["형식충족"] = 0

    # 표현완성도
    if response.strip() and not response.endswith('...'):
        scores["표현완성도"] = 1
    else:
        scores["표현완성도"] = 0

    total = (scores["정확성"] +
             scores["핵심요소반영"] +
             scores["엄밀성"] +
             scores["신중성"] +
             scores["형식충족"] +
             scores["표현완성도"])

    return scores, total

# ── 비교 차트 함수 ────────────────────────────────────────
def make_comparison_chart(scores_a, scores_c):
    categories = list(scores_a.keys())
    max_scores = [3, 2, 2, 1, 1, 1]

    values_a = [scores_a[c] / max_scores[i] * 10
                for i, c in enumerate(categories)]
    values_c = [scores_c[c] / max_scores[i] * 10
                for i, c in enumerate(categories)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='A: 기본 응답',
        x=categories,
        y=values_a,
        marker_color='#6baed6'
    ))
    fig.add_trace(go.Bar(
        name='C: HITL 응답',
        x=categories,
        y=values_c,
        marker_color='#fd8d3c'
    ))
    fig.update_layout(
        barmode='group',
        title='A vs C 항목별 점수 비교',
        yaxis=dict(range=[0, 10], title='점수 (10점 환산)'),
        height=350
    )
    return fig

# ── 세션 상태 초기화 ──────────────────────────────────────
if 'response_a' not in st.session_state:
    st.session_state.response_a = ""
if 'response_c' not in st.session_state:
    st.session_state.response_c = ""
if 'risks' not in st.session_state:
    st.session_state.risks = []
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = 0
if 'step' not in st.session_state:
    st.session_state.step = 1

# ── STEP 1: 질문 입력 ─────────────────────────────────────
st.header("STEP 1 · 질문 입력")

question = st.text_area(
    "검증할 질문을 입력하세요",
    placeholder="예: 패러데이의 전자기 유도 법칙이 무엇인지 5문장 이내로 설명하라.",
    height=100
)

if st.button("🤖 AI 응답 받기", type="primary", disabled=not api_key):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        with st.spinner("AI가 응답을 생성하는 중..."):
            st.session_state.response_a = get_ai_response(question, "basic")
            st.session_state.risks, st.session_state.risk_score = detect_risk(
                st.session_state.response_a
            )
            st.session_state.step = 2

if not api_key:
    st.info("왼쪽 사이드바에 API 키를 입력해야 사용할 수 있습니다.")

# ── STEP 2: 위험 탐지 + 인간 검토 ────────────────────────
if st.session_state.step >= 2 and st.session_state.response_a:
    st.divider()
    st.header("STEP 2 · AI 기본 응답 + 위험 탐지")

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("🤖 A: 기본 응답")
        st.info(st.session_state.response_a)

    with col2:
        st.subheader("⚠️ 위험 탐지 결과")
        score = st.session_state.risk_score

        if score >= 60:
            st.error(f"위험 점수: {score}/100 — 높음")
        elif score >= 30:
            st.warning(f"위험 점수: {score}/100 — 중간")
        else:
            st.success(f"위험 점수: {score}/100 — 낮음")

        if st.session_state.risks:
            for r in st.session_state.risks:
                st.markdown(f"- {r}")
        else:
            st.markdown("- 특별한 위험 요소가 탐지되지 않았습니다.")

    st.subheader("✅ 인간 검토 체크리스트")
    col3, col4 = st.columns(2)
    with col3:
        check1 = st.checkbox("사실 관계가 정확한가?")
        check2 = st.checkbox("핵심 요소가 포함되어 있는가?")
    with col4:
        check3 = st.checkbox("표현이 지나치게 단정적이지 않은가?")
        check4 = st.checkbox("형식 요구를 충족했는가?")

    review_notes = st.text_area(
        "수정 방향 입력 (인간 검토자의 피드백)",
        placeholder="예: 마지막 문장의 '모든 장치'는 범위가 너무 넓습니다. "
                    "'여러 장치'로 수정하고, 변화가 없으면 유도가 일어나지 않는다는 "
                    "점을 추가해주세요.",
        height=100
    )

    if st.button("🔄 HITL 응답 생성", type="primary"):
        if not review_notes.strip():
            st.warning("수정 방향을 입력해주세요.")
        else:
            with st.spinner("HITL 응답을 생성하는 중..."):
                st.session_state.response_c = get_ai_response(
                    question, "hitl", review_notes
                )
                st.session_state.step = 3

# ── STEP 3: 결과 비교 ─────────────────────────────────────
if st.session_state.step >= 3 and st.session_state.response_c:
    st.divider()
    st.header("STEP 3 · 결과 비교")

    col5, col6 = st.columns(2)

    scores_a, total_a = evaluate_response(
        st.session_state.response_a, question
    )
    scores_c, total_c = evaluate_response(
        st.session_state.response_c, question
    )

    with col5:
        st.subheader("🤖 A: 기본 응답")
        st.info(st.session_state.response_a)
        st.metric("총점", f"{total_a}/10")

    with col6:
        st.subheader("✅ C: HITL 응답")
        st.success(st.session_state.response_c)
        delta = total_c - total_a
        st.metric("총점", f"{total_c}/10",
                  delta=f"+{delta}" if delta > 0 else str(delta))

    st.subheader("📊 항목별 점수 비교")
    fig = make_comparison_chart(scores_a, scores_c)
    st.plotly_chart(fig, use_container_width=True)

    # 결과 저장
    st.subheader("💾 결과 저장")
    result_data = {
        "시간": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "질문": [question],
        "A_응답": [st.session_state.response_a],
        "C_응답": [st.session_state.response_c],
        "위험점수": [st.session_state.risk_score],
        "A_총점": [total_a],
        "C_총점": [total_c],
        "점수향상": [total_c - total_a]
    }
    df_result = pd.DataFrame(result_data)
    csv = df_result.to_csv(index=False, encoding='utf-8-sig')

    st.download_button(
        label="📥 결과 CSV 다운로드",
        data=csv,
        file_name=f"hitl_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

    if st.button("🔁 새 질문으로 다시 시작"):
        st.session_state.step = 1
        st.session_state.response_a = ""
        st.session_state.response_c = ""
        st.session_state.risks = []
        st.session_state.risk_score = 0
        st.rerun()
