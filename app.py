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
st.caption("Human-in-the-Loop AI Response Validator | 20721 조성현")

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
    st.markdown("**실험 조건 설명**")
    st.markdown("🔵 **A조건** : AI 기본 응답 (검토 없음)")
    st.markdown("🟢 **B조건** : AI 자기검토 응답")
    st.markdown("🟠 **C조건** : HITL 검증 루프 응답")
    st.divider()
    st.markdown("**사용 방법**")
    st.markdown("1. API 키 입력")
    st.markdown("2. 질문 입력 후 AI 응답 받기")
    st.markdown("3. A/B 응답 자동 생성 확인")
    st.markdown("4. 위험 탐지 결과 확인")
    st.markdown("5. 체크리스트 검토 후 수정 방향 입력")
    st.markdown("6. C응답 생성 및 A/B/C 비교")

# ── 위험 탐지 함수 ────────────────────────────────────────
def detect_risk(text):
    risks = []
    score = 0

    numbers = re.findall(r'\d+', text)
    if numbers:
        risks.append(f"숫자 포함: {len(numbers)}개 ({', '.join(numbers[:5])})")
        score += 20

    proper_nouns = re.findall(r'[A-Z][a-z]+', text)
    if proper_nouns:
        risks.append(f"영문 고유명사 포함: {len(proper_nouns)}개")
        score += 15

    assertive = ['반드시', '항상', '절대', '완전히', '모든', '전혀',
                 '틀림없이', 'always', 'never', 'definitely',
                 'certainly', 'absolutely']
    found_assertive = [w for w in assertive if w in text]
    if found_assertive:
        risks.append(f"단정 표현 발견: {', '.join(found_assertive)}")
        score += 25

    citation_needed = ['연구에 따르면', '보고서에 따르면',
                       '에 의하면', 'according to', 'studies show']
    found_citation = [w for w in citation_needed if w in text]
    if found_citation:
        risks.append(f"출처 없는 인용 가능성: {', '.join(found_citation)}")
        score += 20

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
def evaluate_response(response):
    scores = {
        "정확성": 0,
        "핵심요소반영": 0,
        "엄밀성": 0,
        "신중성": 0,
        "형식충족": 0,
        "표현완성도": 0
    }

    if len(response) > 100:
        scores["정확성"] = 3
    elif len(response) > 50:
        scores["정확성"] = 2
    else:
        scores["정확성"] = 1

    if len(response) > 150:
        scores["핵심요소반영"] = 2
    else:
        scores["핵심요소반영"] = 1

    specific_words = ['왜냐하면', '따라서', '즉', '예를 들어',
                      'because', 'therefore', 'specifically', '구체적으로']
    if any(w in response for w in specific_words):
        scores["엄밀성"] = 2
    else:
        scores["엄밀성"] = 1

    assertive = ['반드시', '항상', '절대', '완전히',
                 'always', 'never', 'definitely']
    if not any(w in response for w in assertive):
        scores["신중성"] = 1
    else:
        scores["신중성"] = 0

    sentences = [s for s in re.split(r'[.!?。]', response) if s.strip()]
    scores["형식충족"] = 1 if len(sentences) >= 2 else 0

    scores["표현완성도"] = 1 if (
        response.strip() and not response.endswith('...')
    ) else 0

    total = sum(scores.values())
    return scores, total

# ── 비교 차트 함수 (A/B/C 3조건) ─────────────────────────
def make_comparison_chart(scores_a, scores_b, scores_c):
    categories = list(scores_a.keys())
    max_scores = [3, 2, 2, 1, 1, 1]

    def normalize(scores):
        return [scores[c] / max_scores[i] * 10
                for i, c in enumerate(categories)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='A: 기본 응답',
        x=categories,
        y=normalize(scores_a),
        marker_color='#6baed6'
    ))
    fig.add_trace(go.Bar(
        name='B: 자기검토 응답',
        x=categories,
        y=normalize(scores_b),
        marker_color='#74c476'
    ))
    fig.add_trace(go.Bar(
        name='C: HITL 응답',
        x=categories,
        y=normalize(scores_c),
        marker_color='#fd8d3c'
    ))
    fig.update_layout(
        barmode='group',
        title='A / B / C 조건별 항목별 점수 비교',
        yaxis=dict(range=[0, 10], title='점수 (10점 환산)'),
        height=380,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1)
    )
    return fig

# ── 세션 상태 초기화 ──────────────────────────────────────
for key, default in {
    'response_a': '',
    'response_b': '',
    'response_c': '',
    'risks': [],
    'risk_score': 0,
    'step': 1,
    'question': ''
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── STEP 1: 질문 입력 ─────────────────────────────────────
st.header("STEP 1 · 질문 입력")

question = st.text_area(
    "검증할 질문을 입력하세요",
    placeholder="예: 패러데이의 전자기 유도 법칙이 무엇인지 5문장 이내로 설명하라.",
    height=100
)

if st.button("🤖 AI 응답 받기 (A·B 동시 생성)",
             type="primary", disabled=not api_key):
    if not question.strip():
        st.warning("질문을 입력해주세요.")
    else:
        st.session_state.question = question
        col_prog1, col_prog2 = st.columns(2)
        with col_prog1:
            with st.spinner("A조건 (기본 응답) 생성 중..."):
                st.session_state.response_a = get_ai_response(
                    question, "basic"
                )
        with col_prog2:
            with st.spinner("B조건 (자기검토 응답) 생성 중..."):
                st.session_state.response_b = get_ai_response(
                    question, "self_review"
                )
        st.session_state.risks, st.session_state.risk_score = detect_risk(
            st.session_state.response_a
        )
        st.session_state.step = 2

if not api_key:
    st.info("왼쪽 사이드바에 API 키를 입력해야 사용할 수 있습니다.")

# ── STEP 2: A·B 응답 + 위험 탐지 + 인간 검토 ────────────
if st.session_state.step >= 2 and st.session_state.response_a:
    st.divider()
    st.header("STEP 2 · A·B 응답 확인 + 위험 탐지")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🔵 A: 기본 응답")
        st.info(st.session_state.response_a)
        scores_a_preview, total_a_preview = evaluate_response(
            st.session_state.response_a
        )
        st.caption(f"자동 평가 점수: {total_a_preview}/10")

    with col2:
        st.subheader("🟢 B: 자기검토 응답")
        st.success(st.session_state.response_b)
        scores_b_preview, total_b_preview = evaluate_response(
            st.session_state.response_b
        )
        st.caption(f"자동 평가 점수: {total_b_preview}/10")

    st.subheader("⚠️ 위험 탐지 결과 (A 응답 기준)")
    score = st.session_state.risk_score
    col_r1, col_r2 = st.columns([1, 3])
    with col_r1:
        if score >= 60:
            st.error(f"위험 점수\n## {score}/100\n높음")
        elif score >= 30:
            st.warning(f"위험 점수\n## {score}/100\n중간")
        else:
            st.success(f"위험 점수\n## {score}/100\n낮음")
    with col_r2:
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
        placeholder="예: A응답의 수식 표현은 너무 전문적입니다. "
                    "수식 없이 말로 풀어서 설명하고, "
                    "마지막 문장을 더 간결하게 마무리해주세요.",
        height=100
    )

    if st.button("🔄 C조건 HITL 응답 생성", type="primary"):
        if not review_notes.strip():
            st.warning("수정 방향을 입력해주세요.")
        else:
            with st.spinner("C조건 (HITL) 응답을 생성하는 중..."):
                st.session_state.response_c = get_ai_response(
                    st.session_state.question, "hitl", review_notes
                )
                st.session_state.step = 3

# ── STEP 3: A/B/C 전체 비교 ──────────────────────────────
if st.session_state.step >= 3 and st.session_state.response_c:
    st.divider()
    st.header("STEP 3 · A / B / C 전체 비교")

    col5, col6, col7 = st.columns(3)

    scores_a, total_a = evaluate_response(st.session_state.response_a)
    scores_b, total_b = evaluate_response(st.session_state.response_b)
    scores_c, total_c = evaluate_response(st.session_state.response_c)

    with col5:
        st.subheader("🔵 A: 기본 응답")
        st.info(st.session_state.response_a)
        st.metric("총점", f"{total_a}/10")

    with col6:
        st.subheader("🟢 B: 자기검토 응답")
        st.success(st.session_state.response_b)
        delta_b = total_b - total_a
        st.metric("총점", f"{total_b}/10",
                  delta=f"+{delta_b}" if delta_b > 0 else str(delta_b))

    with col7:
        st.subheader("🟠 C: HITL 응답")
        st.warning(st.session_state.response_c)
        delta_c = total_c - total_a
        st.metric("총점", f"{total_c}/10",
                  delta=f"+{delta_c}" if delta_c > 0 else str(delta_c))

    st.subheader("📊 A / B / C 항목별 점수 비교")
    fig = make_comparison_chart(scores_a, scores_b, scores_c)
    st.plotly_chart(fig, use_container_width=True)

    # 결과 저장
    st.subheader("💾 결과 저장")
    result_data = {
        "시간": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "질문": [st.session_state.question],
        "A_응답": [st.session_state.response_a],
        "B_응답": [st.session_state.response_b],
        "C_응답": [st.session_state.response_c],
        "위험점수": [st.session_state.risk_score],
        "A_총점": [total_a],
        "B_총점": [total_b],
        "C_총점": [total_c],
        "B_향상": [total_b - total_a],
        "C_향상": [total_c - total_a]
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
        for key in ['response_a', 'response_b', 'response_c',
                    'risks', 'question']:
            st.session_state[key] = '' if isinstance(
                st.session_state[key], str) else []
        st.session_state.risk_score = 0
        st.session_state.step = 1
        st.rerun()
