import os
import re
import time
from difflib import SequenceMatcher

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="LLM Chatbot | Prompt Engineering", page_icon="💬", layout="wide")
st.title("💬 Chatbot Using LLM with Prompt Engineering")
st.caption("Question Answering • Text Summarization • Text Classification • FAQ Support • Response Evaluation")

# -------------------- Configuration --------------------
api_key = os.getenv("OPENAI_API_KEY", "").strip()
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

with st.sidebar:
    st.header("Settings")
    model_name = st.text_input("LLM model", value=model_name,
                               help="Must be a model available to your API key.")
    prompt_style = st.radio("Prompt engineering mode",
                            ["Improved prompt", "Basic prompt"],
                            help="Compare a clear, structured prompt with a simple instruction.")
    st.caption("API key is read from the OPENAI_API_KEY environment variable or .env file.")
    st.divider()
    st.subheader("Project features")
    st.markdown("- Question answering and FAQ support\n- Text summarization\n- Text classification\n- Prompt comparison\n- Quality evaluation and consistency check")

if not api_key:
    st.warning("API key not found. Create a `.env` file from `.env.example` and add your API key.")
    st.code("OPENAI_API_KEY=your_api_key_here\nOPENAI_MODEL=gpt-4o-mini", language="text")
    st.info("Do not share your API key or upload it to GitHub.")
    st.stop()

client = OpenAI(api_key=api_key)

# -------------------- Dataset helpers --------------------
FAQ_PATH = "faq_dataset.csv"
CUSTOM_PATH = "custom_queries.csv"

def load_csv(path, columns):
    try:
        df = pd.read_csv(path)
        if all(c in df.columns for c in columns):
            return df.fillna("")
    except Exception:
        pass
    return pd.DataFrame(columns=columns)

faq_df = load_csv(FAQ_PATH, ["question", "answer"])
custom_df = load_csv(CUSTOM_PATH, ["task", "input", "expected_output"])

def tokens(text):
    return set(re.findall(r"[a-zA-Z0-9]+", str(text).lower()))

def retrieve_faq(query, df, top_n=3):
    q_tokens = tokens(query)
    if not q_tokens or df.empty:
        return []
    ranked = []
    for _, row in df.iterrows():
        candidate = str(row["question"])
        overlap = len(q_tokens & tokens(candidate)) / max(1, len(q_tokens | tokens(candidate)))
        if overlap > 0:
            ranked.append((overlap, candidate, str(row["answer"])))
    ranked.sort(reverse=True, key=lambda x: x[0])
    return ranked[:top_n]

# -------------------- Prompt templates --------------------
def make_prompt(task, user_text, style, faq_context=""):
    if style == "Basic prompt":
        if task == "Question Answering":
            return f"Answer this question clearly:\n{user_text}\n\nFAQ context:\n{faq_context or 'No matching FAQ found.'}"
        if task == "Text Summarization":
            return f"Summarize this text:\n{user_text}"
        return f"Classify this text into one of these categories: Question, Complaint, Suggestion, General Enquiry.\nText: {user_text}"

    if task == "Question Answering":
        return f"""You are a helpful chatbot for an educational project.
TASK: Answer the user's question accurately and in simple language.
RULES:
1. Use the FAQ context when it is relevant.
2. Do not invent college-specific facts. If the context does not contain the answer, say what is unknown and give a general answer if possible.
3. Answer directly, then add brief explanation or steps if useful.
4. Keep the answer clear and well organised.

FAQ CONTEXT:
{faq_context or "No matching FAQ was found. Do not claim that the answer came from the FAQ dataset."}

USER QUESTION:
{user_text}

OUTPUT: A concise, helpful answer."""
    if task == "Text Summarization":
        return f"""You are an accurate text summarization assistant.
TASK: Summarize the input text.
RULES:
1. Preserve the main ideas, important facts, names, and numbers.
2. Do not add facts that are absent from the source.
3. Remove repetition and minor details.
4. Use simple English and 3–5 bullet points, followed by a one-sentence overall summary.
5. If the input is already very short, state the main idea without adding information.

INPUT TEXT:
{user_text}

OUTPUT FORMAT:
Key points:
- ...
Overall summary: ..."""
    return f"""You are a text classification assistant.
TASK: Assign the text to exactly ONE category from this list:
- Question
- Complaint
- Suggestion
- General Enquiry

CLASSIFICATION RULES:
- Question: primarily asks for information or an answer.
- Complaint: expresses a problem, dissatisfaction, or failure.
- Suggestion: proposes an idea or improvement.
- General Enquiry: general message that does not fit the other categories.
Choose the best category based on the main intent. Do not create new categories.
Examples:
Text: "Why is my login not working?" -> Question
Text: "The app keeps crashing and I am unhappy." -> Complaint
Text: "Please add a dark mode." -> Suggestion

TEXT TO CLASSIFY:
{user_text}

OUTPUT FORMAT:
Category: <one category>
Reason: <one short sentence>"""

def call_llm(prompt, temperature=0.2):
    start = time.time()
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "Follow the task instructions. Be truthful and do not invent unsupported facts."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
    )
    answer = response.choices[0].message.content or ""
    return answer.strip(), time.time() - start

def evaluate_response(task, original_input, answer):
    prompt = f"""Evaluate this chatbot response for a student project. Do not assume it is correct merely because it sounds confident.
Task: {task}
Original input: {original_input}
Chatbot response: {answer}

Give a score from 1 (poor) to 5 (excellent) for each criterion:
- Relevance: answers the requested task
- Correctness: appears accurate and does not add unsupported facts
- Clarity: easy to understand
- Completeness: covers the important requirements
Then give one short improvement suggestion.
Return exactly this structure:
Relevance: X/5
Correctness: X/5
Clarity: X/5
Completeness: X/5
Improvement suggestion: ..."""
    return call_llm(prompt, temperature=0.0)

# -------------------- Main interface --------------------
task = st.selectbox("Choose a task", ["Question Answering", "Text Summarization", "Text Classification"])
if task == "Text Summarization":
    placeholder = "Paste a long paragraph or article to summarize..."
elif task == "Text Classification":
    placeholder = "Enter a message to classify, e.g. 'Please add a dark mode to the app.'"
else:
    placeholder = "Ask a question. For FAQ support, try: How do I reset my password?"

user_text = st.text_area("Your input", height=180, placeholder=placeholder)
reference_answer = ""
if task == "Question Answering":
    st.caption(f"FAQ dataset loaded: {len(faq_df)} entries. The app uses simple keyword overlap to find relevant FAQ entries.")
if task == "Text Classification":
    st.caption("Categories: Question, Complaint, Suggestion, General Enquiry.")

col1, col2 = st.columns([1, 1])
with col1:
    generate = st.button("Generate Response", type="primary", use_container_width=True)
with col2:
    show_prompt = st.checkbox("Show prompt sent to the model", value=False)

if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "last_task" not in st.session_state:
    st.session_state.last_task = ""
if "last_time" not in st.session_state:
    st.session_state.last_time = 0.0

if generate:
    if not user_text.strip():
        st.error("Please enter some text first.")
    else:
        faq_matches = retrieve_faq(user_text, faq_df) if task == "Question Answering" else []
        faq_context = "\n".join([f"FAQ question: {q}\nFAQ answer: {a}" for _, q, a in faq_matches])
        prompt = make_prompt(task, user_text.strip(), prompt_style, faq_context)
        try:
            with st.spinner("Contacting the LLM API..."):
                answer, elapsed = call_llm(prompt)
            st.session_state.last_answer = answer
            st.session_state.last_prompt = prompt
            st.session_state.last_input = user_text.strip()
            st.session_state.last_task = task
            st.session_state.last_time = elapsed
            st.session_state.last_style = prompt_style
            st.session_state.last_faq_matches = faq_matches
        except Exception as exc:
            st.error(f"API request failed: {exc}")
            st.caption("Check your API key, internet connection, model name, and API account limits.")

if st.session_state.last_answer:
    st.divider()
    st.subheader("Generated Response")
    st.write(st.session_state.last_answer)
    st.caption(f"Task: {st.session_state.last_task} • Prompt mode: {st.session_state.get('last_style', '')} • Response time: {st.session_state.last_time:.2f} seconds")

    if st.session_state.get("last_faq_matches"):
        with st.expander("FAQ matches used"):
            for score, question, answer in st.session_state.last_faq_matches:
                st.markdown(f"**FAQ:** {question}")
                st.write(answer)
                st.caption(f"Keyword-overlap score: {score:.2f}")

    if show_prompt:
        with st.expander("Prompt sent to the model", expanded=True):
            st.code(st.session_state.last_prompt)

    st.subheader("Evaluate response quality")
    st.write("This is an LLM-assisted rubric, not a guarantee of factual correctness. Review important answers yourself.")
    if st.button("Evaluate Relevance, Correctness, Clarity and Completeness"):
        try:
            with st.spinner("Evaluating response..."):
                evaluation, eval_time = evaluate_response(
                    st.session_state.last_task,
                    st.session_state.last_input,
                    st.session_state.last_answer
                )
            st.markdown(evaluation)
            st.caption(f"Evaluation time: {eval_time:.2f} seconds")
        except Exception as exc:
            st.error(f"Evaluation failed: {exc}")

    st.subheader("Check response consistency")
    st.write("Generates a second response using the same prompt, then estimates textual similarity.")
    if st.button("Generate Second Response and Compare"):
        try:
            with st.spinner("Generating a second response..."):
                second, second_time = call_llm(st.session_state.last_prompt)
            similarity = SequenceMatcher(None, st.session_state.last_answer.lower(), second.lower()).ratio()
            st.markdown("**Second response**")
            st.write(second)
            st.metric("Text similarity between responses", f"{similarity * 100:.1f}%")
            st.caption("Similarity measures wording overlap, not factual correctness or semantic equivalence.")
        except Exception as exc:
            st.error(f"Consistency check failed: {exc}")

    with st.expander("Prompt improvement notes"):
        st.markdown("""
        - **Clear role and task:** tells the model what job to perform.
        - **Context:** supplies matching FAQ information for question answering.
        - **Constraints:** asks the model not to invent facts and to preserve source meaning.
        - **Output format:** requests bullets, labels, or a fixed response structure.
        - **Few-shot examples:** classification prompt includes examples of categories.
        - **Iteration:** compare Basic prompt and Improved prompt on the same test inputs.
        """)

st.divider()
st.subheader("Custom Query Dataset")
st.write("Use the included custom_queries.csv as a small test set. It contains example inputs and expected outputs for manual comparison.")
if not custom_df.empty:
    st.dataframe(custom_df, use_container_width=True, hide_index=True)
else:
    st.info("custom_queries.csv was not found or does not have the expected columns.")

st.caption("Mini Project: Chatbot Using LLM with Prompt Engineering | Python + Streamlit + LLM API")
