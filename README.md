# Chatbot Using LLM with Prompt Engineering

This mini-project matches the report's main requirements:
- Question answering, text summarization, and text classification
- LLM API integration
- Custom query and FAQ datasets
- Basic vs improved prompt comparison
- LLM-assisted response-quality evaluation
- Second-response consistency check

## 1. Requirements
- Python 3.10 or newer recommended
- Internet connection
- API key for an OpenAI-compatible chat-completions API

## 2. Setup (Windows)
Open Command Prompt or terminal in this folder:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and enter your API key:
```text
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```
Keep `.env` private. Do not upload it to GitHub.

## 3. Run the app
```bash
streamlit run app.py
```
A browser tab should open with the chatbot interface.

## 4. How to test
1. Select a task from the dropdown.
2. Enter a question, long text, or message.
3. Choose Basic prompt or Improved prompt in the sidebar.
4. Click Generate Response.
5. Optionally display the prompt sent to the model.
6. Click the evaluation button to get rubric scores for relevance, correctness, clarity, and completeness.
7. Click the consistency button to generate a second answer and compare textual similarity.
8. Repeat the same test input using Basic prompt and Improved prompt and record the differences.

## 5. Dataset files
- `faq_dataset.csv`: example FAQ questions and answers. Edit it for your project domain.
- `custom_queries.csv`: small test set with expected outputs for manual comparison.

FAQ retrieval uses simple keyword overlap for demonstration. It is not a semantic search system. The LLM-assisted evaluator is also only a supporting check; review important results manually.

## 6. Important
- API usage may cost money depending on the provider and account.
- The selected model must be available to your API key.
- Never place a real API key directly in the Python source code.
