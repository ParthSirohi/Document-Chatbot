# Run the Document Chatbot (Windows)

Quick steps to open the project in VS Code / CMD and run the chatbot.

1) Open terminal in project root (the folder containing this file).

2) (Optional but recommended) Create and activate a virtual environment:

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Cmd.exe:

```cmd
python -m venv .venv
.\.venv\Scripts\activate.bat
```

3) Install required Python packages:

```powershell
pip install -r requirements.txt
```

4) (Choose one LLM option)

- Option A — Ollama (local, recommended):

  - Install Ollama from https://ollama.ai
  - Start Ollama server:

  ```powershell
  ollama serve
  ```

  - (Optional) Pull a model (one-time):

  ```powershell
  ollama pull mistral
  ```

- Option B — Groq (cloud):

  - Set your Groq API key in PowerShell:

  ```powershell
  $env:GROQ_API_KEY = "your_key_here"
  ```

5) Run the Streamlit app (chat UI):

```powershell
streamlit run app_simple.py
```

6) Open the app in your browser (Streamlit prints the URL):

http://localhost:8501

Notes:
- On first run the app may build the FAISS index from PDFs (pdf_data/). This can take 1–2 minutes.
- If you get "Cannot connect to Ollama", make sure `ollama serve` is running.
- To stop the app: press Ctrl+C in the terminal.

If you'd like, I can now remove documentation files (markdowns) that are not required to run the chatbot — I'll show a suggested list and wait for your confirmation before deleting anything.
