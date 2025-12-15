# compare-iflows-llm (extended version of sapci-iflow-compare)

This Streamlit application connects to two SAP Cloud Integration (CI/CPI) tenants, downloads the deployed iFlow design-time artifacts, extracts the .iflw XML, and performs a detailed element-by-element comparison.​
It detects functional differences by matching elements (primarily via shared id) and recursively comparing attributes, text, and structure, with additional BPMN-aware handling to make process-flow changes easier to understand.
For faster review, the app can optionally send only the generated diff to a selectable LLM provider (e.g., Gemini/OpenAI/Anthropic) to produce a concise, human-readable summary, while keeping the full iFlow XML local to the session.

---

## 🚀 Features

- Automated iFlow selection: uses RuntimeArtifacts + DesignTime APIs so users only provide host + iFlow name instead of hardcoding full artifact URLs.​
- Config by upload, not path: st.file_uploader lets users load a JSON config directly in the browser (works on Streamlit Cloud) instead of relying on local file paths.​
- Instant form prefill: st.session_state + st.rerun() apply config values immediately so all environment/auth fields are auto-populated in the UI.​
- Cleaner UX for two environments: explicit “Environment 1 / Environment 2” layout with host, iFlow name, display name, and auth grouped logically.
- Multi‑LLM support: pluggable provider selection (Gemini/OpenAI/Anthropic) with model presets and endpoint management instead of Gemini‑only.
- More robust HTTP handling: centralized helpers with timeouts, content‑type checks, and clearer error messages/logging.
- Better XML labeling: richer component labels (Content Modifier, Request Reply, Groovy Script, etc.) for more readable diffs.
- Stronger code structure: single, non-duplicated helpers, type hints, and clearer separation of concerns (runtime lookup, design-time download, diff, LLM) for easier maintenance and extension.

---

## 🛠️ Setup Instructions

### Option 1: Clone the repository

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jwanOo/compare-iflows-llm.git
   cd your-repo-name

### Option 2: Manual Copy
1. **Open Visual Studio Code (or your preferred editor)**
2. **Create a new file named iflow_compare.py**
3. **Copy and paste the code from this repository into iflow_compare.py**

### Install dependencies
1. ***Make sure you have Python 3.8+ installed***
2. ***Install required packages using pip:***
   ```bash
   pip3 install streamlit lxml requests
---
## ⚙️ Usage
1. ***Provide API details for both iFlows (manual entry or via config file)***
2. ***Enter your LLM API URL and Key for AI summarization***
3. ***Run the app:***
    ```bash
    python3 -m streamlit run iflow_compare.py
---
4. ***Click "Compare iFlows" to fetch, extract, and compare the XMLs***
5. ***View technical differences and download the raw report***
6. ***See the Gemini summary for a readable, grouped overview of changes***

## 📝 Configuration
You can use a JSON config file with the following structure:
```json
{
  "api1": {
    "name": "Source iFlow Name",
    "url": "https://source-api-url",
    "oauth_token_url": "https://source-oauth-url",
    "client_id": "source-client-id",
    "client_secret": "source-client-secret"
  },
  "api2": {
    "name": "Target iFlow Name",
    "url": "https://target-api-url",
    "oauth_token_url": "https://target-oauth-url",
    "client_id": "target-client-id",
    "client_secret": "target-client-secret"
  }
}
```

## 🤖 LLM Integration
1. ***The app sends the raw difference report to Google Gemini for summarization***
2. ***The summary is displayed in a separate section for easy review***

## 📄 License
MIT License

## 🙋 Support
For issues, suggestions, or contributions, please open an issue or pull request on GitHub.

