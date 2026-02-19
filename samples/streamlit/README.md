# FabricFlow Streamlit Applications

This directory contains Streamlit web applications for interacting with FabricFlow workflows through intuitive user interfaces.

## Overview

Streamlit UIs provide a user-friendly way to execute FabricFlow workflows without writing code or using the CLI. Each application is tailored to a specific workflow, offering:

- 🎨 **Studio-inspired design** matching FabricFlow Designer's look and feel
- ⚡ **Real-time execution** with progress tracking and status updates
- 📊 **Rich visualizations** for workflow outputs (text, images, metrics)
- 🔧 **Advanced settings** for power users in collapsible sections
- 📥 **Download options** for all generated outputs

## Directory Structure

```
streamlit/
├── README.md                  # This file
├── common/                    # Shared utilities
│   ├── style.py              # Studio CSS theme
│   └── client.py             # FabricFlow client wrapper
├── grammar_helper/           # Grammar correction UI
│   ├── app.py               # Main Streamlit application
```

## Prerequisites

### 1. FabricFlow Server

All Streamlit applications require a running FabricFlow server:

```bash
# Start the server (from solution-fabric-flow directory)
fabric run

# Or with custom host/port
fabric run --host 0.0.0.0 --port 8080
```

The server must be accessible at the URL configured in the Streamlit app (default: `http://127.0.0.1:8000`).

### 2. Python Environment

Streamlit apps require Python 3.12+ with the following packages:

```bash
# Install Streamlit and FabricFlow SDK
pip install streamlit fabricflow

# Or using uv (recommended)
uv pip install streamlit fabricflow
```

## Running Applications

### General Pattern

```bash
# Navigate to the specific app directory
cd samples/streamlit/grammar_helper

# Run the Streamlit app
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`.

### Environment Variables

Configure the FabricFlow server URL using an environment variable:

```bash
# Windows (PowerShell)
$env:FABRICFLOW_URL = "http://localhost:8000"
streamlit run app.py

# Linux/macOS
export FABRICFLOW_URL="http://localhost:8000"
streamlit run app.py
```

If not set, apps default to `http://127.0.0.1:8000`.

### Configuration Options

Streamlit apps can be configured via `.streamlit/config.toml`:

```toml
[server]
port = 8501
headless = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#2563eb"      # FabricFlow primary blue
backgroundColor = "#f8fafc"
secondaryBackgroundColor = "#ffffff"
textColor = "#1e293b"
font = "sans serif"
```

## Available Applications

### Grammar Helper

**Path:** `grammar_helper/`
**Workflow:** `grammar_helper.json`

A QuillBot-inspired grammar correction and writing enhancement tool.

**Features:**
- Text input with character/word counters
- Multiple correction styles (professional, casual, academic)
- Side-by-side comparison of original and corrected text
- Detailed explanations of changes
- Style improvement suggestions
- Downloadable results

**Usage:**
```bash
cd grammar_helper
streamlit run app.py
```

See [grammar_helper/app.py](grammar_helper/app.py) for implementation details.

## Development Guide

### Creating a New Streamlit UI

Follow these steps to create a Streamlit UI for any FabricFlow workflow:

#### 1. Read the Copilot Instructions

Review [.github/instructions/streamlit-ui-generation.instructions.md](../.github/instructions/streamlit-ui-generation.instructions.md) for comprehensive guidelines on:
- Workflow JSON parsing
- Input widget mapping
- Output display logic
- Studio styling
- FabricFlow SDK integration

#### 2. Analyze the Workflow

```bash
# List available workflows
fabric workflow list

# View workflow definition
cat workflows/your_workflow.json
```

Identify:
- **Inputs:** Types, defaults, required fields
- **Outputs:** Types, display requirements
- **Complexity:** Fast (<5s) or slow (>5s) execution

#### 3. Use the Copilot Prompt

Ask GitHub Copilot to generate the UI:

```
Generate a Streamlit UI for the {workflow_name} workflow following
the streamlit-ui-generation.instructions.md guidelines.
```

#### 4. Customize and Test

- Adjust input widgets for better UX
- Add workflow-specific features (e.g., presets, examples)
- Test with various inputs
- Optimize progress feedback

#### 5. Document

Create a README.md for your app with:
- Brief description
- Installation instructions
- Usage examples
- Screenshots (optional)

### Using Shared Utilities

Import common utilities to avoid code duplication:

```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))

from style import apply_studio_theme
from client import StreamlitWorkflowClient, handle_api_error

# Apply Studio styling
apply_studio_theme()

# Use wrapped client
client = StreamlitWorkflowClient(server_url)
```

## Troubleshooting

### "Cannot connect to FabricFlow server"

**Solution:**
1. Verify the server is running: `fabric run`
2. Check the URL in the Streamlit app sidebar
3. Test connectivity: `curl http://127.0.0.1:8000/health`

### "Workflow not found"

**Solution:**
1. List available workflows: `fabric workflow list`
2. Verify workflow name matches exactly
3. Check workflow is in the `workflows/` directory

### "Execution timeout"

**Solution:**
1. Increase timeout in the app code (default: 300s)
2. Check server logs for errors
3. Verify workflow completes successfully via CLI: `fabric workflow run <name>`

### "Module not found: fabricflow"

**Solution:**
```bash
# Install the FabricFlow SDK
pip install fabricflow

# Or from local development
cd ../..  # Navigate to solution-fabric-flow root
pip install -e .
```

### Page doesn't refresh after execution

**Solution:**
- Ensure `st.rerun()` is called after storing results in `st.session_state`
- Clear session state: Delete `~/.streamlit/` directory

## Best Practices

### 1. Always Validate Inputs

```python
def validate_inputs():
    if not input_text.strip():
        st.warning("⚠️ Please enter text to process")
        return False
    if len(input_text) > 10000:
        st.error("❌ Text exceeds maximum length (10,000 characters)")
        return False
    return True
```

### 2. Handle Errors Gracefully

```python
try:
    result = client.workflow.run("workflow_name", inputs=inputs)
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    st.info("Make sure the FabricFlow server is running: `fabric run`")
```

### 3. Provide Progress Feedback

```python
with st.spinner("Processing..."):
    progress_bar = st.progress(0)
    for i in range(100):
        progress_bar.progress(i + 1)
        time.sleep(0.01)
```

### 4. Use Session State

```python
if "result" not in st.session_state:
    st.session_state.result = None

if st.button("Run"):
    st.session_state.result = execute_workflow()
    st.rerun()
```

### 5. Cleanup Resources

```python
try:
    result = execute_workflow()
finally:
    if instance_guid:
        client.instance.terminate(instance_guid)
```

## Performance Tips

1. **Caching:** Use `@st.cache_data` for expensive computations
   ```python
   @st.cache_data
   def load_workflow_schema(workflow_name: str):
       return client.workflow.get(workflow_name)
   ```

2. **Connection Pooling:** Reuse client instances
   ```python
   if "client" not in st.session_state:
       st.session_state.client = Client(server_url)
   ```

3. **Async Execution:** Use async mode for workflows >5s
   ```python
   result = client.instance.execute(
       instance_guid,
       inputs=inputs,
       mode="async"  # Prevents blocking
   )
   ```

## Security Considerations

1. **Input Sanitization:** Validate and sanitize all user inputs
2. **File Uploads:** Limit file sizes and types
3. **API Keys:** Use environment variables, never hardcode
4. **HTTPS:** Use TLS for production deployments
5. **Sessions:** Implement user sessions for multi-user environments

## Contributing

When contributing a new Streamlit UI:

1. Follow the [streamlit-ui-generation.instructions.md](../../.github/instructions/streamlit-ui-generation.instructions.md) guidelines
2. Use the shared utilities in `common/`
3. Include comprehensive documentation
4. Test with various input scenarios
5. Ensure Studio design consistency

## Resources

- **Streamlit Documentation:** https://docs.streamlit.io
- **FabricFlow CLI Guide:** [../../docs/CLI.md](../../docs/CLI.md)
- **FabricFlow REST API:** [../../docs/REST_API_GUIDE.md](../../docs/REST_API_GUIDE.md)
- **FabricFlow Python SDK:** [../../docs/PYTHON_API.md](../../docs/PYTHON_API.md)
- **Copilot Instructions:** [../../.github/instructions/copilot-instructions.md](../../.github/instructions/copilot-instructions.md)

## License

See the main FabricFlow repository for license information.
