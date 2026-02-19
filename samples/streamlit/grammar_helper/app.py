"""
Grammar Helper - FabricFlow Streamlit UI

A QuillBot-inspired grammar correction and writing enhancement tool.
Powered by FabricFlow's grammar_helper workflow.
"""

import os
import sys

import streamlit as st

# Add common utilities to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "common"))
from client import StreamlitWorkflowClient
from style import apply_studio_theme

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Grammar Helper - FabricFlow", page_icon="✍️", layout="wide", initial_sidebar_state="expanded"
)

# Apply Studio theme
apply_studio_theme()

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "client" not in st.session_state:
    st.session_state.client = None
if "result" not in st.session_state:
    st.session_state.result = None
if "input_text" not in st.session_state:
    st.session_state.input_text = ""
if "correction_style" not in st.session_state:
    st.session_state.correction_style = "professional"

# ============================================================================
# SIDEBAR: SERVER CONNECTION & SETTINGS
# ============================================================================
with st.sidebar:
    st.header("⚙️ FabricFlow Settings")

    server_url = st.text_input(
        "Server URL",
        value=os.getenv("FABRICFLOW_URL", "http://127.0.0.1:8000"),
        help="URL of the running FabricFlow server",
    )

    if st.button("🔌 Test Connection", use_container_width=True):
        if st.session_state.client is None:
            st.session_state.client = StreamlitWorkflowClient(server_url)

        success, message = st.session_state.client.test_connection()
        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
            st.info("💡 Start the server with: `fabric run`")

    st.divider()

    # Workflow settings
    st.subheader("Workflow Options")

    st.caption("⏱️ **Execution Mode**")
    st.info("Async mode with progress tracking")

    timeout = st.slider(
        "Timeout (seconds)", min_value=60, max_value=600, value=300, step=30, help="Maximum execution time"
    )

# ============================================================================
# MAIN CONTENT: HEADER
# ============================================================================
st.title("✍️ Grammar Helper")
st.markdown(
    """
Transform your writing with AI-powered grammar correction and style suggestions.
Powered by **FabricFlow** workflow engine.

**Features:**
- ✅ Comprehensive grammar correction
- 📝 Detailed explanations of changes
- 💡 Style improvement suggestions
- 📊 Word count statistics
"""
)

st.divider()

# ============================================================================
# CORRECTION STYLE SELECTOR (QuillBot-inspired modes)
# ============================================================================
st.subheader("🎨 Correction Style")

# Define style options with emojis
style_options = {
    "professional": "💼 Professional",
    "standard": "📄 Standard",
    "casual": "😊 Casual",
    "academic": "🎓 Academic",
    "creative": "✨ Creative",
}

# Use radio buttons with horizontal layout for cleaner UX
selected_style = st.radio(
    "Choose your correction style",
    options=list(style_options.keys()),
    format_func=lambda x: style_options[x],
    horizontal=True,
    index=list(style_options.keys()).index(st.session_state.correction_style),
    label_visibility="collapsed",
)

# Update session state if changed
if selected_style != st.session_state.correction_style:
    st.session_state.correction_style = selected_style

st.divider()

# ============================================================================
# INPUT SECTION
# ============================================================================
st.subheader("📝 Your Text")

# Sample text loader - simplified to single button
col_input, col_sample = st.columns([4, 1])
with col_sample:
    if st.button("📋 Load Sample", use_container_width=True):
        st.session_state.input_text = """me and sarah was working on the project when suddenly the computer crash and we loosed all our work its very frustrating because we was almost finished the report that was suppose to be submited yesterday but we didnt have no backup so now we got to start over from scratch which is gonna take alot of time the manager he dont understand how much effort we already putted into this and between you and I i think there being unreasonable with the deadline everyone in team is stressing out and morale is pretty low we should of backed up the files more regular but hindsight is 20/20 right irregardless we need to focus on getting this done asap"""
        st.rerun()

input_text = st.text_area(
    "Enter or paste your text below",
    value=st.session_state.input_text,
    height=200,
    placeholder="Type or paste your text here...",
    help="Enter the text you want to check for grammar and style improvements",
    label_visibility="collapsed",
)

# Character and word count
if input_text:
    char_count = len(input_text)
    word_count = len(input_text.split())

    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        st.caption(f"📊 **{word_count:,}** words")
    with col3:
        st.caption(f"🔤 **{char_count:,}** characters")

# Advanced settings
with st.expander("🔧 Advanced Settings", expanded=False):
    st.caption("**Model Configuration**")

    col1, col2 = st.columns(2)

    with col1:
        model_repository = st.text_input(
            "Model Repository", value="TheBloke/Mistral-7B-Instruct-v0.2-GGUF", help="HuggingFace repository ID"
        )

        context_size = st.number_input(
            "Context Size", min_value=512, max_value=8192, value=4096, step=512, help="Maximum context length"
        )

    with col2:
        model_filename = st.text_input(
            "Model Filename", value="mistral-7b-instruct-v0.2.Q4_K_M.gguf", help="GGUF model filename"
        )

        thread_count = st.number_input(
            "Thread Count", min_value=1, max_value=32, value=8, step=1, help="CPU threads for inference"
        )

    max_tokens = st.slider(
        "Max Tokens", min_value=256, max_value=1024, value=512, step=64, help="Maximum output tokens per response"
    )

    temperature = st.slider(
        "Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Lower = more deterministic"
    )

# ============================================================================
# EXECUTION BUTTON & VALIDATION
# ============================================================================
st.divider()

# Validation
is_valid = bool(input_text and input_text.strip())
if not is_valid:
    st.warning("⚠️ Please enter some text to check")

# Execute button
if st.button("🚀 Check Grammar & Style", type="primary", disabled=not is_valid, use_container_width=True):
    # Initialize client if needed
    if st.session_state.client is None:
        st.session_state.client = StreamlitWorkflowClient(server_url)

    # Build inputs
    inputs = {
        "input_text": input_text,
        "correction_style": st.session_state.correction_style,
        "model_repository": model_repository,
        "model_filename": model_filename,
        "context_size": context_size,
        "thread_count": thread_count,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    # Execute workflow with progress bar
    result = st.session_state.client.execute_workflow_with_progress_bar(
        "grammar_helper", inputs=inputs, timeout=timeout
    )

    if result:
        st.session_state.result = result
        st.success("✅ Analysis complete!")
        st.rerun()

# ============================================================================
# RESULTS SECTION
# ============================================================================
if st.session_state.result:
    st.divider()
    st.header("✨ Results")

    # Safeguard: Extract outputs if not already done by client
    raw_result = st.session_state.result
    result = raw_result.get("outputs", raw_result) if isinstance(raw_result, dict) else raw_result

    # Statistics
    st.subheader("📊 Statistics")

    col1, col2, col3, col4 = st.columns(4)

    original_count = result.get("word_count_original", 0)
    corrected_count = result.get("word_count_corrected", 0)
    styled_count = result.get("word_count_styled", 0)

    with col1:
        st.metric("Original Words", original_count, help="Word count in original text")

    with col2:
        st.metric(
            "Corrected Words",
            corrected_count,
            delta=corrected_count - original_count if original_count else None,
            help="Word count after grammar corrections",
        )

    with col3:
        st.metric(
            "Styled Words",
            styled_count,
            delta=styled_count - original_count if original_count else None,
            help="Word count after tone adjustment",
        )

    with col4:
        st.metric("Correction Style", st.session_state.correction_style.title(), help="Style used for tone")

    st.divider()

    # Three-way comparison: Original → Corrected → Styled
    st.subheader("📋 Text Evolution")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📄 ORIGINAL**")
        st.text_area(
            "Original", value=result.get("original_text", ""), height=250, disabled=True, label_visibility="collapsed"
        )

    with col2:
        st.markdown("**✅ GRAMMAR CORRECTED**")
        st.text_area(
            "Corrected", value=result.get("corrected_text", ""), height=250, disabled=True, label_visibility="collapsed"
        )

    with col3:
        st.markdown("**✨ TONE ADJUSTED**")
        st.text_area(
            "Styled", value=result.get("styled_text", ""), height=250, disabled=True, label_visibility="collapsed"
        )

    st.divider()

    # Changes explained
    st.subheader("💡 Grammar Corrections")

    changes_explained = result.get("changes_explained", "No changes explained")
    st.markdown(changes_explained)

    st.divider()

    # Full summary report
    with st.expander("📋 View Full Report"):
        summary_report = result.get("summary_report", "No report available")
        st.text(summary_report)

    # ========================================================================
    # DOWNLOAD SECTION
    # ========================================================================
    st.divider()
    st.subheader("📥 Download Results")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Download corrected text
        st.download_button(
            label="✅ Corrected Text",
            data=result.get("corrected_text", ""),
            file_name="corrected_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col2:
        # Download styled text (primary output)
        st.download_button(
            label="✨ Styled Text",
            data=result.get("styled_text", ""),
            file_name="styled_text.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col3:
        # Download full report
        st.download_button(
            label="📋 Full Report",
            data=result.get("summary_report", ""),
            file_name="grammar_report.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with col4:
        # Download all results as JSON
        import json

        results_json = json.dumps(result, indent=2)
        st.download_button(
            label="📦 JSON",
            data=results_json,
            file_name="results.json",
            mime="application/json",
            use_container_width=True,
        )

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.caption("Powered by **FabricFlow** | Grammar Helper Workflow | QuillBot-inspired UI")
