"""
Studio-aligned styling for Streamlit applications.

This module provides minimal CSS overrides to align Streamlit UIs with the
FabricFlow Studio design system. It imports the Inter font family and applies
core accent colors from the Studio palette.

Design Philosophy:
- Minimal overrides only - rely on Streamlit's native theming
- Match Studio's light mode color palette
- Import Inter font family for typography consistency
- Never override core component styles heavily (causes contrast issues)
"""

import streamlit as st


def apply_studio_theme():
    """
    Apply minimal Studio theme styling to match FabricFlow Studio.

    Uses light mode palette:
    - Primary: #2563eb (blue-600)
    - Background: #f8fafc (slate-50)
    - Surface: #ffffff
    - Text: #1e293b (slate-800)
    - Border: #e2e8f0 (slate-200)

    Only overrides:
    - Font family (Inter)
    - Primary button colors
    - Heading font weights
    - Message border colors
    """
    st.markdown(
        """
        <style>
        /* Import Inter font */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        /* Apply Inter to all text */
        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Headings font-weight */
        h1, h2, h3, h4, h5, h6 {
            font-weight: 600;
        }
        
        /* Primary button color to match Studio */
        .stButton > button[kind="primary"] {
            background-color: #2563eb;
            border-color: #2563eb;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #1d4ed8;
            border-color: #1d4ed8;
        }
        
        /* Message borders to match Studio */
        .stSuccess {
            border-left-color: #10b981;
        }
        
        .stInfo {
            border-left-color: #2563eb;
        }
        
        .stWarning {
            border-left-color: #f59e0b;
        }
        
        .stError {
            border-left-color: #ef4444;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )
