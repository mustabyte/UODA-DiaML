
import streamlit as st

THEMES = {
    "dark": {
        "background": "#0B1220",
        "sidebar": "#111D30",
        "card": "#19283D",
        "text": "#F1F5F9",
        "muted": "#94A3B8",
        "primary": "#2DD4BF",
        "secondary": "#38BDF8",
        "border": "#283B52"
    },
    "light": {
        "background": "#F6F8FB",
        "sidebar": "#FFFFFF",
        "card": "#FFFFFF",
        "text": "#14243A",
        "muted": "#64748B",
        "primary": "#0D9488",
        "secondary": "#38BDF8",
        "border": "#E2E8F0"
    }
}


def get_theme():
    mode = st.context.theme.type
    return THEMES.get(mode, THEMES["dark"])
