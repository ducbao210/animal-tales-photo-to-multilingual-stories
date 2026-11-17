"""Streamlit entry point for browsing generated stories."""
from __future__ import annotations

import os
from html import escape
from pathlib import Path

import streamlit as st

from src.display_ui.gallery import load_story_records


LANGUAGE_NAMES = {
    "en": "English",
    "fr-FR": "Français",
    "es-ES": "Español",
    "vi-VN": "Tiếng Việt",
}
DEFAULT_STORY_FONT_SIZE = 28


def _move_page(state_key: str, direction: int, page_count: int) -> None:
    current = st.session_state.get(state_key, 0)
    st.session_state[state_key] = max(0, min(current + direction, page_count - 1))


def _translated_text(record: dict, page: dict, locale: str) -> str:
    if locale == "en":
        return str(page.get("text", ""))
    translated_pages = record["translations"].get(locale, {}).get("pages", [])
    translated = next(
        (item for item in translated_pages if item.get("page") == page.get("page")),
        None,
    )
    return str(translated.get("text", page.get("text", ""))) if translated else str(page.get("text", ""))


def _render_story(record: dict) -> None:
    story = record["story"]
    pages = story.get("pages", [])
    if not pages:
        st.info("This output has no story pages.")
        return

    story_id = record["image_id"]
    page_key = f"page:{story_id}"
    language_key = f"language:{story_id}"
    page_index = min(st.session_state.get(page_key, 0), len(pages) - 1)
    st.session_state[page_key] = page_index

    st.subheader(story.get("title") or record["input_name"])
    details = " · ".join(
        value for value in (story.get("species"), story.get("color")) if value
    )
    if details:
        st.caption(details)
    if record["input_path"]:
        st.image(str(record["input_path"]), caption=f"Input: {record['input_name']}", width=180)

    available_languages = ["en"] + [
        locale for locale in record["translations"] if locale in LANGUAGE_NAMES
    ]
    current_language = st.session_state.get(language_key, "en")
    if current_language not in available_languages:
        current_language = "en"
    locale = st.selectbox(
        "Story language",
        available_languages,
        index=available_languages.index(current_language),
        format_func=lambda code: LANGUAGE_NAMES[code],
        key=language_key,
    )

    page = pages[page_index]
    phase = str(page.get("section", ""))
    phase_image = record["phase_images"].get(phase) or record["keyframe_path"]
    left, right = st.columns([1, 1])
    with left:
        if phase_image and Path(phase_image).is_file():
            st.image(str(phase_image), caption=phase.replace("_", " ").title())
        else:
            st.info("No character keyframe image is available.")
    with right:
        st.markdown(f"**{phase.replace('_', ' ').title()}**")
        st.markdown(f"### Page {page.get('page', page_index + 1)}")
        font_size = st.slider(
            "Text size",
            min_value=18,
            max_value=44,
            value=DEFAULT_STORY_FONT_SIZE,
            step=2,
            format="%d px",
            key=f"font-size:{story_id}",
        )
        page_text = escape(_translated_text(record, page, locale))
        st.markdown(
            f'<div style="font-size: {font_size}px; line-height: 1.45;">{page_text}</div>',
            unsafe_allow_html=True,
        )

    previous, count, next_page = st.columns([1, 2, 1])
    with previous:
        st.button(
            "‹ Previous",
            key=f"previous:{story_id}",
            disabled=page_index == 0,
            on_click=_move_page,
            args=(page_key, -1, len(pages)),
        )
    with count:
        st.caption(f"Page {page_index + 1} of {len(pages)}")
    with next_page:
        st.button(
            "Next ›",
            key=f"next:{story_id}",
            disabled=page_index >= len(pages) - 1,
            on_click=_move_page,
            args=(page_key, 1, len(pages)),
        )


def main() -> None:
    st.set_page_config(page_title="Animal Story Reader", page_icon="📖", layout="wide")
    st.title("Animal Story Reader")
    output_root = os.environ.get("STORY_PIPELINE_OUTPUT_ROOT", "outputs")
    project_root = os.environ.get("STORY_PIPELINE_PROJECT_ROOT", ".")
    records = load_story_records(output_root, project_root)
    if not records:
        st.info("No completed story outputs were found. Run the pipeline first.")
        return

    photo_tabs = st.tabs([record["input_name"] for record in records])
    for tab, record in zip(photo_tabs, records):
        with tab:
            _render_story(record)


main()
