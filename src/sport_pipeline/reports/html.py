"""Small HTML helpers for dependency-free static reports."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSS_PATH = REPO_ROOT / "reports" / "templates" / "static_report_v1.css"


def html_escape(value: object) -> str:
    """Escape a scalar for HTML display."""

    if value is None:
        return ""
    return escape(str(value), quote=True)


def load_report_css(css_path: str | Path | None = None) -> str:
    """Load the bundled static report CSS template."""

    path = Path(css_path) if css_path is not None else DEFAULT_CSS_PATH
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def render_table(headers: Sequence[str], rows: Iterable[Mapping[str, object] | Sequence[object]]) -> str:
    """Render a compact HTML table from mappings or sequences."""

    head_cells = "".join(f"<th>{html_escape(header)}</th>" for header in headers)
    body_rows = []
    for row in rows:
        if isinstance(row, Mapping):
            values = [row.get(header, "") for header in headers]
        else:
            values = list(row)
        cells = "".join(f"<td>{html_escape(value)}</td>" for value in values)
        body_rows.append(f"<tr>{cells}</tr>")
    if not body_rows:
        body_rows.append(f"<tr><td colspan=\"{len(headers)}\">No rows</td></tr>")
    return f"<table><thead><tr>{head_cells}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def render_kv_table(rows: Mapping[str, object]) -> str:
    """Render key/value metadata."""

    rendered_rows = [{"key": key, "value": value} for key, value in rows.items()]
    return render_table(("key", "value"), rendered_rows)


def render_cards(cards: Iterable[Mapping[str, object]]) -> str:
    """Render failure browser cards."""

    rendered = []
    for card in cards:
        title = html_escape(card.get("title", "failure case"))
        fields = {
            key: value
            for key, value in card.items()
            if key not in {"title", "links"} and value not in (None, "")
        }
        body = render_kv_table(fields)
        links = []
        for label, url in card.get("links", []):
            links.append(f"<a href=\"{html_escape(url)}\">{html_escape(label)}</a>")
        link_html = f"<p class=\"links\">{' '.join(links)}</p>" if links else ""
        rendered.append(f"<article class=\"case-card\"><h3>{title}</h3>{body}{link_html}</article>")
    if not rendered:
        return "<p>No failure cases selected.</p>"
    return "<div class=\"case-grid\">" + "".join(rendered) + "</div>"


def nav_links(run_id: str) -> str:
    """Return relative navigation between Phase 9 report pages."""

    links = (
        ("Pipeline dashboard", "../../pipeline_dashboard/{run_id}/index.html"),
        ("Target availability", "../../target_availability/{run_id}/index.html"),
        ("Experiment compare", "../../experiment_compare/{run_id}/index.html"),
        ("Failure browser", "../../failure_browser/{run_id}/index.html"),
        ("Clip quality", "../../clip_quality/{run_id}/index.html"),
    )
    items = "".join(
        f"<a href=\"{template.format(run_id=html_escape(run_id))}\">{html_escape(label)}</a>"
        for label, template in links
    )
    return f"<nav>{items}</nav>"


def render_page(
    title: str,
    run_id: str,
    sections: Sequence[tuple[str, str]],
    *,
    subtitle: str | None = None,
    css: str | None = None,
) -> str:
    """Render a complete static HTML report page."""

    css_text = load_report_css() if css is None else css
    section_html = "".join(
        f"<section><h2>{html_escape(section_title)}</h2>{section_body}</section>"
        for section_title, section_body in sections
    )
    subtitle_html = f"<p class=\"subtitle\">{html_escape(subtitle)}</p>" if subtitle else ""
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n"
        "<head>\n"
        "  <meta charset=\"utf-8\">\n"
        "  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"  <title>{html_escape(title)}</title>\n"
        f"  <style>{css_text}</style>\n"
        "</head>\n"
        "<body>\n"
        "  <main>\n"
        f"    <header><p class=\"eyebrow\">run_id: {html_escape(run_id)}</p><h1>{html_escape(title)}</h1>{subtitle_html}{nav_links(run_id)}</header>\n"
        f"    {section_html}\n"
        "  </main>\n"
        "</body>\n"
        "</html>\n"
    )


def write_page(path: str | Path, html: str) -> Path:
    """Write an HTML page, creating parent directories."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
