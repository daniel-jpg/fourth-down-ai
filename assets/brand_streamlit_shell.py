"""Brand the initial HTTP page shell for the self-hosted Fourth Down AI demo.

Streamlit's set_page_config updates title/icon only after client startup; link
preview bots consume the unmodified static HTML title and crown favicon.
Patch the pinned Streamlit package's STATIC HTML during Docker BUILD, not at
request time. No framework internals or model/inference code are changed.
"""
from __future__ import annotations

from html import escape
from pathlib import Path
import re


TITLE = "Fourth Down AI"
DESCRIPTION = (
    "NFL fourth-down decision support with model-based win probabilities "
    "for going for it, kicking a field goal, or punting."
)
URL = "https://fourth-down-ai-nfl.onrender.com"
FAVICON_PATH = "/favicon.png?v=20261013-brand1"

_TITLE_PATTERN = re.compile(r"<title>\s*Streamlit\s*</title>", re.IGNORECASE)
_FAVICON_PATTERN = re.compile(
    r'<link\b(?=[^>]*\brel\s*=\s*[\x27"](?:shortcut\s+)?icon[\x27"])'
    r'(?=[^>]*\bhref\s*=\s*[\x27"][^\x27"]*favicon(?:\.png|\.ico)[^\x27"]*[\x27"])'
    r'[^>]*>',
    re.IGNORECASE,
)


def brand_index_html(original: str) -> str:
    """Fail closed if the pinned vendor HTML shape changes."""
    if len(_TITLE_PATTERN.findall(original)) != 1:
        raise ValueError("Streamlit upstream HTML title has changed; check pinned version")
    if len(_FAVICON_PATTERN.findall(original)) != 1:
        raise ValueError("Streamlit upstream favicon link has changed; check pinned version")
    title = escape(TITLE, quote=True)
    description = escape(DESCRIPTION, quote=True)
    initial_meta = (
        f"<title>{title}</title>\n"
        f'<meta name="application-name" content="{title}" />\n'
        f'<meta name="description" content="{description}" />\n'
        '<meta property="og:type" content="website" />\n'
        f'<meta property="og:url" content="{URL}/" />\n'
        f'<meta property="og:site_name" content="{title}" />\n'
        f'<meta property="og:title" content="{title}" />\n'
        f'<meta property="og:description" content="{description}" />\n'
        '<meta name="twitter:card" content="summary" />\n'
        f'<meta name="twitter:title" content="{title}" />\n'
        f'<meta name="twitter:description" content="{description}" />'
    )
    branded = _TITLE_PATTERN.sub(lambda _match: initial_meta, original, count=1)
    branded = _FAVICON_PATTERN.sub(
        f'<link rel="icon" type="image/png" sizes="192x192" href="{FAVICON_PATH}" />\n'
        f'<link rel="apple-touch-icon" href="{FAVICON_PATH}" />',
        branded,
        count=1,
    )
    if '<title>Streamlit</title>' in branded:
        raise ValueError("Default browser title remains in HTML")
    return branded


def write_football_favicon(destination: Path) -> None:
    """Draw the project's own transparent, high-resolution football icon."""
    from PIL import Image, ImageDraw

    dimensions = 192
    background = Image.new("RGBA", (dimensions, dimensions), (0, 0, 0, 0))
    draw = ImageDraw.Draw(background)
    draw.ellipse((5, 5, 186, 186), fill="#104C35", outline="#B0DF96", width=5)
    # Draw the football at high resolution, rotate as a single composited object.
    ball = Image.new("RGBA", (160, 100), (0, 0, 0, 0))
    pen = ImageDraw.Draw(ball)
    pen.ellipse((11, 17, 149, 83), fill="#A8572D", outline="#F9F7EC", width=5)
    pen.arc((28, 23, 66, 78), -75, 75, fill="#F9F7EC", width=6)
    pen.arc((94, 23, 132, 78), 105, 255, fill="#F9F7EC", width=6)
    pen.line((62, 50, 100, 50), fill="#F9F7EC", width=6)
    for x in (68, 80, 92):
        pen.line((x, 40, x, 60), fill="#F9F7EC", width=4)
    tilted = ball.rotate(-27, resample=Image.Resampling.BICUBIC, expand=True)
    background.alpha_composite(tilted, ((192 - tilted.width) // 2, (192 - tilted.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    background.save(destination, format="PNG", optimize=True)


def patch_streamlit_package(static_dir: Path) -> Path:
    """Write branded index and favicon to the installed Streamlit static dir."""
    index = static_dir / "index.html"
    favicon = static_dir / "favicon.png"
    original = index.read_text(encoding="utf-8")
    result = brand_index_html(original)  # Validate BEFORE changing files.
    write_football_favicon(favicon)
    index.write_text(result, encoding="utf-8")
    assert favicon.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert f"<title>{TITLE}</title>" in index.read_text(encoding="utf-8")
    return index


def main() -> None:
    import streamlit
    static_dir = Path(streamlit.__file__).resolve().parent / "static"
    if not static_dir.is_dir():
        raise SystemExit(f"Streamlit static directory missing: {static_dir}")
    patched = patch_streamlit_package(static_dir)
    print("FOURTH_DOWN_BRAND_SHELL_OK", patched)


if __name__ == "__main__":
    main()
