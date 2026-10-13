"""Offline tests for link previews and first-paint browser branding."""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

MODULE = Path(__file__).parents[1] / "assets" / "brand_streamlit_shell.py"
spec = importlib.util.spec_from_file_location("brand_streamlit_shell", MODULE)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

VENDOR_SAMPLE = '''<!doctype html>
<html lang="en"><head>
<link rel="shortcut icon" href="./favicon.png" />
<title>Streamlit</title>
<meta charset="UTF-8" />
</head><body><div id="root"></div></body></html>'''


def test_head_contains_branded_title_description_og_twitter_and_favicon():
    branded = mod.brand_index_html(VENDOR_SAMPLE)
    assert '<title>Fourth Down AI</title>' in branded
    assert '<title>Streamlit</title>' not in branded
    assert '<meta property="og:title" content="Fourth Down AI" />' in branded
    assert 'name="twitter:title" content="Fourth Down AI"' in branded
    assert 'name="description"' in branded
    assert 'rel="icon"' in branded and 'href="/favicon.png?v=20261013-brand1"' in branded
    assert 'rel="apple-touch-icon"' in branded
    assert branded.count("<title>") == 1
    assert branded.count('property="og:title"') == 1
    assert '<div id="root"></div>' in branded


@pytest.mark.parametrize("unexpected", [
    '<html><head><title>Fourth Down AI</title></head></html>',
    '<html><head><title>Streamlit</title></head></html>',
    '<html><head><title>Streamlit</title><title>Streamlit</title>'
    '<link rel="icon" href="favicon.png" /></head></html>',
])
def test_unexpected_package_html_fails_closed(unexpected):
    with pytest.raises(ValueError):
        mod.brand_index_html(unexpected)


def test_patch_writes_png_and_brand_to_synthetic_installed_package(tmp_path):
    index = tmp_path / "index.html"
    index.write_text(VENDOR_SAMPLE, encoding="utf-8")
    output = mod.patch_streamlit_package(tmp_path)
    assert output == index
    assert 'Streamlit</title>' not in output.read_text(encoding="utf-8")
    with Image.open(tmp_path / "favicon.png") as icon:
        assert icon.size == (192, 192)
        assert icon.mode == "RGBA"
        assert icon.getpixel((96, 96))[3] == 255


def test_app_has_matching_dynamic_title():
    app_path = Path(__file__).parents[1] / "app.py"
    assert 'page_title="Fourth Down AI"' in app_path.read_text(encoding="utf-8")
