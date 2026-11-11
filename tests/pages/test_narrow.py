"""Every page fits a phone: nothing pushes the page wider than 400 px."""

import pytest
from page_helpers import documents_store

pytestmark = pytest.mark.pages

PHONE = {"width": 400, "height": 800}


def width(page) -> int:
    return page.evaluate("document.documentElement.scrollWidth")


@pytest.mark.parametrize(
    "path",
    ["/", "/c/c", "DOCS", "/chat", "/settings", "/lab", "/worker", "/eval-sets", "/e/c", "RUN"],
)
def test_no_page_is_wider_than_a_phone(page, seeded, path):
    page.set_viewport_size(PHONE)
    if path == "DOCS":
        documents_store(seeded, "narrow-docs")
    path = {"DOCS": "/c/narrow-docs", "RUN": f"/c/c/runs/{seeded.run_id}"}.get(path, path)
    page.goto(seeded.url + path)
    page.wait_for_function("document.getElementById('main').innerText.trim().length > 0")
    page.wait_for_load_state("networkidle")
    assert width(page) <= 400


def test_new_store_form_fits_a_phone(page, seeded):
    page.set_viewport_size(PHONE)
    page.goto(seeded.url + "/")
    page.get_by_text("+ New store").click()
    assert width(page) <= 400
