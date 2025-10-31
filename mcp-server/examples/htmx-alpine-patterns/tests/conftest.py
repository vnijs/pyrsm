"""
Pytest configuration and fixtures for Playwright tests
"""
import pytest
from playwright.sync_api import Page, expect
from django.contrib.staticfiles.testing import StaticLiveServerTestCase


@pytest.fixture(scope="session")
def django_db_setup():
    """Set up test database"""
    pass


@pytest.fixture(scope="function")
def clear_local_storage(page: Page):
    """Clear localStorage before each test"""
    # Note: localStorage can only be cleared after navigating to a page
    # Tests should call this after page.goto()
    page.context.clear_cookies()
    yield
    # Cleanup after test - navigate to ensure we can clear
    try:
        if page.url and not page.url.startswith("about:blank"):
            page.evaluate("localStorage.clear()")
    except:
        pass  # Ignore errors if page is closed


@pytest.fixture(scope="function")
def console_messages(page: Page):
    """Capture console messages for debugging"""
    messages = []

    def handle_console(msg):
        messages.append({
            'type': msg.type,
            'text': msg.text,
            'location': msg.location
        })

    page.on("console", handle_console)
    yield messages


@pytest.fixture(scope="function")
def wait_for_alpine(page: Page):
    """Wait for Alpine.js to initialize"""
    def wait():
        page.wait_for_function("() => window.Alpine !== undefined")
        page.wait_for_timeout(100)  # Small delay for Alpine to fully initialize
    return wait


@pytest.fixture(scope="function")
def wait_for_htmx(page: Page):
    """Wait for HTMX to complete request"""
    def wait():
        page.wait_for_function("() => !document.body.classList.contains('htmx-request')")
    return wait


@pytest.fixture
def get_local_storage(page: Page):
    """Helper to get localStorage value"""
    def get(key: str):
        return page.evaluate(f"localStorage.getItem('{key}')")
    return get


@pytest.fixture
def set_local_storage(page: Page):
    """Helper to set localStorage value"""
    def set(key: str, value: str):
        page.evaluate(f"localStorage.setItem('{key}', '{value}')")
    return set


@pytest.fixture
def clear_storage(page: Page):
    """Helper to clear localStorage after page navigation"""
    def clear():
        try:
            page.evaluate("localStorage.clear()")
        except:
            pass
    return clear


@pytest.fixture(autouse=True)
def setup_page(page: Page):
    """Auto-setup for each test - navigate and clear storage"""
    # This fixture runs automatically before each test
    # But doesn't do anything by default - tests control navigation
    yield
    # Cleanup
    try:
        if page.url and not page.url.startswith("about:blank"):
            page.evaluate("localStorage.clear()")
    except:
        pass


# Custom assertions
def assert_dropdown_excludes(page: Page, dropdown_selector: str, excluded_value: str):
    """Assert that a dropdown does NOT contain a specific value"""
    options = page.locator(f"{dropdown_selector} option").all_text_contents()
    assert excluded_value not in options, f"Expected '{excluded_value}' to be excluded from dropdown"


def assert_dropdown_includes(page: Page, dropdown_selector: str, included_value: str):
    """Assert that a dropdown DOES contain a specific value"""
    options = page.locator(f"{dropdown_selector} option").all_text_contents()
    assert included_value in options, f"Expected '{included_value}' to be in dropdown"


# Export custom assertions
pytest.assert_dropdown_excludes = assert_dropdown_excludes
pytest.assert_dropdown_includes = assert_dropdown_includes
