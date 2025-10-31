"""
Playwright tests for Dependent Dropdowns pattern

Tests the key reactive pattern: variable selected in dropdown A
cannot appear in dropdown B, with full state restoration.
"""
import pytest
import json
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:8000"
PATTERN_URL = f"{BASE_URL}/patterns/dependent-dropdowns/"


@pytest.mark.ui
class TestDependentDropdowns:
    """Test suite for dependent dropdowns pattern"""

    def test_page_loads_successfully(self, page: Page, clear_local_storage):
        """Verify page loads and Alpine component initializes"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        # Clear localStorage after navigation
        page.evaluate("localStorage.clear()")

        # Check page title
        expect(page).to_have_title("Dependent Dropdowns Example")

        # Check main heading
        heading = page.locator("h2")
        expect(heading).to_contain_text("Dependent Dropdowns")

        # Verify dataset dropdown exists
        dataset_select = page.locator("select").first
        expect(dataset_select).to_be_visible()

    def test_selecting_dataset_shows_variable_dropdowns(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that selecting dataset reveals X and Y variable dropdowns"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Initially, variable dropdowns should be hidden
        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        # Should not be visible initially
        expect(x_select).not_to_be_visible()
        expect(y_select).not_to_be_visible()

        # Select dataset
        dataset_select = page.locator("select").first
        dataset_select.select_option("salary")

        # Wait for Alpine reactivity
        page.wait_for_timeout(100)

        # Variable dropdowns should now be visible
        expect(x_select).to_be_visible()
        expect(y_select).to_be_visible()

    def test_x_selection_excludes_from_y_dropdown(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that selecting X variable excludes it from Y dropdown"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Select dataset
        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(100)

        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        # Get Y options before selection
        y_options_before = y_select.locator("option").all_text_contents()
        assert "salary" in y_options_before, "salary should be available in Y"

        # Select "salary" in X dropdown
        x_select.select_option("salary")
        page.wait_for_timeout(100)

        # Get Y options after selection
        y_options_after = y_select.locator("option").all_text_contents()

        # "salary" should no longer be in Y dropdown
        assert "salary" not in y_options_after, "salary should be excluded from Y"

        # But other options should still be there
        assert "rank" in y_options_after, "rank should still be available"

    def test_y_selection_excludes_from_x_dropdown(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that selecting Y variable excludes it from X dropdown"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Select dataset
        page.locator("select").first.select_option("diamonds")
        page.wait_for_timeout(100)

        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        # Select "price" in Y dropdown
        y_select.select_option("price")
        page.wait_for_timeout(100)

        # Get X options after Y selection
        x_options = x_select.locator("option").all_text_contents()

        # "price" should not be in X dropdown
        assert "price" not in x_options, "price should be excluded from X"

        # But other options should be there
        assert "carat" in x_options, "carat should still be available"

    def test_both_selections_mutual_exclusion(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test mutual exclusion when both X and Y are selected"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(100)

        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        # Select X
        x_select.select_option("salary")
        page.wait_for_timeout(100)

        # Select Y
        y_select.select_option("rank")
        page.wait_for_timeout(100)

        # Verify exclusions
        x_options = x_select.locator("option").all_text_contents()
        y_options = y_select.locator("option").all_text_contents()

        assert "rank" not in x_options, "rank should be excluded from X"
        assert "salary" not in y_options, "salary should be excluded from Y"

    def test_clearing_selection_restores_options(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that clearing a selection restores excluded options"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(100)

        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        # Select X
        x_select.select_option("salary")
        page.wait_for_timeout(100)

        # Verify exclusion
        y_options_with_x = y_select.locator("option").all_text_contents()
        assert "salary" not in y_options_with_x

        # Clear X selection
        x_select.select_option("")  # Empty option
        page.wait_for_timeout(100)

        # Verify salary is back in Y dropdown
        y_options_after_clear = y_select.locator("option").all_text_contents()
        assert "salary" in y_options_after_clear

    def test_state_persists_to_localstorage(
        self, page: Page, clear_local_storage, wait_for_alpine, get_local_storage
    ):
        """Test that selections are saved to localStorage"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Make selections
        page.locator("select").first.select_option("diamonds")
        page.wait_for_timeout(100)

        page.locator("select").nth(1).select_option("carat")
        page.wait_for_timeout(100)

        page.locator("select").nth(2).select_option("price")
        page.wait_for_timeout(100)

        # Check localStorage
        stored_state = get_local_storage("dependentDropdowns")
        assert stored_state is not None, "State should be saved to localStorage"

        state = json.loads(stored_state)
        assert state["dataset"] == "diamonds"
        assert state["xVariable"] == "carat"
        assert state["yVariable"] == "price"

    def test_state_restores_after_page_reload(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that state fully restores after page reload"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Make selections
        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(100)

        page.locator("select").nth(1).select_option("discipline")
        page.wait_for_timeout(100)

        page.locator("select").nth(2).select_option("salary")
        page.wait_for_timeout(100)

        # Reload page
        page.reload()
        wait_for_alpine()
        page.wait_for_timeout(200)  # Extra time for restoration

        # Verify all selections restored
        dataset_select = page.locator("select").first
        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        expect(dataset_select).to_have_value("salary")
        expect(x_select).to_have_value("discipline")
        expect(y_select).to_have_value("salary")

    def test_state_restoration_maintains_exclusions(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that exclusions are correct after state restoration"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Setup state
        page.locator("select").first.select_option("iris")
        page.wait_for_timeout(100)

        page.locator("select").nth(1).select_option("sepal_length")
        page.wait_for_timeout(100)

        page.locator("select").nth(2).select_option("petal_length")
        page.wait_for_timeout(100)

        # Reload
        page.reload()
        wait_for_alpine()
        page.wait_for_timeout(200)

        # Check that exclusions are maintained
        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        x_options = x_select.locator("option").all_text_contents()
        y_options = y_select.locator("option").all_text_contents()

        assert "petal_length" not in x_options, "petal_length should be excluded from X"
        assert "sepal_length" not in y_options, "sepal_length should be excluded from Y"

    def test_switching_datasets_clears_variables(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that changing dataset clears variable selections"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Select dataset and variables
        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(100)

        page.locator("select").nth(1).select_option("salary")
        page.wait_for_timeout(100)

        page.locator("select").nth(2).select_option("rank")
        page.wait_for_timeout(100)

        # Switch dataset
        page.locator("select").first.select_option("diamonds")
        page.wait_for_timeout(100)

        # Variables should be cleared
        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        expect(x_select).to_have_value("")
        expect(y_select).to_have_value("")

    def test_clear_button_resets_everything(
        self, page: Page, clear_local_storage, wait_for_alpine, get_local_storage
    ):
        """Test that Clear All button resets state completely"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Make selections
        page.locator("select").first.select_option("diamonds")
        page.wait_for_timeout(100)

        page.locator("select").nth(1).select_option("carat")
        page.wait_for_timeout(100)

        # Click Clear All button
        clear_button = page.get_by_role("button", name="Clear All")
        clear_button.click()
        page.wait_for_timeout(100)

        # Verify everything is cleared
        dataset_select = page.locator("select").first
        expect(dataset_select).to_have_value("")

        # Verify localStorage is cleared
        stored = get_local_storage("dependentDropdowns")
        assert stored is None, "localStorage should be cleared"

    def test_summary_appears_with_both_selections(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that analysis summary appears when both variables selected"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(100)

        # Summary should not be visible yet
        summary = page.locator(".alert-info")
        expect(summary).not_to_be_visible()

        # Select X
        page.locator("select").nth(1).select_option("yrs_service")
        page.wait_for_timeout(100)

        # Still not visible
        expect(summary).not_to_be_visible()

        # Select Y
        page.locator("select").nth(2).select_option("salary")
        page.wait_for_timeout(100)

        # Now summary should appear
        expect(summary).to_be_visible()
        expect(summary).to_contain_text("yrs_service")
        expect(summary).to_contain_text("salary")

    def test_console_logs_state_management(
        self, page: Page, clear_local_storage, wait_for_alpine, console_messages
    ):
        """Test that component logs state management events"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(200)

        # Check for initialization log
        log_texts = [msg['text'] for msg in console_messages]

        # Should see initialization message
        assert any('initialized' in text.lower() for text in log_texts), \
            "Should log component initialization"

        # Should see state save message
        assert any('saved' in text.lower() or 'saving' in text.lower() for text in log_texts), \
            "Should log state saving"


@pytest.mark.ui
@pytest.mark.slow
class TestDependentDropdownsEdgeCases:
    """Edge case tests for dependent dropdowns"""

    def test_rapid_selection_changes(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test rapid selection changes don't break exclusion logic"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        page.locator("select").first.select_option("salary")
        page.wait_for_timeout(50)

        x_select = page.locator("select").nth(1)
        y_select = page.locator("select").nth(2)

        # Rapidly change selections
        for var in ["salary", "rank", "discipline", "yrs_service"]:
            x_select.select_option(var)
            page.wait_for_timeout(20)

        # Check final state is consistent
        x_value = x_select.input_value()
        y_options = y_select.locator("option").all_text_contents()

        if x_value:
            assert x_value not in y_options, "X selection should be excluded from Y"

    def test_back_button_restores_state(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that browser back button maintains state"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Make selections
        page.locator("select").first.select_option("diamonds")
        page.wait_for_timeout(100)

        page.locator("select").nth(1).select_option("carat")
        page.wait_for_timeout(100)

        # Navigate away
        page.goto(f"{BASE_URL}/patterns/")
        page.wait_for_timeout(100)

        # Go back
        page.go_back()
        wait_for_alpine()
        page.wait_for_timeout(200)

        # State should be restored
        dataset_select = page.locator("select").first
        x_select = page.locator("select").nth(1)

        expect(dataset_select).to_have_value("diamonds")
        expect(x_select).to_have_value("carat")
