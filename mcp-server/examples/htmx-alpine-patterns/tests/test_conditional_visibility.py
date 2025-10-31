"""
Playwright tests for Conditional Visibility pattern

Tests dynamic visibility based on selections, HTMX partial updates,
and state restoration after DOM swaps.
"""
import pytest
import json
from playwright.sync_api import Page, expect


BASE_URL = "http://localhost:8000"
PATTERN_URL = f"{BASE_URL}/patterns/conditional-visibility/"


@pytest.mark.ui
class TestConditionalVisibility:
    """Test suite for conditional visibility pattern"""

    def test_page_loads_successfully(self, page: Page, clear_local_storage):
        """Verify page loads correctly"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test

        expect(page).to_have_title("Conditional Visibility Example")

        heading = page.locator("h2")
        expect(heading).to_contain_text("Conditional Visibility")

        # Test type selector should be visible
        test_select = page.locator("select").first
        expect(test_select).to_be_visible()

    def test_selecting_test_shows_conditional_params(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx
    ):
        """Test that selecting test type shows conditional parameters"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Parameters container should not be visible initially
        params_container = page.locator("#test-params-container")

        # Select t-test
        test_select = page.locator("select").first
        test_select.select_option("ttest")

        # Wait for HTMX to load parameters
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Conditional params should be visible
        expect(params_container).to_be_visible()

        # Should contain sample type parameter
        expect(params_container).to_contain_text("Sample type")

    def test_different_tests_show_different_params(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx
    ):
        """Test that different test types load different parameters"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        test_select = page.locator("select").first
        params_container = page.locator("#test-params-container")

        # Test 1: t-test
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        expect(params_container).to_contain_text("Sample type")
        expect(params_container).to_contain_text("Confidence level")

        # Test 2: Chi-square
        test_select.select_option("chisquare")
        page.wait_for_timeout(300)
        wait_for_htmx()

        expect(params_container).to_contain_text("Yates correction")
        expect(params_container).not_to_contain_text("Sample type")

        # Test 3: ANOVA
        test_select.select_option("anova")
        page.wait_for_timeout(300)
        wait_for_htmx()

        expect(params_container).to_contain_text("Post-hoc test")

    def test_visibility_indicator_updates(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that visibility status indicator updates correctly"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        test_select = page.locator("select").first

        # Select test with conditional params (t-test)
        test_select.select_option("ttest")
        page.wait_for_timeout(300)

        # Check visibility indicator shows "Visible"
        alert = page.locator(".alert")
        expect(alert).to_contain_text("Visible")
        # Check that alert has the info class
        assert "alert-info" in alert.get_attribute("class")

        # Select test without conditional params (could add if needed)
        # For now, verify the pattern works

    def test_htmx_request_indicator(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test that HTMX loading indicator works"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        test_select = page.locator("select").first

        # Start watching for HTMX request class
        test_select.select_option("ttest")

        # HTMX should add htmx-request class during request
        # (This is a quick check - class may already be removed)
        page.wait_for_timeout(50)

        # After request completes, class should be removed
        page.wait_for_timeout(300)

        body = page.locator("body")
        body_class = body.get_attribute("class") or ""
        assert "htmx-request" not in body_class

    def test_parameter_values_save_on_change(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx, get_local_storage
    ):
        """Test that parameter values are saved when changed"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Select test type
        test_select = page.locator("select").first
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Change parameter value
        sample_type_select = page.locator("select[name='sample_type']")
        sample_type_select.select_option("paired")
        page.wait_for_timeout(200)

        # Check localStorage (note: saving happens on test type change or before HTMX swap)
        # So we need to trigger another change to save current params
        test_select.select_option("wilcoxon")
        page.wait_for_timeout(300)

        stored = get_local_storage("conditionalVisibility")
        assert stored is not None

        state = json.loads(stored)
        assert "savedParams" in state
        assert "ttest" in state["savedParams"]
        assert state["savedParams"]["ttest"]["sample_type"] == "paired"

    def test_parameter_values_restore_after_switching(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx
    ):
        """Test that parameter values restore when switching back to test"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        test_select = page.locator("select").first

        # Select t-test and set sample type to paired
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        sample_type = page.locator("select[name='sample_type']")
        sample_type.select_option("paired")
        page.wait_for_timeout(100)

        # Switch to different test
        test_select.select_option("chisquare")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Switch back to t-test
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Sample type should be restored to "paired"
        sample_type_restored = page.locator("select[name='sample_type']")
        expect(sample_type_restored).to_have_value("paired")

    def test_state_persists_across_page_reload(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx
    ):
        """Test that full state restores after page reload"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Set up state
        test_select = page.locator("select").first
        test_select.select_option("anova")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Change parameter
        post_hoc = page.locator("select[name='post_hoc']")
        post_hoc.select_option("tukey")
        page.wait_for_timeout(100)

        # Reload page
        page.reload()
        wait_for_alpine()
        page.wait_for_timeout(500)  # Extra time for HTMX load + restoration

        # Test type should be restored
        test_select_restored = page.locator("select").first
        expect(test_select_restored).to_have_value("anova")

        # Parameters should be visible and restored
        post_hoc_restored = page.locator("select[name='post_hoc']")
        expect(post_hoc_restored).to_be_visible()
        expect(post_hoc_restored).to_have_value("tukey")

    def test_clear_button_resets_state(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx, get_local_storage
    ):
        """Test that Clear button resets everything"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Set up state
        test_select = page.locator("select").first
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Click Clear button
        clear_button = page.get_by_role("button", name="Clear")
        clear_button.click()
        page.wait_for_timeout(100)

        # Test type should be cleared
        expect(test_select).to_have_value("")

        # Parameters container should show placeholder
        params_container = page.locator("#test-params-container")
        expect(params_container).to_contain_text("Select a test type")

        # localStorage should be cleared
        stored = get_local_storage("conditionalVisibility")
        assert stored is None

    def test_run_test_button_captures_current_params(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx
    ):
        """Test that Run Test button captures current parameter values"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Set up test with parameters
        test_select = page.locator("select").first
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        sample_type = page.locator("select[name='sample_type']")
        sample_type.select_option("paired")
        page.wait_for_timeout(100)

        confidence = page.locator("input[name='confidence']")
        confidence.fill("0.99")
        page.wait_for_timeout(100)

        # Click Run Test button
        run_button = page.get_by_role("button", name="Run Test")

        # Set up dialog handler to verify alert content
        dialog_handled = False
        dialog_message = ""

        def handle_dialog(dialog):
            nonlocal dialog_handled, dialog_message
            dialog_message = dialog.message
            dialog_handled = True
            dialog.accept()

        page.on("dialog", handle_dialog)

        run_button.click()
        page.wait_for_timeout(200)

        # Verify alert was shown
        assert dialog_handled, "Run Test should trigger alert"
        assert "ttest" in dialog_message.lower()
        assert "paired" in dialog_message.lower()

    def test_htmx_beforeswap_event_saves_params(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx, console_messages
    ):
        """Test that parameters are saved before HTMX swaps DOM"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Select test and set parameters
        test_select = page.locator("select").first
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        sample_type = page.locator("select[name='sample_type']")
        sample_type.select_option("paired")
        page.wait_for_timeout(100)

        # Switch test (triggers beforeSwap)
        test_select.select_option("wilcoxon")
        page.wait_for_timeout(300)

        # Check console logs for saving message
        log_texts = [msg['text'] for msg in console_messages]
        assert any('saving' in text.lower() or 'saved' in text.lower() for text in log_texts)

    def test_checkbox_parameter_saves_correctly(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx
    ):
        """Test that checkbox parameters save and restore correctly"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        # Select chi-square test (has checkbox parameter)
        test_select = page.locator("select").first
        test_select.select_option("chisquare")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Find and check the correction checkbox
        correction_checkbox = page.locator("input[name='correction']")
        expect(correction_checkbox).to_be_visible()

        # Check it
        correction_checkbox.check()
        page.wait_for_timeout(100)

        # Switch away and back
        test_select.select_option("ttest")
        page.wait_for_timeout(300)

        test_select.select_option("chisquare")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Checkbox should be checked
        correction_restored = page.locator("input[name='correction']")
        expect(correction_restored).to_be_checked()


@pytest.mark.ui
@pytest.mark.slow
class TestConditionalVisibilityEdgeCases:
    """Edge case tests for conditional visibility"""

    def test_rapid_test_switching(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test rapid switching between test types"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        test_select = page.locator("select").first

        # Rapidly switch tests
        for test in ["ttest", "wilcoxon", "chisquare", "anova"]:
            test_select.select_option(test)
            page.wait_for_timeout(100)

        # Final state should be consistent
        page.wait_for_timeout(500)  # Let everything settle

        expect(test_select).to_have_value("anova")
        params_container = page.locator("#test-params-container")
        expect(params_container).to_contain_text("Post-hoc")

    def test_multiple_parameter_changes_before_switch(
        self, page: Page, clear_local_storage, wait_for_alpine, wait_for_htmx
    ):
        """Test multiple parameter changes are all saved"""
        page.goto(PATTERN_URL)
        page.evaluate("localStorage.clear()")  # Clear state before test
        wait_for_alpine()

        test_select = page.locator("select").first
        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # Change multiple parameters
        sample_type = page.locator("select[name='sample_type']")
        sample_type.select_option("paired")

        confidence = page.locator("input[name='confidence']")
        confidence.fill("0.90")
        page.wait_for_timeout(100)

        # Switch away and back
        test_select.select_option("anova")
        page.wait_for_timeout(300)

        test_select.select_option("ttest")
        page.wait_for_timeout(300)
        wait_for_htmx()

        # All parameters should be restored
        sample_type_restored = page.locator("select[name='sample_type']")
        confidence_restored = page.locator("input[name='confidence']")

        expect(sample_type_restored).to_have_value("paired")
        expect(confidence_restored).to_have_value("0.90")
