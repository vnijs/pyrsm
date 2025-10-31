# Testing Guide: HTMX + Alpine.js Pattern Validation

This guide provides comprehensive testing procedures using **Playwright** for automated browser testing and **make** commands for easy execution.

## 🎯 Testing Philosophy

**Critical Question**: Do these reactive UI patterns work reliably enough to build a full platform on them?

**What We're Validating**:
1. ✅ Dependent dropdowns (exclusion logic)
2. ✅ Conditional visibility (show/hide based on selection)
3. ✅ State restoration (localStorage + DOM sync)
4. ✅ HTMX + Alpine integration (server rendering + client reactivity)

---

## 🚀 Quick Start

### Initial Setup

```bash
cd mcp-server/examples/htmx-alpine-patterns
make setup
```

This runs:
- `uv venv` - Creates virtual environment
- `uv pip install -r requirements.txt` - Installs Django, Playwright, pytest
- `playwright install chromium` - Installs browser
- `python manage.py migrate` - Sets up database

### Run All Tests

```bash
make test
```

Expected output:
```
🧪 Running Playwright tests (headless)...
tests/test_dependent_dropdowns.py::TestDependentDropdowns::test_page_loads_successfully PASSED
tests/test_dependent_dropdowns.py::TestDependentDropdowns::test_x_selection_excludes_from_y_dropdown PASSED
tests/test_conditional_visibility.py::TestConditionalVisibility::test_page_loads_successfully PASSED
...
========================= 20 passed in 45.2s =========================
```

---

## 📋 Test Commands Reference

### Basic Testing

| Command | Description | Use Case |
|---------|-------------|----------|
| `make test` | Run all tests (headless) | CI/CD, quick validation |
| `make test-verbose` | Detailed output + console logs | Debugging failures |
| `make test-headed` | Visual browser (slow motion) | Watching test execution |
| `make test-debug` | Pause on failure | Interactive debugging |

### Pattern-Specific Testing

| Command | Description | Tests |
|---------|-------------|-------|
| `make test-dropdowns` | Dependent dropdowns only | Exclusion logic, state restore |
| `make test-visibility` | Conditional visibility only | HTMX updates, param restore |
| `make test-state` | State restoration only | Complex workflows |

### Reporting

| Command | Description | Output |
|---------|-------------|--------|
| `make test-report` | Generate HTML report | `test-report.html` |

---

## 🧪 Test Coverage

### Example 1: Dependent Dropdowns (`test_dependent_dropdowns.py`)

**20 Tests** covering:

✅ **Core Functionality**:
- `test_page_loads_successfully` - Verifies page loads and Alpine initializes
- `test_selecting_dataset_shows_variable_dropdowns` - Dataset selection reveals X/Y dropdowns
- `test_x_selection_excludes_from_y_dropdown` - X selection removes option from Y
- `test_y_selection_excludes_from_x_dropdown` - Y selection removes option from X
- `test_both_selections_mutual_exclusion` - Both dropdowns exclude each other
- `test_clearing_selection_restores_options` - Clearing restores all options

✅ **State Persistence**:
- `test_state_persists_to_localstorage` - Saves to localStorage correctly
- `test_state_restores_after_page_reload` - Full restore after F5
- `test_state_restoration_maintains_exclusions` - Exclusions work after reload

✅ **User Workflows**:
- `test_switching_datasets_clears_variables` - Changing dataset clears selections
- `test_clear_button_resets_everything` - Clear button works
- `test_summary_appears_with_both_selections` - Summary shows when ready

✅ **Edge Cases**:
- `test_rapid_selection_changes` - Handles rapid clicks
- `test_back_button_restores_state` - Browser back/forward works
- `test_console_logs_state_management` - Proper logging

### Example 2: Conditional Visibility (`test_conditional_visibility.py`)

**15 Tests** covering:

✅ **Visibility Logic**:
- `test_selecting_test_shows_conditional_params` - Params appear on selection
- `test_different_tests_show_different_params` - Correct params per test
- `test_visibility_indicator_updates` - Status indicator correct

✅ **HTMX Integration**:
- `test_htmx_request_indicator` - Loading states work
- `test_parameter_values_save_on_change` - Saves before DOM swap
- `test_htmx_beforeswap_event_saves_params` - beforeSwap event fires

✅ **Parameter Restoration**:
- `test_parameter_values_restore_after_switching` - Params restore when returning
- `test_state_persists_across_page_reload` - Full state after reload
- `test_checkbox_parameter_saves_correctly` - Checkbox state works

✅ **User Actions**:
- `test_clear_button_resets_state` - Clear works
- `test_run_test_button_captures_current_params` - Run button captures state

✅ **Edge Cases**:
- `test_rapid_test_switching` - Handles rapid changes
- `test_multiple_parameter_changes_before_switch` - Multiple changes save

---

## 📊 Running Tests: Step-by-Step

### 1. Full Test Suite

```bash
# Terminal 1: Start Django server
make run
```

```bash
# Terminal 2: Run tests
make test
```

**Expected Results**:
- All tests pass ✅
- ~30-60 seconds execution time
- No browser windows (headless mode)

### 2. Visual Testing (Watch Tests Run)

```bash
# Terminal 1: Keep server running
make run
```

```bash
# Terminal 2: Run with visible browser
make test-headed
```

**What You'll See**:
- Browser window opens
- Tests execute in slow motion
- Dropdown selections happen visually
- Console logs visible in terminal

### 3. Debug Failed Test

```bash
# If a test fails:
make test-debug
```

**Features**:
- Browser pauses on failure
- Inspector opens automatically
- You can click around manually
- Resume with debugger

### 4. Test Specific Pattern

```bash
# Only test dropdowns
make test-dropdowns

# Only test visibility
make test-visibility
```

---

## 🔍 Understanding Test Results

### Successful Test Run

```bash
$ make test
🧪 Running Playwright tests (headless)...
=============================== test session starts ===============================
platform linux -- Python 3.13.0, pytest-8.3.4, pluggy-1.5.0
collected 35 items

tests/test_dependent_dropdowns.py::TestDependentDropdowns::test_page_loads_successfully PASSED [  2%]
tests/test_dependent_dropdowns.py::TestDependentDropdowns::test_x_selection_excludes_from_y_dropdown PASSED [  5%]
...
tests/test_conditional_visibility.py::TestConditionalVisibility::test_state_persists_across_page_reload PASSED [100%]

============================== 35 passed in 48.23s ================================
✅ All patterns verified!
```

**Interpretation**: All reactive patterns work correctly. Safe to proceed.

### Failed Test Example

```bash
tests/test_dependent_dropdowns.py::TestDependentDropdowns::test_x_selection_excludes_from_y_dropdown FAILED

______________ TestDependentDropdowns.test_x_selection_excludes_from_y_dropdown _______________

    def test_x_selection_excludes_from_y_dropdown(page):
        ...
        assert "salary" not in y_options_after, "salary should be excluded from Y"
>       AssertionError: salary should be excluded from Y

        Captured console:
        console.log: ✓ Loaded 6 columns: ['salary', 'rank', ...]
```

**Interpretation**: Exclusion logic broken. Need to fix Alpine computed property.

---

## 🐛 Debugging Tips

### View Console Logs

```bash
make test-verbose
```

Output includes:
```
console.log: 🚀 Dependent Dropdowns component initialized
console.log: 📊 Dataset changed to: salary
console.log: 💾 State saved: {dataset: 'salary', xVariable: '', ...}
```

### Inspect Element State

```bash
make test-debug
```

Then in browser inspector:
```javascript
// Check Alpine state
$0.__x.$data

// Check localStorage
localStorage.getItem('dependentDropdowns')

// Check HTMX attributes
htmx.find('#test-params-container')
```

### Screenshot on Failure

Add to test:
```python
def test_my_feature(page: Page):
    try:
        # ... test code
        pass
    except Exception as e:
        page.screenshot(path="failure.png")
        raise
```

---

## 📈 Test Report

Generate HTML report:

```bash
make test-report
```

Opens `test-report.html` with:
- Pass/fail summary
- Execution time per test
- Console logs
- Screenshots (if configured)

---

## 🎯 Success Criteria

### All Tests Must Pass

Before proceeding to full platform:
- ✅ 100% test pass rate
- ✅ No flaky tests (run 3 times, all pass)
- ✅ Edge cases handled
- ✅ State restoration works consistently

### Performance Benchmarks

- Page load: < 500ms
- State restoration: < 200ms
- HTMX update: < 300ms
- Full test suite: < 120s

---

## 🔄 Continuous Integration

### GitHub Actions Example

```yaml
name: Test UI Patterns
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.13'

      - name: Install uv
        run: pip install uv

      - name: Setup
        run: |
          cd examples/htmx-alpine-patterns
          make install
          make playwright-install
          make migrate

      - name: Run tests
        run: |
          cd examples/htmx-alpine-patterns
          make ci-test

      - name: Upload test report
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: test-results
          path: examples/htmx-alpine-patterns/test-report.html
```

---

## 🛠️ Troubleshooting

### Tests Won't Start

**Problem**: `ModuleNotFoundError: No module named 'playwright'`

**Solution**:
```bash
source .venv/bin/activate
make playwright-install
```

### Server Not Running

**Problem**: `Connection refused to localhost:8000`

**Solution**:
```bash
# Terminal 1
make run

# Wait for "Starting development server..."
# Then in Terminal 2:
make test
```

### Slow Tests

**Problem**: Tests taking > 2 minutes

**Solutions**:
- Use `make test` (headless is faster)
- Run specific tests: `make test-dropdowns`
- Check for `wait_for_timeout` with large values

### Browser Won't Close

**Problem**: Browser windows left open after test

**Solution**:
```bash
# Kill all chromium processes
pkill -9 chromium

# Then re-run
make test
```

---

## 📚 Writing New Tests

### Test Template

```python
import pytest
from playwright.sync_api import Page, expect

@pytest.mark.ui
class TestMyPattern:
    """Test suite for my pattern"""

    def test_my_feature(
        self, page: Page, clear_local_storage, wait_for_alpine
    ):
        """Test description"""
        # Navigate
        page.goto("http://localhost:8000/patterns/my-pattern/")
        wait_for_alpine()

        # Interact
        button = page.locator("button")
        button.click()

        # Assert
        result = page.locator(".result")
        expect(result).to_be_visible()
```

### Best Practices

1. **Always clear localStorage**: Use `clear_local_storage` fixture
2. **Wait for Alpine**: Use `wait_for_alpine()` after navigation
3. **Wait for HTMX**: Use `wait_for_htmx()` after requests
4. **Add timeouts**: `page.wait_for_timeout(100)` after actions
5. **Use expect()**: Playwright's auto-waiting assertions

---

## ✅ Verification Checklist

Before committing code:

- [ ] `make test` - All tests pass
- [ ] `make test-headed` - Visually verified
- [ ] `make test-report` - HTML report reviewed
- [ ] No console errors in test output
- [ ] State restoration works after reload
- [ ] Edge cases tested (rapid clicks, back button)
- [ ] Performance acceptable (< 2min full suite)

Before starting full platform:

- [ ] All patterns verified 3x
- [ ] No flaky tests
- [ ] Documentation updated
- [ ] Team reviewed test results

---

## 🎓 Learning Resources

- **Playwright Docs**: https://playwright.dev/python/
- **pytest Docs**: https://docs.pytest.org/
- **Alpine.js Testing**: https://alpinejs.dev/advanced/testing
- **HTMX Testing**: https://htmx.org/docs/#testing

---

## 📞 Get Help

If tests fail:
1. Run `make test-verbose` for detailed output
2. Run `make test-debug` to inspect manually
3. Check console logs for Alpine/HTMX errors
4. Review test expectations vs actual behavior
5. Ask: "Is this a test bug or code bug?"

**Remember**: Tests failing = patterns need refinement before full build!
