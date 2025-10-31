# HTMX + Alpine.js UI Pattern Examples

This directory contains small, focused examples to validate HTMX+Alpine.js patterns before building the full web platform.

## Purpose

Test and validate solutions for UI dynamics challenges:
1. **Dependent dropdowns**: Variable selected in dropdown A cannot appear in dropdown B
2. **Conditional visibility**: Show input C only when input A equals X
3. **State restoration**: Maintain UI state with localStorage when UI has dependencies

## Running the Examples

```bash
# Install dependencies
cd examples/htmx-alpine-patterns
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
uv pip install -r requirements.txt

# Run development server
python manage.py migrate
python manage.py runserver

# Visit http://localhost:8000/patterns/
```

## Examples

### 1. Dependent Dropdowns (`/patterns/dependent-dropdowns/`)

**Problem**: When selecting variables for regression (X and Y), the same variable shouldn't appear in both dropdowns.

**Solution**:
- Alpine.js reactive `availableColumns` computed properties
- Filters options based on selections
- State restoration preserves exclusions

**Test Cases**:
- Select X variable → Y dropdown excludes it
- Select Y variable → X dropdown excludes it
- Refresh page → selections restore correctly
- Clear selection → both dropdowns show all options

### 2. Conditional Visibility (`/patterns/conditional-visibility/`)

**Problem**: Statistical tests have conditional parameters (e.g., "paired/independent" only for t-test).

**Solution**:
- Alpine.js `x-show` with reactive conditions
- Server-side HTMX partial updates for parameter forms
- State restoration re-applies visibility rules

**Test Cases**:
- Select "t-test" → sample type selector appears
- Select "chi-square" → sample type selector hidden
- Switch between tests → correct fields shown
- Refresh page → test + conditional params restore

### 3. State Restoration with UI Dynamics (`/patterns/state-restoration/`)

**Problem**: Complex forms with dependencies break when restoring from localStorage.

**Solution**:
- Progressive restoration (primary → secondary → conditional)
- `$nextTick()` for DOM update synchronization
- Distribution-specific parameter storage

**Test Cases**:
- Complete form with all dependencies → refresh → all state restores
- Change dataset → variable selections clear correctly
- Switch distribution → previous distribution params saved
- Browser back button → state remains consistent

## Key Files

```
htmx-alpine-patterns/
├── README.md                          # This file
├── manage.py                          # Django management
├── config/
│   ├── settings.py                    # Django settings
│   └── urls.py                        # URL routing
├── app/
│   ├── views.py                       # View functions for examples
│   ├── urls.py                        # App URL patterns
│   ├── templates/
│   │   ├── base.html                  # Base template with HTMX/Alpine
│   │   └── patterns/
│   │       ├── index.html             # Example list
│   │       ├── dependent_dropdowns.html
│   │       ├── conditional_visibility.html
│   │       └── state_restoration.html
│   └── static/
│       ├── js/
│       │   ├── dependent-dropdowns.js # Alpine component
│       │   ├── conditional-visibility.js
│       │   └── state-restoration.js
│       └── css/
│           └── patterns.css           # Styling
└── requirements.txt                   # Python dependencies
```

## Learning Outcomes

After completing these examples, you'll have validated patterns for:

✅ Reactive form dependencies
✅ Dynamic visibility with state persistence
✅ HTMX partial updates without losing Alpine state
✅ Progressive state restoration for complex forms
✅ Integration of server-side rendering with client-side reactivity

## Next Steps

Once these patterns are validated:
1. Apply to full analysis wizard
2. Build out dashboard mode
3. Add real MCP tool integration
4. Implement collaboration features

## References

- **Existing work**: `/home/vnijs/gh/rsm-django-components/components/rsm-django-radiant`
- **Alpine.js docs**: https://alpinejs.dev/
- **HTMX docs**: https://htmx.org/
