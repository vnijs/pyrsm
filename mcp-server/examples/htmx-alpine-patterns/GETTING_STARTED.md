# Getting Started with HTMX+Alpine Pattern Examples

## Quick Start

```bash
# Navigate to examples directory
cd /home/vnijs/gh/pyrsm/mcp-server/examples/htmx-alpine-patterns

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt

# Initialize database
python manage.py migrate

# Run development server
python manage.py runserver

# Open browser to http://localhost:8000/patterns/
```

## What's Implemented

✅ **Complete Examples**:

1. **Dependent Dropdowns** (`/patterns/dependent-dropdowns/`)
   - Variable exclusion logic (X can't be in Y, Y can't be in X)
   - State persistence with localStorage
   - Progressive restoration with `$nextTick()`

2. **Conditional Visibility** (`/patterns/conditional-visibility/`)
   - Show/hide parameters based on test type selection
   - HTMX partial updates for server-rendered forms
   - State synchronization between Alpine and HTMX
   - Parameter value restoration after DOM swaps

3. **HTMX Partials**:
   - `column_options.html` - Dynamic column lists
   - `test_params.html` - Conditional test parameters
   - `distribution_params.html` - Distribution-specific inputs

## Key Files

```
htmx-alpine-patterns/
├── app/
│   ├── views.py                    # Django views + HTMX endpoints
│   ├── urls.py                     # URL routing
│   ├── templates/
│   │   ├── base.html               # Base with HTMX/Alpine CDN
│   │   └── patterns/
│   │       ├── index.html          # Landing page
│   │       ├── dependent_dropdowns.html
│   │       ├── conditional_visibility.html
│   │       └── partials/           # HTMX partials
│   └── static/
│       ├── js/
│       │   ├── dependent-dropdowns.js    # Alpine component
│       │   └── conditional-visibility.js # Alpine component
│       └── css/
│           └── patterns.css        # Styling
├── config/
│   ├── settings.py                 # Django settings
│   └── urls.py                     # Project URLs
├── manage.py                       # Django management
└── requirements.txt                # Dependencies
```

## Testing Checklist

### Dependent Dropdowns
- [ ] Select X variable → Y dropdown excludes it
- [ ] Select Y variable → X dropdown excludes it
- [ ] Refresh page (F5) → selections restore correctly
- [ ] Clear all → both dropdowns show all options
- [ ] Switch datasets → variables clear properly
- [ ] Console logs show state management

### Conditional Visibility
- [ ] Select "ttest" → sample type parameters appear
- [ ] Select "chisquare" → different parameters
- [ ] Switch between tests → HTMX updates correctly
- [ ] Refresh page → test type and parameters restore
- [ ] Fill parameters → switch test → return → params restore
- [ ] Console shows HTMX swap events

## Next Steps

### Still To Build:
1. **State Restoration Example** (Complex workflow)
   - Full analysis workflow: dataset → variables → distribution → params
   - Tests progressive restoration with multiple dependencies
   - Validates HTMX + Alpine synchronization at scale

2. **Integration Testing**
   - Automated tests for each pattern
   - Edge case validation
   - Cross-browser testing

3. **Documentation**
   - Inline code comments
   - Video walkthrough
   - Common pitfalls guide

### Integration with Full Platform:
Once patterns are validated, apply to:
- Wizard mode (step-by-step analysis)
- Dashboard mode (Radiant-style interface)
- Real MCP tool integration
- WebSocket collaboration

## Troubleshooting

### Django not found
```bash
# Make sure virtual environment is activated
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Port 8000 already in use
```bash
# Use different port
python manage.py runserver 8001
```

### HTMX not loading partials
- Check browser DevTools Network tab
- Verify CSRF token in requests
- Check Django logs for errors

### Alpine state not persisting
- Open browser DevTools Console
- Check localStorage (Application tab)
- Verify `init()` is being called

## Architecture Reference

See [WEB_ARCHITECTURE.md](../../../WEB_ARCHITECTURE.md) for full web platform architecture.

## Feedback

As you test these examples, note:
- Which patterns work well?
- Where does state restoration break?
- Any UI quirks with HTMX/Alpine interaction?
- Performance issues?

This feedback will guide the full platform implementation.
