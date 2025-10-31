# pyrsm AI Assistant Enhancement Plan

**Goal:** Enable GitHub Copilot, Cursor, and Claude Code to deeply understand pyrsm so students get intelligent, context-aware code suggestions.

**Date:** October 26, 2025
**Status:** Planning Phase

---

## Three-Pronged Approach

### **Phase 1: Enhance In-Repository Documentation** (Immediate Impact)
This directly improves GitHub Copilot's understanding since it indexes repository content.

#### 1.1 Expand README.md
- Add comprehensive "Quick Start" section with common use cases
- Add "Core Concepts" explaining the three main modules (basics, model, multivariate)
- Add code examples for most common operations (hypothesis testing, regression, classification)
- Add "API Overview" section with links to key classes
- Keep it concise but informative (~200-300 lines)

#### 1.2 Create `/docs` folder structure
```
docs/
├── api/                    # API reference (auto-generated from docstrings)
├── tutorials/              # Converted from Jupyter notebooks
├── guides/                 # How-to guides
└── examples/              # Quick reference examples
```

#### 1.3 Enhance docstrings systematically
- Ensure all public classes have complete docstrings with examples
- Add "Examples" section to every major class/function showing typical usage
- Use Google/NumPy docstring style consistently
- Focus on: `basics.*`, `model.*`, utility functions

#### 1.4 Create quick-reference examples
- `docs/examples/hypothesis_testing.md` - All basics examples
- `docs/examples/regression_models.md` - Linear/logistic regression
- `docs/examples/ml_models.md` - Random forest, XGBoost, MLP
- Each with 5-10 copy-pasteable code snippets

---

### **Phase 2: Build Custom MCP Server for pyrsm** (MCP-Compatible Editors)
Create a dedicated MCP server that serves pyrsm documentation to Claude Code, Cursor, and VS Code.

#### 2.1 Create MCP server structure
```
mcp-server/
├── package.json
├── src/
│   ├── index.ts          # Main MCP server
│   ├── indexer.ts        # Index pyrsm docs
│   └── search.ts         # Search/retrieve docs
└── docs-index/           # Pre-built documentation index
```

#### 2.2 Documentation indexing strategy
- Index all docstrings from Python source
- Index Jupyter notebooks from `examples/` (convert to markdown)
- Index markdown files from `docs/`
- Create searchable index with embeddings or keyword search
- Include code examples with full context

#### 2.3 MCP server features
- Tool: `search_pyrsm_docs(query)` - Search pyrsm documentation
- Tool: `get_pyrsm_example(class_or_function)` - Get usage examples
- Tool: `list_pyrsm_classes()` - List all available classes
- Resource: Expose all examples as MCP resources

#### 2.4 Installation guide for students
- Create `INSTALL_MCP.md` with step-by-step setup for:
  - VS Code (using MCP extension)
  - Cursor (via mcp.json config)
  - Claude Code (via CLI config)

---

### **Phase 3: Submit to Context7** (Broader Reach)
Get pyrsm added to Context7's curated library index.

#### 3.1 Prepare documentation site
- Use Sphinx or MkDocs to generate comprehensive docs site
- Convert Jupyter notebooks to HTML/Markdown tutorials
- Deploy to GitHub Pages or Read the Docs
- URL: `https://vnijs.github.io/pyrsm/` or similar

#### 3.2 Submit to Context7
- Follow their "adding projects" guide on GitHub
- Provide documentation URL, description, and justification
- Highlight: Educational use case, 50+ students using it, business analytics focus

#### 3.3 Create `.context7.json` (if supported)
- Configuration file pointing to documentation structure
- May help with acceptance/indexing process

---

## Testing & Validation Strategy

### Automated Testing via GitHub Copilot API
Use GitHub Copilot's API (or similar completion APIs) to test documentation effectiveness.

#### Test Scenarios
Create a standardized test suite with prompts like:

1. **Basic Usage Tests**
   - "I want to run a linear regression with 'sales' as the response and x1-x5 as the explanatory variables. Show me the summary output"
   - "Test if the mean of 'price' is significantly different from 100"
   - "Compare proportions between two groups in my dataset"
   - "Run a random forest classification model with these features"

2. **Advanced Usage Tests**
   - "Create a logistic regression with interaction terms"
   - "Generate predictions with confidence intervals from my regression model"
   - "Plot residuals from my linear model"
   - "Calculate VIF for multicollinearity check"

3. **Data Loading Tests**
   - "Load the diamonds dataset from pyrsm"
   - "Load the titanic dataset and show first few rows"

#### Testing Methodology

**Option A: Manual Testing Protocol**
```python
# Create test_prompts.py
prompts = [
    "I want to run a linear regression with 'sales' as the response and x1-x5 as the explanatory variables. Show me the summary output",
    "Test if the mean of 'price' is significantly different from 100",
    # ... more prompts
]

# For each prompt:
# 1. Use GitHub Copilot to generate code
# 2. Rate on scale 1-5:
#    - Uses pyrsm? (vs generic sklearn/statsmodels)
#    - Syntactically correct?
#    - Follows pyrsm idioms?
#    - Includes appropriate methods (e.g., .summary())?
# 3. Record results in spreadsheet
```

**Option B: Automated API Testing**
```python
# Use GitHub Copilot API or OpenAI Codex API
import openai

def test_documentation_impact(prompt, context_files=[]):
    """
    Test if documentation helps AI generate better pyrsm code

    Args:
        prompt: Natural language request
        context_files: List of doc files to include as context
    """
    # Without context
    response_baseline = openai.Completion.create(
        model="code-davinci-002",  # or copilot API
        prompt=prompt,
        max_tokens=200
    )

    # With documentation context
    full_prompt = "\n".join([
        "# pyrsm library documentation:",
        *[open(f).read() for f in context_files],
        "",
        "# User request:",
        prompt
    ])

    response_enhanced = openai.Completion.create(
        model="code-davinci-002",
        prompt=full_prompt,
        max_tokens=200
    )

    return {
        "baseline": response_baseline.choices[0].text,
        "enhanced": response_enhanced.choices[0].text,
        "uses_pyrsm": "pyrsm" in response_enhanced.choices[0].text
    }
```

**Option C: VS Code Extension Testing**
- Create a VS Code extension that logs Copilot suggestions
- Compare suggestions before/after documentation improvements
- Metrics: % of suggestions using pyrsm, correctness score, user acceptance rate

#### Test Phases

1. **Baseline (Week 0):**
   - Run test suite with current minimal documentation
   - Record baseline scores for each prompt
   - Document what Copilot suggests (generic sklearn? statsmodels? nothing?)

2. **After Phase 1 (Week 1-2):**
   - Re-run test suite after README/docstring enhancements
   - Measure improvement in pyrsm-specific suggestions
   - Identify remaining gaps

3. **After Phase 2 (Week 3):**
   - Test with MCP server active in Claude Code/Cursor
   - Compare MCP-enabled vs non-MCP suggestions

4. **After Phase 3 (Week 4+):**
   - Test if Context7 integration improves suggestions
   - Final comparison against baseline

#### Success Metrics

- **Primary:** % of test prompts where AI suggests pyrsm-specific code (target: >80%)
- **Secondary:** Syntactic correctness of suggestions (target: >90%)
- **Tertiary:** Follows pyrsm idioms/patterns (target: >70%)
- **User feedback:** Student satisfaction surveys before/after

---

## Implementation Timeline

### Week 1: Quick Wins (Phase 1 - Core)
1. **Baseline Testing** - Run initial test suite, document current state
2. Expand README with examples and API overview *(requires review)*
3. Audit and enhance docstrings in top 10 most-used classes *(requires review for major changes)*
4. Create `docs/examples/` with quick-reference snippets *(requires review)*

### Week 2: Documentation Infrastructure (Phase 1 Complete)
5. Set up Sphinx/MkDocs with auto-API generation *(configuration requires review)*
6. Convert key Jupyter notebooks to tutorial markdown *(minimal conversion, leverage existing)*
7. Build and test documentation site locally
8. **Mid-point Testing** - Re-run test suite, measure improvement

### Week 3: MCP Server (Phase 2)
9. Build custom MCP server for pyrsm *(architecture requires review)*
10. Test with Claude Code, Cursor, VS Code
11. Create installation guide for students *(requires review)*
12. **MCP Testing** - Test with MCP server active

### Week 4: Deployment & Submission (Phase 3)
13. Deploy documentation site to GitHub Pages
14. Submit pyrsm to Context7
15. Create student onboarding materials *(requires review)*
16. **Final Testing** - Complete test suite, measure total improvement

---

## Expected Student Experience

### Before
- Copilot suggests generic pandas/sklearn code
- Students need to constantly reference notebooks
- AI assistants don't understand pyrsm patterns

### After
- Copilot suggests pyrsm-specific code: `sm = pyrsm.basics.single_mean(data, var='price', comp_value=100)`
- MCP-enabled editors provide inline documentation and examples
- Tab completion shows pyrsm classes with full docstrings
- AI assistants understand pyrsm patterns and idioms
- Students write correct code faster with fewer errors

---

## Review & Approval Process

All major documentation additions will be submitted for review before committing:

1. **README expansion** - Review before merging
2. **New docstring patterns** - Review first example, then proceed with pattern
3. **docs/ folder structure** - Review initial structure
4. **Quick-reference examples** - Review first example file
5. **MCP server architecture** - Review design before implementation
6. **Student installation guides** - Review before distribution

Minor changes (typo fixes, formatting, minor clarifications) can proceed without review.

---

## Notes

- Existing Jupyter notebooks in `examples/` are excellent and will be leveraged throughout
- Good docstrings already exist in many classes (single_mean, regress, etc.) - we'll build on this foundation
- Focus on **practical examples** since students learn by doing
- All three phases work together but each provides independent value
- Testing is integrated throughout to measure actual impact

---

## Resources & References

- Context7: https://context7.com/
- Context7 GitHub: https://github.com/upstash/context7
- MCP Protocol: https://modelcontextprotocol.io/
- GitHub Copilot API: https://docs.github.com/en/copilot
- Sphinx Documentation: https://www.sphinx-doc.org/
- MkDocs: https://www.mkdocs.org/
