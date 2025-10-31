# MCP Server Integration Design

## Three Modes of File Loading

### Mode 1: Direct File Path (✅ IMPLEMENTED)

**Use case**: VS Code, Claude Code, command-line workflows

**How it works**:
```
User: "Load data from /home/vnijs/data/sales.csv"
AI calls: data_load_file(file_path='/home/vnijs/data/sales.csv')
MCP server: Loads CSV, stores in DATA_STORE['sales']
```

**Status**: ✅ Working (tested with test_file_loading.py)

**Files**:
- `server_regression.py:93-292` - data_load_file tool
- `test_file_loading.py` - validation tests

---

### Mode 2: Shiny App Integration (🚧 DESIGN)

**Use case**: pyrsm-genai-shiny web interface

**Architecture**:
```
Shiny UI (file upload)
    ↓
Save to persistent directory
    ↓
MCP server loads via data_load_file
    ↓
Shared DATA_STORE for both Shiny and AI
```

**Implementation steps**:

1. **Create persistent upload directory**:
```python
# In app.py
UPLOAD_DIR = Path.home() / "pyrsm_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
```

2. **Save uploaded files persistently**:
```python
@reactive.Effect
@reactive.event(input.data_file)
def _():
    file_info = input.data_file()[0]
    if file_info:
        # Save to persistent location
        persistent_path = UPLOAD_DIR / file_info['name']
        shutil.copy(file_info['datapath'], persistent_path)

        # Load via pandas (current method)
        data = pd.read_parquet(persistent_path)
        datasets[dataset_name] = data

        # Optional: Also register with MCP server
        # This allows AI to access the same data
```

3. **Share state between Shiny and MCP** (optional):
```python
# Option A: MCP server reads from UPLOAD_DIR
# Option B: Shiny calls MCP tools for analysis
# Option C: Both maintain separate registries
```

**Status**: 🚧 Design complete, implementation pending

**Benefits**:
- Users upload files via Shiny UI
- AI assistant can reference uploaded files
- No need to manually specify file paths
- Shared data registry

---

### Mode 3: VS Code/Claude Code Native Upload (🔮 FUTURE)

**Use case**: Drag-and-drop file upload in AI chat

**Current limitations**:
- MCP protocol doesn't support file upload UI
- Would require extension to MCP spec
- Clients could implement file picker dialogs

**Workaround**:
Use Mode 1 (file paths) with tab completion in terminal

**Status**: 🔮 Waiting for MCP protocol evolution

---

## Current Integration Status

### Working Now
- ✅ Load pyrsm built-in datasets (50+)
- ✅ Load files via absolute paths
- ✅ Regression analysis with state management
- ✅ Context-aware suggested next steps

### Shiny App Integration (Design Ready)
- File upload: Use existing Shiny `input_file()`
- Data storage: Save to `~/pyrsm_uploads/`
- MCP access: Use `data_load_file()` with persistent paths
- State sharing: Optional - both can maintain registries

### Next Steps
1. Test file path mode in Claude Code
2. Implement persistent upload directory in Shiny app
3. Test end-to-end: Upload in Shiny → Analyze in Claude Code

---

## File Locations

**MCP Server**: `/home/vnijs/gh/pyrsm/mcp-server/`
- `server_regression.py` - Main server with all tools
- `test_file_loading.py` - File loading validation

**Shiny App**: `/home/vnijs/gh/pyrsm-genai-shiny/pyrsm-genai/`
- `app.py:141-224` - File upload UI and handlers
- `tools/tool_handlers.py` - Analysis tool execution

**Shared Upload Directory** (to be created):
- `~/pyrsm_uploads/` - Persistent storage for uploaded files
