# Django File Browser + MCP Server Integration

## Quick Start

### 1. Select Files in Django

Browse and select files at: `http://localhost:8000/filebrowser/`

### 2. Get MCP-ready File Paths

**API Endpoint**: `GET /filebrowser/api/mcp-file-info/`

**Response**:
```json
{
  "count": 1,
  "files": [{
    "absolute_path": "/home/vnijs/data/sales.csv",
    "name": "sales.csv",
    "root": "data",
    "rel_path": "sales.csv"
  }],
  "instructions": [
    "data_load_file(file_path=\"/home/vnijs/data/sales.csv\")"
  ]
}
```

### 3. Use in Claude Code

**User**: "Load the files I selected in Django"

**AI reads**: `curl http://localhost:8000/filebrowser/api/mcp-file-info/`

**AI calls MCP**: `data_load_file(file_path="/home/vnijs/data/sales.csv")`

**Result**: Dataset loaded and ready for analysis!

---

## Integration Workflow

```
┌─────────────────────┐
│  Django File Browser│
│  (django-sfiles)    │
└──────────┬──────────┘
           │
           │ User selects files
           │ via web UI
           ▼
┌─────────────────────┐
│  Session Storage    │
│  selected_files[]   │
└──────────┬──────────┘
           │
           │ GET /api/mcp-file-info/
           ▼
┌─────────────────────┐
│  Absolute File Paths│
│  + MCP Instructions │
└──────────┬──────────┘
           │
           │ User tells AI:
           │ "Analyze selected files"
           ▼
┌─────────────────────┐
│  pyrsm MCP Server   │
│  data_load_file()   │
└──────────┬──────────┘
           │
           │ Loads into DATA_STORE
           ▼
┌─────────────────────┐
│  Ready for Analysis │
│  regress_fit(), etc │
└─────────────────────┘
```

---

## Files Modified

### django-sfiles
**File**: `/home/vnijs/gh/django-sfiles/filebrowser/api.py`
- Added: `FileBrowserAPI.get_mcp_file_info()` method

**File**: `/home/vnijs/gh/django-sfiles/filebrowser/views.py`
- Added: `get_mcp_file_info()` view (line 674-699)

**File**: `/home/vnijs/gh/django-sfiles/filebrowser/urls.py`
- Added: `path('api/mcp-file-info/', ...)` route (line 18)

### pyrsm MCP Server
**File**: `/home/vnijs/gh/pyrsm/mcp-server/server_regression.py`
- Existing: `data_load_file()` tool (line 93-292)
- Status: ✅ Already working

---

## Usage Examples

### Example 1: Single File Analysis

**In Django**: Select `sales_data.csv`

**In Terminal**:
```bash
curl http://localhost:8000/filebrowser/api/mcp-file-info/
```

**Response**:
```json
{
  "instructions": [
    "data_load_file(file_path=\"/home/vnijs/data/sales_data.csv\")"
  ]
}
```

**In Claude Code**:
```
User: "Load that file and run a regression: sales ~ price + marketing"

AI: [Calls MCP tools]
  1. data_load_file(file_path="/home/vnijs/data/sales_data.csv")
  2. regress_fit(data_name="sales_data", rvar="sales", evar=["price", "marketing"])

Result: ✓ Model fitted (R² = 0.85, p < 0.001)
```

### Example 2: Multiple File Comparison

**In Django**: Select `q1_sales.csv`, `q2_sales.csv`, `q3_sales.csv`

**In Claude Code**:
```
User: "Compare these quarterly sales files"

AI: [Gets MCP info, loads each file]
  1. data_load_file(...q1_sales.csv) → "q1_sales"
  2. data_load_file(...q2_sales.csv) → "q2_sales"
  3. data_load_file(...q3_sales.csv) → "q3_sales"
  4. Compares trends across datasets

Result: "Q3 sales increased 15% vs Q1..."
```

---

## Configuration

### Django Settings

```python
# In settings.py
FILEBROWSER_ROOTS = {
    'home': os.path.expanduser('~/'),
    'data': '/home/vnijs/data',
    'uploads': '/home/vnijs/pyrsm_uploads'
}

FILEBROWSER_MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
```

### MCP Server

**Config**: `.mcp.json` in Claude Code or `~/.config/Code/User/mcp.json` in VS Code

```json
{
  "mcpServers": {
    "pyrsm-regression": {
      "command": "/home/vnijs/gh/pyrsm/mcp-server/.venv/bin/python",
      "args": ["/home/vnijs/gh/pyrsm/mcp-server/server_regression.py"]
    }
  }
}
```

---

## Benefits

✅ **No manual path typing** - Select files in Django UI
✅ **Secure paths** - Django validates all file access
✅ **Session persistence** - Selections survive across requests
✅ **Multi-file support** - Analyze multiple datasets
✅ **MCP state management** - Load once, analyze many times
✅ **Clean separation** - Django handles UI, MCP handles computation

---

## Next Steps

1. **Start Django**: `cd ~/gh/django-sfiles && python manage.py runserver`
2. **Select files**: Browse to http://localhost:8000/filebrowser/
3. **Get paths**: `curl http://localhost:8000/filebrowser/api/mcp-file-info/`
4. **Analyze in Claude Code**: "Load and analyze the selected files"
