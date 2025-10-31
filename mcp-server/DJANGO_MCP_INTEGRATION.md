# Django File Browser & MCP Server Integration Analysis

## Directory Structures

### django-sfiles
**Location**: `/home/vnijs/gh/django-sfiles/`

```
django-sfiles/
├── filebrowser/              # Main app
│   ├── views.py             # 672 lines - core file operations
│   ├── api.py               # 121 lines - public API interface
│   ├── urls.py              # REST endpoints
│   ├── utils.py             # SecurePathHandler (145 lines)
│   ├── models.py            # (empty - no DB)
│   ├── templatetags/
│   │   └── file_icons.py    # Font Awesome icon mapping
│   ├── templates/filebrowser/
│   │   ├── index.html
│   │   ├── directory_contents.html    # HTMX partial
│   │   ├── file_content.html
│   │   ├── modal_browser.html
│   │   ├── modals.html
│   │   ├── new_file.html
│   │   └── example.html
│   ├── admin.py
│   ├── apps.py
│   └── tests.py
├── sfiles/                   # Django project
│   ├── settings.py
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── examples/
│   └── example_integration.py    # Shows Django integration
├── tests/
├── README.md
└── pyproject.toml
```

### rsm-django-files (WIP Component)
**Location**: `/home/vnijs/gh/rsm-django-components/components/rsm-django-files/`

```
rsm-django-files/
├── PLAN.md                  # Comprehensive architecture plan
├── TODO.md                  # Phase-by-phase implementation (140 tasks)
├── CC_INSTANCE_FILES.md
├── pyproject.toml
└── examples/
    └── .gitkeep             # NOT YET IMPLEMENTED
```

### pyrsm MCP Server
**Location**: `/home/vnijs/gh/pyrsm/mcp-server/`

```
pyrsm/mcp-server/
├── server_regression.py     # Main MCP server (~450 lines)
├── server.py               # Original simple server
├── test_*.py               # Validation scripts
├── INTEGRATION_DESIGN.md   # File loading modes
├── REGRESSION_TOOLS.md     # Tool documentation
├── README.md
├── Makefile
└── pyproject.toml
```

---

## Key Components

### 1. django-sfiles: SecurePathHandler (utils.py)

**Location**: `/home/vnijs/gh/django-sfiles/filebrowser/utils.py`

**Core Security Features**:
- Path traversal prevention using `os.path.commonpath()`
- Symlink detection and validation
- Boundary checking against allowed roots
- Hidden file filtering (dot-files)
- MIME type detection

**Key Methods**:
```python
validate_path(path: str, root: str) -> str
    # Returns absolute path if valid, raises PermissionDenied

get_directory_contents(path: str, root: str) -> Dict
    # Returns: {
    #     'path': relative_path,
    #     'directories': [...],
    #     'files': [{name, path, size, modified, mime_type}, ...]
    # }

get_breadcrumbs(path: str) -> List[Dict]
    # Generates navigation breadcrumbs

is_safe_filename(filename: str) -> bool
    # Validates filename for path traversal attempts
```

**Configuration**:
```python
# Settings expected:
FILEBROWSER_ROOTS = {
    'home': os.path.expanduser('~/'),
    'data': os.path.join(BASE_DIR, 'data')
}
FILEBROWSER_MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB default
```

---

### 2. django-sfiles: View Endpoints (views.py)

**Location**: `/home/vnijs/gh/django-sfiles/filebrowser/views.py`

**File Operations**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `browse_directory` | GET | HTMX: Get directory contents as HTML partial |
| `get_directory_json` | GET | JSON: Directory structure for API |
| `select_file` | POST | Session: Add file to selection |
| `upload_file` | POST | Upload single/multiple files |
| `download_file` | GET | Download single file |
| `download_multiple` | POST | ZIP multiple files |
| `open_file` | GET | Display file content (1MB limit) |
| `save_file` | POST | Write content to file |
| `create_folder` | POST | Create directory |
| `modal_browse` | GET | Modal file picker view |
| `modal_upload` | POST | Upload from modal |
| `check_file_exists` | GET | Verify file existence |

**Session Management**:
```python
request.session['selected_files'] = [
    {
        'root': 'home',
        'path': 'documents/file.csv',
        'absolute_path': '/home/user/documents/file.csv',
        'name': 'file.csv',
        'timestamp': '2025-10-28T...'
    },
    ...
]
```

---

### 3. django-sfiles: Public API (api.py)

**Location**: `/home/vnijs/gh/django-sfiles/filebrowser/api.py`

**FileBrowserAPI Interface**:
```python
class FileBrowserAPI:
    @staticmethod
    def get_selected_files(request) -> List[Dict]
        # Returns session selected files
    
    @staticmethod
    def get_selected_file_paths(request) -> List[str]
        # Returns absolute paths
    
    @staticmethod
    def clear_selection(request) -> None
    
    @staticmethod
    def add_file_to_selection(request, root, path) -> Dict
        # Add programmatically
    
    @staticmethod
    def get_file_content(root, path, encoding='utf-8') -> str
        # Read file safely
    
    @staticmethod
    def save_file_content(root, path, content, create_dirs=True) -> str
        # Write file safely
```

---

### 4. pyrsm MCP Server: File Loading (server_regression.py)

**Location**: `/home/vnijs/gh/pyrsm/mcp-server/server_regression.py`

**State Management**:
```python
# Persistent registries during server lifetime
DATA_STORE = {
    'sample': <DataFrame>,
    'diamonds': <DataFrame>,
    'loaded_file': <DataFrame>,  # From file uploads
    ...
}

MODEL_STORE = {
    'reg_abc123': {
        'obj': <regression object>,
        'rvar': 'price',
        'evar': ['carat', 'depth'],
        'fitted_at': '2025-10-28 10:30:00',
        'metadata': {...}
    },
    ...
}
```

**MCP Tools**:
1. `data_load` - Load pyrsm built-in datasets (50+)
2. `data_load_file` - Load CSV/Excel/Parquet/JSON from file path
3. `data_list` - List loaded datasets
4. `data_info` - Get dataset metadata
5. `regress_fit` - Fit regression model
6. `regress_summary` - Get model stats (no refitting)
7. `regress_plot` - Generate diagnostic plots
8. `regress_list` - List stored models

---

## Integration Patterns

### Pattern 1: Direct File Path Integration

**Use Case**: VS Code terminal, Claude Code command-line

**Flow**:
```
User: "Load data from ~/Downloads/sales.csv"
  ↓
AI calls: data_load_file(file_path='/home/vnijs/Downloads/sales.csv')
  ↓
MCP server: Loads CSV, stores in DATA_STORE['sales']
  ↓
Response: Dataset loaded (2500 rows × 12 columns)
  ↓
User: "Fit regression on this data"
AI calls: regress_fit(data_name='sales', rvar='revenue', evar=['cost', 'quantity'])
```

**Status**: ✅ WORKING (tested in `test_file_loading.py`)

**Implementation**: See `server_regression.py:234-292`

---

### Pattern 2: Django File Browser + MCP Integration

**Use Case**: Educational platform where Django hosts file browser UI, AI accesses files via MCP

**Architecture**:
```
Django App (File Browser UI)
    ├── User selects files via HTMX interface
    ├── Files stored in `FILEBROWSER_ROOTS` (e.g., ~/data/)
    └── Session tracks selected files
         
AI Assistant (Claude Code)
    ├── User: "Analyze the files I selected in the Django app"
    ├── MCP calls: data_load_file(file_path='/home/vnijs/data/selected_file.csv')
    ├── MCP server loads from persistent location
    └── Analysis happens with full state persistence
```

**Integration Points**:
1. **File Storage**: Django FileType roots → accessible to MCP via absolute paths
2. **Session Bridge**: Django session selects files → User provides paths to AI
3. **Shared Registry**: Both Django and MCP can work with same data

---

### Pattern 3: Shiny App + MCP Integration

**Use Case**: Web UI for file upload + AI-powered analysis

**Proposed Flow**:
```
Shiny App (pyrsm-genai-shiny)
    1. User uploads file via Shiny UI
    2. File saved to ~/pyrsm_uploads/dataset.csv
    3. Display in Shiny (pandas operations)
         
Claude Code (AI Analysis)
    1. User: "Analyze the uploaded data"
    2. MCP calls: data_load_file(file_path='~/pyrsm_uploads/dataset.csv')
    3. Shared DATA_STORE for both Shiny and AI
    4. AI provides analysis, Shiny displays results
```

**Design Status**: 🚧 Ready for implementation (see `INTEGRATION_DESIGN.md`)

---

## File Upload/Browse Functionality

### django-sfiles Features

**Browsing**:
- Multi-root directory support
- HTMX partial updates (no full refresh)
- Breadcrumb navigation
- Hidden file filtering
- File metadata display (size, modified date, MIME type)
- Modal file picker for selections

**Uploading**:
- Single/multiple file upload
- Drag-and-drop support
- 100MB default size limit (configurable)
- Filename validation
- Directory creation on upload

**Downloading**:
- Single file download
- Multiple files as ZIP archive
- Streaming response for large files
- Timestamp-based ZIP naming

**File Operations**:
- Create folders
- View file content (with 1MB limit)
- Save file content (text files)
- Check file existence
- File selection management via sessions

---

## API Endpoints Summary

### django-sfiles URL Patterns

**Location**: `/home/vnijs/gh/django-sfiles/filebrowser/urls.py`

```python
# Main browsing
path('', views.filebrowser, name='index')
path('browse/<str:root>/', views.browse_directory, name='browse_directory')

# JSON APIs
path('api/<str:root>/directory/', views.get_directory_json, name='get_directory_json')
path('api/<str:root>/select/', views.select_file, name='select_file')
path('api/<str:root>/save/', views.save_file, name='save_file')
path('api/<str:root>/create-folder/', views.create_folder, name='create_folder')
path('api/<str:root>/upload/', views.upload_file, name='upload_file')
path('api/<str:root>/download/', views.download_file, name='download_file')

# Multi-file operations
path('api/download-multiple/', views.download_multiple, name='download_multiple')
path('api/selected/', views.get_selected_files, name='get_selected_files')
path('api/clear-selection/', views.clear_selection, name='clear_selection')

# File content
path('open/<str:root>/', views.open_file, name='open_file')
path('new-file/', views.new_file, name='new_file')

# Modal operations
path('modal/browse/', views.modal_browse, name='modal_browse')
path('modal/upload/', views.modal_upload, name='modal_upload')

# Utilities
path('example/', views.example, name='example')
```

---

## Suggested MCP-Django Integration Approach

### Option A: Lightweight Bridge (Recommended)

**Architecture**:
1. Django FileRouter (read-only) with absolute paths
2. MCP uses `data_load_file()` with Django-managed file paths
3. No database synchronization needed

**Advantages**:
- Minimal coupling
- No schema changes
- File security handled by Django
- Simple to implement

**Implementation**:
```python
# In Django views
def get_file_for_analysis(request, file_id):
    """Return absolute path to file for AI analysis"""
    file_info = get_selected_file(request, file_id)
    return JsonResponse({'absolute_path': file_info['absolute_path']})

# In Claude Code
# User: "Use this file from the Django app"
# MCP calls: data_load_file(file_path='/home/vnijs/data/project1/data.csv')
```

---

### Option B: Embedded Analysis View

**Architecture**:
1. Django hosts file browser
2. Selected files trigger MCP analysis
3. Results displayed in Django interface

**Advantages**:
- Single unified UI
- Better UX flow
- File selection context retained

**Implementation**:
```python
# Django view
def analyze_selected_files(request):
    """Trigger MCP analysis on selected files"""
    files = FileBrowserAPI.get_selected_file_paths(request)
    
    # Call MCP server with file paths
    mcp_response = call_mcp_tool(
        'data_load_file',
        file_path=files[0]
    )
    
    return render(request, 'analysis.html', {
        'mcp_result': mcp_response
    })
```

---

### Option C: Hybrid (Most Flexible)

**Architecture**:
1. Django file browser for selection
2. Direct MCP access via file paths
3. Optional Shiny preview layer

**Enables**:
- Django for file management & selection
- MCP for computation & analysis
- Shiny for visualization

---

## rsm-django-files Implementation Status

**Current State**: 🚧 WIP - Planning phase complete

**What Exists**:
- PLAN.md (26 architectural decisions)
- TODO.md (140 tasks across 24 phases)
- pyproject.toml (dependencies defined)

**Not Yet Implemented**:
- Core views
- Models
- Templates
- Tests
- Services layer

**Timeline Estimate**:
- Phase 1-5 (Core extraction): 2-3 weeks
- Phase 6-10 (Educational features): 2 weeks
- Phase 11-15 (Integration): 2 weeks
- Phase 16-24 (Polish & release): 2 weeks

**Total**: ~8-10 weeks to full release

---

## Recommended MCP Integration Points

### Short Term (Week 1-2)

1. **File Path Exposure**
   - Add MCP tool to list Django-managed file paths
   - Return absolute paths for `data_load_file()`

2. **Session Bridge**
   - Endpoint to export selected files as JSON
   - Include absolute paths for MCP consumption

3. **Upload Directory**
   - Configure `FILEBROWSER_ROOTS['uploads']`
   - MCP `data_load_file()` points here

### Medium Term (Week 3-4)

1. **Analysis Results Storage**
   - MCP analysis results → Django file storage
   - Create `analysis_results/` folder structure

2. **Metadata Tracking**
   - Store analysis metadata in Django
   - Link original data → analysis

### Long Term (Week 5+)

1. **Full rsm-django-files Integration**
   - Use new component architecture
   - MCP-aware file management

2. **Educational Workflows**
   - Assignment submission via Django
   - AI-powered analysis of submissions

---

## Key Files Reference

### django-sfiles
- **Security**: `/home/vnijs/gh/django-sfiles/filebrowser/utils.py` (145 lines)
- **Views**: `/home/vnijs/gh/django-sfiles/filebrowser/views.py` (672 lines)
- **API**: `/home/vnijs/gh/django-sfiles/filebrowser/api.py` (121 lines)
- **URLs**: `/home/vnijs/gh/django-sfiles/filebrowser/urls.py`
- **Example**: `/home/vnijs/gh/django-sfiles/examples/example_integration.py`

### MCP Server
- **Main Server**: `/home/vnijs/gh/pyrsm/mcp-server/server_regression.py` (~450 lines)
- **File Loading**: `/home/vnijs/gh/pyrsm/mcp-server/server_regression.py:234-292`
- **Integration Design**: `/home/vnijs/gh/pyrsm/mcp-server/INTEGRATION_DESIGN.md`

### rsm-django-files
- **Plan**: `/home/vnijs/gh/rsm-django-components/components/rsm-django-files/PLAN.md`
- **Tasks**: `/home/vnijs/gh/rsm-django-components/components/rsm-django-files/TODO.md`
- **Config**: `/home/vnijs/gh/rsm-django-components/components/rsm-django-files/pyproject.toml`

