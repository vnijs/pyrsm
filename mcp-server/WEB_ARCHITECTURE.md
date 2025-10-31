# Web-Based AI Statistical Analysis Platform Architecture

## Executive Summary

This document outlines the architecture for transforming the Jupyter-based MCP statistical analysis tool into a comprehensive web platform for education. The platform prioritizes **novice users (students)** while supporting expert users, providing AI-driven guidance through natural language interactions.

**Technology Stack**: Django + HTMX + Alpine.js + Django Channels + MCP Tools

**Key Design Principles**:
1. Progressive disclosure (novices see wizards, experts see dashboards)
2. AI-first interaction (natural language → guided workflows)
3. Full collaboration (teams, state persistence, project management)
4. Build on existing infrastructure (integrate with current Django sites)

---

## 1. Why This Stack?

### Django
- **Already in use**: Your existing infrastructure (`rsm-django-components`, `django-sfiles`)
- **Educational focus**: Built-in auth, user management, course/project organization
- **Mature ecosystem**: ORM, migrations, admin, testing
- **Security**: CSRF protection, SQL injection prevention, secure file handling

### HTMX
- **Server-driven**: Keep business logic in Python (not duplicated in JS)
- **Minimal JS**: Students/instructors don't need to learn React/Vue
- **Progressive enhancement**: Works without JS, enhanced with JS
- **Existing patterns**: Already used in `rsm-django-radiant` (probability calculator)

### Alpine.js
- **Reactive UI**: Handle form dynamics (dependent dropdowns, conditional visibility)
- **State management**: localStorage persistence for user preferences
- **Lightweight**: 15KB minified, no build step
- **Existing work**: You have `reactive-calculator.js` component ready to reuse

### Django Channels
- **Real-time collaboration**: Multiple users can work on same project
- **Async support**: Long-running AI/statistical computations don't block
- **WebSocket integration**: Live updates for analysis progress
- **State synchronization**: Share analysis state across team members

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   Wizard     │  │  Dashboard   │  │Report Builder│          │
│  │  (Novices)   │  │  (Explore)   │  │ (Ace Editor) │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Interaction Layer                           │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  HTMX (Server-Driven Partials)                         │     │
│  │  - Form updates, result displays, dynamic content      │     │
│  └────────────────────────────────────────────────────────┘     │
│  ┌────────────────────────────────────────────────────────┐     │
│  │  Alpine.js (Client-Side Reactivity)                    │     │
│  │  - Form dependencies, conditional visibility, state    │     │
│  └────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                       Django Application Layer                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │  Views   │  │  Forms   │  │  Models  │  │   API    │       │
│  │ (Logic)  │  │(Validation)│ │ (State)  │  │(Endpoints)│       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      AI & Computation Layer                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  LLM Service │  │  MCP Server  │  │    Celery    │          │
│  │ (Gemini API) │  │(pyrsm tools) │  │(Background)  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                        Data & State Layer                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │PostgreSQL│  │  Redis   │  │   Files  │  │WebSockets│       │
│  │(Projects)│  │ (Cache)  │  │(Uploads) │  │(Collab)  │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Interaction Modes

### Mode 1: Wizard (Novices - Students)

**Goal**: Step-by-step guided analysis with AI assistance

**Flow**:
```
1. Welcome screen with goal selection
   ↓
2. AI asks clarifying questions (dataset, variables, hypothesis)
   ↓
3. Progressive steps with validation
   ↓
4. Results with AI interpretation
   ↓
5. Suggested next steps (clickable)
```

**UI Components**:
- Stepper/progress bar (Bootstrap)
- Single-step forms (HTMX updates)
- AI chat panel (right sidebar)
- Visual feedback (loading states, validation)

**Example Workflow**:
```
Step 1: "What do you want to analyze?"
  → Student: "I want to see if there's a difference in salaries"

Step 2: "Which dataset?"
  → [Dropdown with datasets] → Student selects "salary"

Step 3: "Which groups do you want to compare?"
  → [Multi-select: rank, discipline, sex] → Student picks "rank"

Step 4: "AI suggests: Compare means test"
  → [Show explanation] → Student confirms

Step 5: Results + Interpretation
  → [Output + AI explanation] → Suggested next steps as buttons
```

### Mode 2: Dashboard (Experts - Exploration)

**Goal**: Flexible workspace for ad-hoc analysis

**Layout**:
```
┌────────────────────────────────────────────┐
│  [Dataset Selector] [Variable Panel]      │
├──────────────────┬─────────────────────────┤
│                  │                         │
│  Analysis Panel  │   Results & Plots      │
│  (Left Sidebar)  │   (Main Area)          │
│                  │                         │
│  - Data          │  [Tabs: Summary|Plot|  │
│  - Basics        │   Code|Report]          │
│  - Model         │                         │
│  - Multivariate  │                         │
│                  │                         │
└──────────────────┴─────────────────────────┘
```

**Features**:
- Radiant-style menu structure
- Instant updates (HTMX)
- AI assistant available on-demand
- Code generation visible
- Export to report

### Mode 3: Report Builder (Documentation)

**Goal**: Create reproducible analysis documents

**Components**:
- Ace Editor (Monaco alternative: lighter weight)
- Live preview (Markdown → HTML)
- Code blocks execute inline
- Export to PDF/HTML/Quarto

---

## 4. State Management

### Server-Side State (Django Models)

```python
# Project/Session persistence
class AnalysisProject(models.Model):
    user = models.ForeignKey(User)
    team = models.ForeignKey(Team, null=True)  # Collaboration
    name = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)

class AnalysisSession(models.Model):
    project = models.ForeignKey(AnalysisProject)
    active_dataset = models.CharField(max_length=100)
    state_snapshot = models.JSONField()  # Full UI state

class DatasetUpload(models.Model):
    project = models.ForeignKey(AnalysisProject)
    file = models.FileField(upload_to='datasets/')
    name = models.CharField(max_length=100)
    uploaded_at = models.DateTimeField(auto_now_add=True)

class AnalysisResult(models.Model):
    session = models.ForeignKey(AnalysisSession)
    analysis_type = models.CharField(max_length=50)  # 'regress', 'single_mean', etc.
    model_id = models.CharField(max_length=100)
    parameters = models.JSONField()
    results = models.JSONField()  # Serialized output
    code = models.TextField()  # Generated Python code
    created_at = models.DateTimeField(auto_now_add=True)
```

### Client-Side State (Alpine.js + localStorage)

```javascript
// Per-user UI preferences
Alpine.store('analysisState', {
    activeDataset: null,
    selectedVariables: [],
    analysisMode: 'wizard',  // or 'dashboard', 'report'

    // UI dynamics state
    formState: {
        distribution: 'Normal',
        distributionParams: {},
        inputType: 'values',
        conditionalVisible: {}
    },

    // Persist to localStorage
    save() { localStorage.setItem('analysisState', JSON.stringify(this)) },
    restore() { /* Load from localStorage */ }
})
```

### WebSocket State (Real-time Collaboration)

```python
# channels/consumers.py
class AnalysisConsumer(AsyncWebsocketConsumer):
    async def receive(self, text_data):
        data = json.loads(text_data)

        # Broadcast state changes to team members
        await self.channel_layer.group_send(
            self.project_group_name,
            {
                'type': 'analysis_update',
                'user': self.user.username,
                'action': data['action'],
                'state': data['state']
            }
        )
```

---

## 5. HTMX + Alpine.js UI Dynamics Patterns

### Pattern 1: Dependent Dropdowns (Variable Exclusion)

**Problem**: Variable selected in dropdown A cannot appear in dropdown B

**Solution**:

```html
<div x-data="{
    allColumns: ['age', 'salary', 'rank', 'sex'],
    selectedX: null,
    selectedY: null,

    get availableForY() {
        return this.allColumns.filter(c => c !== this.selectedX)
    },
    get availableForX() {
        return this.allColumns.filter(c => c !== this.selectedY)
    }
}">
    <!-- X Variable -->
    <select x-model="selectedX" @change="debounceUpdate()">
        <option value="">Select X variable</option>
        <template x-for="col in availableForX" :key="col">
            <option :value="col" x-text="col"></option>
        </template>
    </select>

    <!-- Y Variable -->
    <select x-model="selectedY" @change="debounceUpdate()">
        <option value="">Select Y variable</option>
        <template x-for="col in availableForY" :key="col">
            <option :value="col" x-text="col"></option>
        </template>
    </select>
</div>
```

**State Restoration**:
```javascript
// On page load
init() {
    const saved = localStorage.getItem('variableSelection')
    if (saved) {
        const state = JSON.parse(saved)
        // Restore in order: allColumns → selectedX → selectedY
        this.allColumns = state.allColumns
        this.$nextTick(() => {
            this.selectedX = state.selectedX
            this.$nextTick(() => {
                this.selectedY = state.selectedY
            })
        })
    }
}
```

### Pattern 2: Conditional Visibility

**Problem**: Show input C only when input A equals X

**Solution**:

```html
<div x-data="{
    testType: 'ttest',
    sampleType: null,

    get showSampleType() {
        return ['ttest', 'wilcoxon'].includes(this.testType)
    }
}">
    <!-- Test Type -->
    <select x-model="testType" @change="handleTestChange()">
        <option value="ttest">t-test</option>
        <option value="wilcoxon">Wilcoxon test</option>
        <option value="chisquare">Chi-square</option>
    </select>

    <!-- Conditional: Sample Type (only for t-test/wilcoxon) -->
    <div x-show="showSampleType" x-transition>
        <select x-model="sampleType">
            <option value="independent">Independent</option>
            <option value="paired">Paired</option>
        </select>
    </div>
</div>
```

**State Restoration with Conditionals**:
```javascript
restoreState() {
    const saved = JSON.parse(localStorage.getItem('testState'))

    // Step 1: Restore primary selection
    this.testType = saved.testType

    // Step 2: Wait for DOM to update conditional visibility
    this.$nextTick(() => {
        // Step 3: Restore conditional fields only if they should be visible
        if (this.showSampleType && saved.sampleType) {
            this.sampleType = saved.sampleType
        }
    })
}
```

### Pattern 3: HTMX Partial Update + Alpine State Sync

**Problem**: HTMX swaps HTML but loses Alpine state

**Solution**: Use Alpine `x-data` with persistence

```html
<!-- Main container with state -->
<div x-data="analysisForm()" x-init="restoreState()">

    <!-- Distribution selector (triggers HTMX) -->
    <select
        x-model="formData.distribution"
        @change="saveState(); loadDistributionParams()"
        hx-post="/api/get-distribution-params/"
        hx-target="#param-container"
        hx-trigger="change">
        <option value="Normal">Normal</option>
        <option value="Binomial">Binomial</option>
    </select>

    <!-- HTMX target: parameters swap in -->
    <div id="param-container"
         hx-trigger="load"
         @htmx:afterSwap="restoreParamValues()">
        <!-- Server renders parameter inputs here -->
    </div>
</div>

<script>
function analysisForm() {
    return {
        formData: {
            distribution: 'Normal',
            params: {}
        },

        saveState() {
            localStorage.setItem('formState', JSON.stringify(this.formData))
        },

        restoreState() {
            const saved = localStorage.getItem('formState')
            if (saved) this.formData = JSON.parse(saved)
        },

        // After HTMX swaps new inputs, restore their values
        restoreParamValues() {
            const params = this.formData.params[this.formData.distribution]
            if (params) {
                Object.entries(params).forEach(([name, value]) => {
                    const input = document.querySelector(`[name="${name}"]`)
                    if (input) input.value = value
                })
            }
        }
    }
}
</script>
```

**Key Insight**: Use Alpine for state management + HTMX for DOM updates. Let server render correct HTML structure, Alpine restores values.

---

## 6. MCP Integration Architecture

### Current State (Jupyter)

```
%%mcp magic → LLM (Gemini) → Tool selection → Code generation → Auto-execute
```

### Web Platform Integration

```
User input → Django view → LLM service → MCP tools → Celery task → WebSocket update → UI
```

**Django View Example**:

```python
# views.py
from .services.ai_service import AIAnalysisService
from .tasks import run_analysis_async

def analyze_data(request):
    """Handle natural language analysis request"""
    user_prompt = request.POST.get('prompt')
    project_id = request.session['active_project']

    # Synchronous: Quick LLM tool selection
    ai_service = AIAnalysisService()
    tool_call = ai_service.select_tool(
        prompt=user_prompt,
        context=get_analysis_context(project_id)
    )

    # Asynchronous: Execute analysis in background
    task = run_analysis_async.delay(
        project_id=project_id,
        tool_name=tool_call['name'],
        arguments=tool_call['arguments']
    )

    # Return immediate response with task ID
    return JsonResponse({
        'task_id': task.id,
        'status': 'processing',
        'estimated_time': 5  # seconds
    })
```

**Celery Task**:

```python
# tasks.py
from celery import shared_task
from .mcp_bridge import call_mcp_tool
from channels.layers import get_channel_layer

@shared_task(bind=True)
def run_analysis_async(self, project_id, tool_name, arguments):
    """Execute MCP tool and broadcast results"""

    # Call MCP server (same as Jupyter version)
    result = call_mcp_tool(tool_name, **arguments)

    # Store result in database
    AnalysisResult.objects.create(
        project_id=project_id,
        analysis_type=tool_name,
        parameters=arguments,
        results=result['output'],
        code=result['generated_code']
    )

    # Broadcast to WebSocket group
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f'project_{project_id}',
        {
            'type': 'analysis_complete',
            'result': result
        }
    )

    return result
```

**WebSocket Consumer**:

```python
# consumers.py
class AnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.project_id = self.scope['url_route']['kwargs']['project_id']
        await self.channel_layer.group_add(
            f'project_{self.project_id}',
            self.channel_name
        )
        await self.accept()

    async def analysis_complete(self, event):
        """Broadcast analysis results to connected clients"""
        await self.send(text_data=json.dumps({
            'type': 'result',
            'data': event['result']
        }))
```

**Frontend (Alpine.js)**:

```javascript
// Real-time result updates
Alpine.data('analysisWorkspace', () => ({
    results: [],
    socket: null,

    init() {
        this.connectWebSocket()
    },

    connectWebSocket() {
        const projectId = this.projectId
        this.socket = new WebSocket(
            `ws://${window.location.host}/ws/analysis/${projectId}/`
        )

        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data)
            if (data.type === 'result') {
                this.results.push(data.data)
                // HTMX can update results panel
                htmx.trigger('#results-panel', 'newResult', data.data)
            }
        }
    }
}))
```

---

## 7. Component Structure

```
pyrsm-web/
├── manage.py
├── config/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py (Channels routing)
│   └── celery.py
│
├── apps/
│   ├── core/              # Base models, auth, teams
│   ├── analysis/          # Main analysis app
│   │   ├── models.py      # Project, Session, Result
│   │   ├── views/
│   │   │   ├── wizard.py  # Wizard mode views
│   │   │   ├── dashboard.py
│   │   │   └── report.py
│   │   ├── forms/
│   │   │   ├── dataset_form.py
│   │   │   └── analysis_forms.py
│   │   ├── services/
│   │   │   ├── ai_service.py     # LLM integration
│   │   │   └── mcp_bridge.py     # MCP tool calling
│   │   ├── consumers.py   # WebSocket consumers
│   │   ├── tasks.py       # Celery tasks
│   │   └── templates/
│   │       ├── wizard/
│   │       ├── dashboard/
│   │       └── partials/  # HTMX partials
│   │
│   ├── datasets/          # File upload, management
│   │   ├── models.py      # DatasetUpload, DataFile
│   │   ├── views.py       # Browse, upload, select
│   │   └── api.py         # RESTful endpoints
│   │
│   └── collaboration/     # Teams, sharing, comments
│       ├── models.py      # Team, Membership, Comment
│       └── views.py
│
├── static/
│   ├── js/
│   │   ├── alpine-components/
│   │   │   ├── analysis-form.js
│   │   │   ├── variable-selector.js
│   │   │   └── reactive-calculator.js  # From rsm-django-components
│   │   └── htmx-handlers.js
│   ├── css/
│   └── vendor/
│
├── templates/
│   ├── base.html
│   ├── analysis/
│   └── components/
│
└── tests/
    ├── test_ui_dynamics.py      # Test dependent dropdowns
    ├── test_state_restoration.py
    └── test_mcp_integration.py
```

---

## 8. Educational Features

### For Students

1. **Guided Learning Paths**
   - Pre-configured analysis workflows
   - AI explanations at each step
   - "Why are we doing this?" tooltips
   - Example datasets with learning objectives

2. **Assignment Mode**
   - Instructor creates assignment template
   - Students complete steps
   - Auto-grading based on analysis correctness
   - Submit report for review

3. **Collaboration**
   - Group projects
   - Share workspaces
   - Comment on results
   - Peer review

### For Instructors

1. **Course Management**
   - Create course projects
   - Assign datasets
   - Template analyses
   - Track student progress

2. **Assessment**
   - View submission history
   - Compare approaches
   - Provide feedback inline
   - Export grades

3. **Content Creation**
   - Build interactive tutorials
   - Embed in course sites
   - Link to lecture materials

---

## 9. Deployment Considerations

### Development
```bash
# Docker Compose
services:
  web:
    build: .
    command: python manage.py runserver
    volumes:
      - .:/code

  celery:
    build: .
    command: celery -A config worker

  redis:
    image: redis:alpine

  postgres:
    image: postgres:15
```

### Production
- **Web**: Gunicorn + Nginx
- **WebSockets**: Daphne (ASGI server)
- **Celery**: Supervisor or systemd
- **Database**: PostgreSQL (managed service)
- **Cache**: Redis (managed service)
- **Files**: S3-compatible storage
- **Monitoring**: Sentry, Prometheus

---

## 10. Migration Path from Jupyter

### Phase 1: Parallel Development
- Keep Jupyter %%mcp working
- Build web version alongside
- Shared MCP server

### Phase 2: Feature Parity
- All MCP tools available in web
- Import Jupyter notebooks
- Export to Jupyter

### Phase 3: Web-First
- Default to web platform
- Jupyter for advanced users
- API for programmatic access

---

## 11. Next Steps

### Immediate (This Branch)
1. ✅ Create this architecture doc
2. Create `examples/htmx-alpine-patterns/` Django app
3. Build 3 small UI pattern examples
4. Test and iterate

### Short Term (2-3 weeks)
1. Core Django models (Project, Session, Result)
2. Basic wizard flow (one analysis end-to-end)
3. MCP integration (reuse existing server)
4. File upload + dataset management

### Medium Term (1-2 months)
1. Dashboard mode
2. Real-time collaboration (Channels)
3. Report builder
4. Educational features (assignments)

### Long Term (3-6 months)
1. Full deployment
2. Integration with existing course sites
3. Advanced features (ML models, custom analyses)
4. Mobile-responsive design

---

## 12. Success Metrics

### Technical
- Page load < 2s
- Analysis results < 10s
- 99.9% uptime
- Zero data loss

### Educational
- 80% task completion rate (students finish analyses)
- Reduced time to first insight (vs manual coding)
- Positive feedback on AI guidance
- Adoption in multiple courses

### User Experience
- Novices can complete analyses without instructor help
- Experts prefer web over Jupyter for teaching
- Collaboration features actively used
- Report exports meet publication standards

---

## Appendix A: Technology Alternatives Considered

| Technology | Alternative | Why Not Chosen |
|------------|-------------|----------------|
| Django | Flask | Need full-featured framework for education |
| HTMX | React | More complexity, harder to maintain |
| Alpine.js | Vue.js | Heavier, requires build step |
| PostgreSQL | MySQL | Better JSON support, more features |
| Celery | RQ | Less mature, fewer integrations |
| Channels | Polling | Real-time is better UX |
| Gunicorn | uWSGI | Simpler configuration |

## Appendix B: References

- **Existing Work**: `/home/vnijs/gh/rsm-django-components`
  - `rsm-django-radiant`: Probability calculator with HTMX+Alpine
  - `rsm-django-htmx`: Reusable components
  - `reactive-calculator.js`: State management pattern

- **MCP Server**: `/home/vnijs/gh/pyrsm/mcp-server`
  - `server_regression.py`: Tool definitions
  - `mcp_bridge_magic.py`: Jupyter integration

- **Django File Management**: `/home/vnijs/gh/django-sfiles`
  - Secure file browsing
  - Upload/download patterns
