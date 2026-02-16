# Agent Task Delegation System

This directory contains agent task definitions for automated execution of the IMPLEMENTATION_ROADMAP.md.

## Quick Start

```bash
# Show current status
python3 scripts/orchestrator.py status

# Execute next ready task
python3 scripts/orchestrator.py next

# Auto-execute all ready tasks
python3 scripts/orchestrator.py auto
```

## Directory Structure

```
admin/agent-tasks/
├── README.md                          (this file)
├── tasks/                             (individual agent task specs)
│   ├── phase1-task1.2.2-uss-decision.md
│   ├── phase1-task1.3-claude-fix.md
│   ├── phase2-task2.1.1-impl-audit.md
│   └── ... (all agent tasks)
├── examples/                          (example executions)
│   └── task-execution-example.md
└── agent-integration.md               (how to integrate with agents)
```

## Task Execution Flow

```
orchestrator.py
    ↓
  Identify next ready task
    ↓
  Load task definition from tasks/
    ↓
  If manual → prompt user
  If agent → delegate to agent
    ↓
  Execute task
    ↓
  Run validation
    ↓
  Update status in admin/tasks.json
    ↓
  Calculate new health score
    ↓
  Proceed to next task or report
```

## Agent Tasks by Phase

### Phase 1: Stabilization (Week 1)
- **Manual**: 1.1 (git migration), 1.2.1 (delete .audit-venv)
- **consolidation-architect**: 1.2.2 (USS decision)
- **documentation-architect**: 1.3.1, 1.3.2, 1.4 (docs fixes)
- **integration-test**: 1.5 (validation)

### Phase 2: Integration (Weeks 2-3)
- **code-review-specialist**: 2.1.1 (implementation audit)
- **documentation-architect**: 2.1.2, 2.2, 2.3 (docs + consolidation)
- **consolidation-architect**: 2.4 (script merge)

### Phase 3: Enhancement (Week 4)
- **integration-test + bare-metal-runtime**: 3.1 (test coverage)
- **documentation-architect**: 3.2 (catalog)
- **chief-architect**: 3.3 (USS integration)
- **bare-metal-runtime**: 3.4 (benchmarks)

## Status Tracking

All task state stored in `admin/tasks.json`. View with:

```bash
cat admin/tasks.json | jq '.tasks[] | select(.status != "complete")'
```

## See Documentation

- `IMPLEMENTATION_ROADMAP.md` - Complete roadmap with all tasks
- `scripts/orchestrator.py` - Task execution engine
- `tasks/` - Individual agent task specifications
