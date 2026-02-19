# FabricFlow Scripts

This directory contains CI/CD validation and utility scripts for the FabricFlow shared library.

## Available Scripts

### `generate_manifest.py` (Primary Script)

**Purpose:** Generates metadata JSON files for tasks and workflows using AST parsing.

This is the primary script used in CI. It validates tasks implicitly - if metadata can be generated successfully, the task has valid syntax, correct decorator usage, and proper structure.

**Features:**
- AST-based parsing (no imports required)
- Works without installing task dependencies (cv2, torch, etc.)
- Generates category-organized metadata for tasks
- Preserves manual edits in existing metadata files
- Task validation happens during generation

**Usage:**
```bash
# Generate all metadata (tasks + workflows)
python scripts/generate_manifest.py

# Generate only task metadata
python scripts/generate_manifest.py --tasks-only

# Generate only workflow metadata
python scripts/generate_manifest.py --workflows-only
```

**Output:**
- `tasks/_metadata/{category}/{task_id}.json`
- `workflows/_metadata/{workflow_id}.json`

### `validate_workflow.py`

Validates workflow JSON files by checking:
- JSON syntax
- Workflow schema (inputs, tasks, outputs structure)
- Required task fields (name, task type)
- Task dependency validity
- DAG validation (no cycles, valid dependency graph)
- Task output references

**Usage:**
```bash
# Validate all workflows
python scripts/validate_workflow.py --all

**Exit codes:**
- `0`: All validations passed
- `1`: One or more validations failed

---

## CI/CD Integration

The CI workflow uses these scripts in the following order:

1. **validate-workflows** - Validates workflow JSON and DAG logic
2. **generate-manifests** - Generates metadata and validates it's committed
   - Task validation happens implicitly during generation
   - If metadata generation succeeds, the task is valid
3. **lint** - Code quality checks

**Why no separate task validation?**  
Task validation is implicit in metadata generation. If `generate_manifest.py` successfully creates metadata for a task, it proves the task has valid syntax, correct decorator usage, and proper structure. This eliminates redundancy and speeds up CI.

---

## Development Workflow

```bash
# 1. Add or modify tasks/workflows
# 2. Generate updated metadata
python scripts/generate_manifest.py

# 3. Commit metadata along with code
git add tasks/_metadata/ workflows/_metadata/
git commit -m "Update task and metadata"

# 4. CI will verify metadata matches source
```

---

## Metadata Structure

### Task Metadata (`tasks/_metadata/{category}/{task_id}.json`)

```json
{
  "id": "load_image",
  "version": "1.0.0",
  "category": "io",
  "display_name": "Load Image",
  "description": "Load an image from file path",
  "author": "FabricFlow Team",
  "inputs": { ... },
  "outputs": { ... },
  "dependencies": {
    "python": ["opencv-python>=4.8.0"],
    "system": []
  }
}
```

### Workflow Metadata (`workflows/_metadata/{workflow_id}.json`)

```json
{
  "id": "stable_diffusion",
  "version": "1.0.0",
  "category": "diffusion",
  "display_name": "Stable Diffusion",
  "description": "Generate images using Stable Diffusion",
  "required_tasks": ["load_stable_diffusion_pipeline", "generate_image_from_prompt"],
  "dependencies": {
    "python": ["diffusers>=0.20.0", "torch>=2.0.0"],
    "models": []
  }
}
```

---

## Requirements

- Python 3.8+
- No external dependencies for core scripts
- AST parsing works without task dependencies (cv2, torch, etc.)

---

## Notes

**Registry Architecture:**  
This repository serves as a task/workflow registry. Tasks execute in FabricFlow, not here. Therefore, validation uses AST parsing rather than importing modules, avoiding dependency requirements.

**Metadata is Source of Truth:**  
Generated metadata files should be committed to git. CI validates that committed metadata matches source files.
