## Description

Please provide a clear description of what this PR does.

## Type of Change

- [ ] New task
- [ ] New workflow  
- [ ] Bug fix
- [ ] Enhancement to existing task/workflow
- [ ] Documentation update
- [ ] Other (please describe)

## Checklist

### For New Tasks

- [ ] Task file created in appropriate category (`tasks/<category>/<task_name>.py`)
- [ ] Task uses `@task` decorator with complete metadata
- [ ] Task follows function-based pattern (not class-based)
- [ ] Metadata JSON created (`tasks/_metadata/<task_name>.json`)
- [ ] All required metadata fields are present
- [ ] Dependencies listed in metadata
- [ ] Task has comprehensive docstring
- [ ] Unit tests added (`tests/tasks/test_<task_name>.py`)
- [ ] Validation passes (`python scripts/validate_task.py <task_name>`)
- [ ] Task tested end-to-end in a workflow

### For New Workflows

- [ ] Workflow JSON created (`workflows/<workflow_name>.json`)
- [ ] Workflow metadata created (`workflows/_metadata/<workflow_name>.json`)
- [ ] All required tasks are listed in metadata `required_tasks`
- [ ] Workflow forms valid DAG (no cycles)
- [ ] Example inputs provided in metadata
- [ ] Validation passes (`python scripts/validate_workflow.py <workflow_name>`)
- [ ] Workflow tested end-to-end
- [ ] Screenshots or logs included (if applicable)

### General

- [ ] Branch is up to date with base branch
- [ ] Commit messages are clear and descriptive
- [ ] CHANGELOG.md updated (if applicable)
- [ ] No merge conflicts
- [ ] All CI checks pass

## Testing

Describe how you tested your changes:

```
# Commands used for testing

```

### Test Results

Provide test output or screenshots:

```
# Test output

```

## Screenshots (if applicable)

<!-- Add screenshots here -->

## Additional Notes

<!-- Any additional information, context, or notes for reviewers -->

## Related Issues

Closes #(issue number)
Relates to #(issue number)
