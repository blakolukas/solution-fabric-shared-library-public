#!/usr/bin/env python3
"""
Workflow validation script for FabricFlow CI pipeline.

This script validates workflow JSON files by:
- Checking JSON syntax
- Validating workflow schema (inputs, tasks, outputs)
- Checking for required fields
- Validating task references and dependencies
- DAG validation (no cycles, valid dependency graph)
"""

import argparse
import json
import sys
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def find_workflow_files(workflows_dir: Path) -> List[Path]:
    """Find all workflow JSON files (excluding _metadata directory)."""
    workflow_files = []
    for file in workflows_dir.glob("*.json"):
        if file.is_file():
            workflow_files.append(file)

    return sorted(workflow_files)


def load_workflow_json(
    file_path: Path,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Load and parse workflow JSON file.

    Returns:
        (workflow_dict, error_message)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
        return workflow, None
    except json.JSONDecodeError as e:
        return None, f"Invalid JSON: {e}"
    except Exception as e:
        return None, f"Error loading file: {e}"


def validate_workflow_schema(workflow: Dict[str, Any], file_path: Path) -> List[str]:
    """
    Validate the basic schema of a workflow definition.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Workflow can have: inputs, tasks, outputs (core) and optional metadata fields
    valid_top_level_keys = {
        "inputs",
        "tasks",
        "outputs",
        "description",
        "_designer",
        "metadata",
        "version",
        "name",
    }
    for key in workflow.keys():
        if key not in valid_top_level_keys:
            errors.append(
                f"Unknown top-level key: '{key}' (valid keys: {valid_top_level_keys})"
            )

    # Validate inputs section (if present)
    if "inputs" in workflow:
        if not isinstance(workflow["inputs"], dict):
            errors.append("'inputs' must be a dictionary")

    # Validate tasks section (if present)
    if "tasks" in workflow:
        if not isinstance(workflow["tasks"], list):
            errors.append("'tasks' must be a list")
        else:
            for idx, task in enumerate(workflow["tasks"]):
                if not isinstance(task, dict):
                    errors.append(f"Task at index {idx} must be a dictionary")
                    continue

                # Validate required task fields
                if "name" not in task:
                    errors.append(f"Task at index {idx} missing required field 'name'")
                if "task" not in task:
                    errors.append(f"Task at index {idx} missing required field 'task'")

                # Validate task structure
                valid_task_keys = {
                    "name",
                    "task",
                    "inputs",
                    "outputs",
                    "dependencies",
                    "cache",
                    "displayName",
                    "_design",
                }
                for key in task.keys():
                    if key not in valid_task_keys:
                        errors.append(
                            f"Task '{task.get('name', f'index_{idx}')}' has unknown key '{key}' "
                            f"(valid keys: {valid_task_keys})"
                        )

                # Validate task fields
                if "inputs" in task and not isinstance(task["inputs"], dict):
                    errors.append(
                        f"Task '{task.get('name', f'index_{idx}')}' 'inputs' must be a dictionary"
                    )

                if "outputs" in task:
                    outputs = task["outputs"]
                    if not isinstance(outputs, (list, dict)):
                        errors.append(
                            f"Task '{task.get('name', f'index_{idx}')}' 'outputs' must be a list or dictionary"
                        )

                if "dependencies" in task:
                    deps = task["dependencies"]
                    if not isinstance(deps, list):
                        errors.append(
                            f"Task '{task.get('name', f'index_{idx}')}' 'dependencies' must be a list"
                        )
                    elif not all(isinstance(d, str) for d in deps):
                        errors.append(
                            f"Task '{task.get('name', f'index_{idx}')}' 'dependencies' must be a list of strings"
                        )

                if "cache" in task and not isinstance(task["cache"], bool):
                    errors.append(
                        f"Task '{task.get('name', f'index_{idx}')}' 'cache' must be a boolean"
                    )

    # Validate outputs section (if present)
    if "outputs" in workflow:
        if not isinstance(workflow["outputs"], dict):
            errors.append("'outputs' must be a dictionary")

    return errors


def validate_task_dependencies(workflow: Dict[str, Any]) -> List[str]:
    """
    Validate task dependencies and check for cycles in the DAG.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if "tasks" not in workflow:
        return errors

    tasks = workflow["tasks"]
    task_names = {task["name"] for task in tasks if "name" in task}

    # Build dependency graph
    dependencies: Dict[str, List[str]] = {}
    for task in tasks:
        if "name" not in task:
            continue

        task_name = task["name"]
        deps = task.get("dependencies", [])

        # Check that all dependencies reference valid tasks
        for dep in deps:
            if dep not in task_names:
                errors.append(f"Task '{task_name}' depends on unknown task '{dep}'")

        dependencies[task_name] = deps

    # Check for cycles using topological sort (Kahn's algorithm)
    indegree = {name: 0 for name in task_names}
    for task_name, deps in dependencies.items():
        indegree[task_name] += len(deps)

    queue = deque([name for name, degree in indegree.items() if degree == 0])
    visited = 0

    # Build reverse dependency map (dependents)
    dependents: Dict[str, List[str]] = defaultdict(list)
    for task_name, deps in dependencies.items():
        for dep in deps:
            dependents[dep].append(task_name)

    while queue:
        current = queue.popleft()
        visited += 1

        for dependent in dependents.get(current, []):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)

    if visited != len(task_names):
        errors.append("Cycle detected in task dependencies (DAG validation failed)")
        # Try to identify which tasks are part of the cycle
        remaining = [name for name, degree in indegree.items() if degree > 0]
        errors.append(f"  Tasks involved in cycle: {', '.join(remaining)}")

    return errors


def validate_task_references(workflow: Dict[str, Any]) -> List[str]:
    """
    Validate that task output references in inputs and workflow outputs are valid.

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    if "tasks" not in workflow:
        return errors

    tasks = workflow["tasks"]
    task_names = {task["name"] for task in tasks if "name" in task}

    # Build task outputs map
    task_outputs: Dict[str, Set[str]] = {}
    for task in tasks:
        if "name" not in task:
            continue

        task_name = task["name"]
        outputs = task.get("outputs", [])

        if isinstance(outputs, list):
            task_outputs[task_name] = set(outputs)
        elif isinstance(outputs, dict):
            # Outputs can be {new_name: old_name} format
            task_outputs[task_name] = set(outputs.keys())
        else:
            task_outputs[task_name] = set()

    # Validate task input references
    for task in tasks:
        if "name" not in task or "inputs" not in task:
            continue

        task_name = task["name"]
        inputs = task["inputs"]

        for input_key, input_value in inputs.items():
            # Check if it's a path reference
            if isinstance(input_value, str):
                # Path references like "tasks.task_name.output" or "inputs.param"
                if input_value.startswith("tasks."):
                    parts = input_value.split(".")
                    if len(parts) >= 3:
                        ref_task = parts[1]
                        ref_output = parts[2]

                        if ref_task not in task_names:
                            errors.append(
                                f"Task '{task_name}' input '{input_key}' references unknown task '{ref_task}'"
                            )
                        elif (
                            ref_task in task_outputs
                            and ref_output not in task_outputs[ref_task]
                        ):
                            errors.append(
                                f"Task '{task_name}' input '{input_key}' references unknown output "
                                f"'{ref_output}' from task '{ref_task}'"
                            )

            # Check if it's a path object
            elif isinstance(input_value, dict) and "path" in input_value:
                path = input_value["path"]
                if isinstance(path, str) and path.startswith("tasks."):
                    parts = path.split(".")
                    if len(parts) >= 3:
                        ref_task = parts[1]
                        ref_output = parts[2]

                        if ref_task not in task_names:
                            errors.append(
                                f"Task '{task_name}' input '{input_key}' references unknown task '{ref_task}'"
                            )
                        elif (
                            ref_task in task_outputs
                            and ref_output not in task_outputs[ref_task]
                        ):
                            errors.append(
                                f"Task '{task_name}' input '{input_key}' references unknown output "
                                f"'{ref_output}' from task '{ref_task}'"
                            )

    # Validate workflow output references
    if "outputs" in workflow:
        for output_key, output_value in workflow["outputs"].items():
            if isinstance(output_value, str) and output_value.startswith("tasks."):
                parts = output_value.split(".")
                if len(parts) >= 3:
                    ref_task = parts[1]
                    ref_output = parts[2]

                    if ref_task not in task_names:
                        errors.append(
                            f"Workflow output '{output_key}' references unknown task '{ref_task}'"
                        )
                    elif (
                        ref_task in task_outputs
                        and ref_output not in task_outputs[ref_task]
                    ):
                        errors.append(
                            f"Workflow output '{output_key}' references unknown output "
                            f"'{ref_output}' from task '{ref_task}'"
                        )

    return errors


def validate_workflow_file(
    file_path: Path, project_root: Path
) -> Tuple[bool, List[str]]:
    """
    Validate a single workflow file.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Load workflow JSON
    workflow, load_error = load_workflow_json(file_path)
    if load_error:
        errors.append(load_error)
        return False, errors

    # Validate schema
    schema_errors = validate_workflow_schema(workflow, file_path)
    errors.extend(schema_errors)

    # Validate task dependencies (DAG)
    dag_errors = validate_task_dependencies(workflow)
    errors.extend(dag_errors)

    # Validate task references
    ref_errors = validate_task_references(workflow)
    errors.extend(ref_errors)

    return len(errors) == 0, errors


def validate_all_workflows(workflows_dir: Path, project_root: Path) -> int:
    """
    Validate all workflow files in the workflows directory.

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    if not workflows_dir.exists():
        print(f"ERROR: Workflows directory not found: {workflows_dir}")
        return 1

    workflow_files = find_workflow_files(workflows_dir)

    if not workflow_files:
        print(f"⚠️  No workflow files found in {workflows_dir}")
        return 0

    print(f"Found {len(workflow_files)} workflow file(s) to validate")
    print()

    all_valid = True
    validated_count = 0
    failed_count = 0

    for workflow_file in workflow_files:
        rel_path = workflow_file.relative_to(project_root)
        is_valid, errors = validate_workflow_file(workflow_file, project_root)

        if is_valid:
            print(f"✅ {rel_path}")
            validated_count += 1
        else:
            print(f"❌ {rel_path}")
            for error in errors:
                print(f"   {error}")
            failed_count += 1
            all_valid = False

    print()
    print(f"{'='*60}")
    print(f"Validation Summary:")
    print(f"  Passed: {validated_count}")
    print(f"  Failed: {failed_count}")
    print(f"  Total:  {len(workflow_files)}")
    print(f"{'='*60}")

    return 0 if all_valid else 1


def validate_specific_workflows(workflow_paths: List[str], project_root: Path) -> int:
    """
    Validate specific workflow files.

    Returns:
        Exit code (0 for success, 1 for failures)
    """
    all_valid = True

    for workflow_path_str in workflow_paths:
        workflow_path = Path(workflow_path_str)
        if not workflow_path.exists():
            print(f"ERROR: File not found: {workflow_path}")
            all_valid = False
            continue

        is_valid, errors = validate_workflow_file(workflow_path, project_root)

        rel_path = (
            workflow_path.relative_to(project_root)
            if workflow_path.is_relative_to(project_root)
            else workflow_path
        )

        if is_valid:
            print(f"✅ {rel_path}")
        else:
            print(f"❌ {rel_path}")
            for error in errors:
                print(f"   {error}")
            all_valid = False

    return 0 if all_valid else 1


def main():
    parser = argparse.ArgumentParser(
        description="Validate FabricFlow workflow files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_workflow.py --all
  python validate_workflow.py workflows/stable_diffusion.json
  python validate_workflow.py workflows/stable_diffusion.json workflows/yolo_object_detection.json
        """,
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate all workflow files in the workflows/ directory",
    )

    parser.add_argument(
        "workflows",
        nargs="*",
        help="Specific workflow file paths to validate",
    )

    args = parser.parse_args()

    # Determine project root
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    workflows_dir = project_root / "workflows"

    if args.all:
        exit_code = validate_all_workflows(workflows_dir, project_root)
    elif args.workflows:
        exit_code = validate_specific_workflows(args.workflows, project_root)
    else:
        parser.print_help()
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
