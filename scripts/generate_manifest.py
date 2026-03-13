#!/usr/bin/env python3
"""
Generate metadata JSON files for FabricFlow tasks and workflows.

This script extracts metadata from @task decorators and workflow JSON files,
generating standardized manifest files in _metadata/ directories.
"""

import argparse
import ast
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# AST PARSING UTILITIES
# ============================================================================


def parse_file(file_path: Path) -> Optional[ast.Module]:
    """Parse a Python file and return its AST."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
        return ast.parse(source, filename=str(file_path))
    except Exception as e:
        print(f"  Warning: Could not parse {file_path.name}: {e}")
        return None


def get_file_source(file_path: Path) -> Optional[str]:
    """Read and return the source code of a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return None


def find_task_decorator(tree: ast.Module) -> Optional[Tuple[ast.FunctionDef, ast.expr]]:
    """
    Find the first function with @task decorator.

    Returns:
        (function_node, decorator_node) or None
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                # Handle @task
                if isinstance(decorator, ast.Name) and decorator.id == "task":
                    return node, decorator
                # Handle @task(...)
                if isinstance(decorator, ast.Call):
                    if (
                        isinstance(decorator.func, ast.Name)
                        and decorator.func.id == "task"
                    ):
                        return node, decorator
    return None


def extract_decorator_kwargs(decorator: ast.expr) -> Dict[str, Any]:
    """Extract keyword arguments from @task(...) decorator."""
    kwargs = {}

    if not isinstance(decorator, ast.Call):
        return kwargs

    for keyword in decorator.keywords:
        if keyword.arg is None:
            continue

        value = keyword.value

        # Extract literal values (Python 3.8+)
        if isinstance(value, ast.Constant):
            kwargs[keyword.arg] = value.value
        elif isinstance(value, ast.List):
            # Extract list of strings
            list_values = []
            for elt in value.elts:
                if isinstance(elt, ast.Constant):
                    list_values.append(elt.value)
            kwargs[keyword.arg] = list_values
        elif isinstance(value, ast.Dict):
            # Extract dict (for parameters, output_types)
            dict_value = {}
            for k, v in zip(value.keys, value.values):
                if isinstance(k, ast.Constant):
                    key = k.value
                else:
                    continue

                if isinstance(v, ast.Constant):
                    dict_value[key] = v.value
            kwargs[keyword.arg] = dict_value

    return kwargs


def extract_function_parameters(
    func_node: ast.FunctionDef,
) -> Dict[str, Dict[str, Any]]:
    """Extract function parameters and their types from function signature."""
    parameters = {}
    args = func_node.args

    # Get defaults (aligned to the right)
    defaults = [None] * (len(args.args) - len(args.defaults)) + list(args.defaults)

    for arg, default in zip(args.args, defaults):
        param_name = arg.arg

        # Skip 'self' and 'context'
        if param_name in ("self", "context"):
            continue

        param_info = {
            "required": default is None,
        }

        # Extract type annotation
        if arg.annotation:
            param_info["type"] = (
                ast.unparse(arg.annotation) if hasattr(ast, "unparse") else "any"
            )
        else:
            param_info["type"] = "any"

        # Extract default value (Python 3.8+)
        if default is not None:
            if isinstance(default, ast.Constant):
                param_info["default"] = default.value
            elif isinstance(default, ast.List):
                param_info["default"] = []
            elif isinstance(default, ast.Dict):
                param_info["default"] = {}

        parameters[param_name] = param_info

    return parameters


def extract_docstring(func_node: ast.FunctionDef) -> Optional[str]:
    """Extract the docstring from a function node."""
    docstring = ast.get_docstring(func_node)
    if docstring:
        # Get first line only
        lines = [line.strip() for line in docstring.strip().split("\n") if line.strip()]
        return lines[0] if lines else None
    return None


def extract_imports(source: str) -> List[str]:
    """Extract import statements from source code."""
    imports = []

    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
    except Exception:
        pass

    return imports


def infer_dependencies_from_imports(imports: List[str]) -> List[str]:
    """Map import names to package requirements."""
    import_to_package = {
        "cv2": "opencv-python>=4.8.0",
        "numpy": "numpy>=1.24.0",
        "torch": "torch>=2.0.0",
        "ultralytics": "ultralytics>=8.0.0",
        "diffusers": "diffusers>=0.20.0",
        "transformers": "transformers>=4.30.0",
        "onnxruntime": "onnxruntime>=1.15.0",
        "PIL": "Pillow>=10.0.0",
        "chromadb": "chromadb>=0.5.0",
        "llama_cpp": "llama-cpp-python>=0.2.0",
        "pytesseract": "pytesseract>=0.3.10",
        "reportlab": "reportlab>=4.0.0",
    }

    dependencies = []
    for import_name in imports:
        # Check base module name
        base_name = import_name.split(".")[0]
        if base_name in import_to_package:
            dependencies.append(import_to_package[base_name])

    return sorted(list(set(dependencies)))


def extract_task_metadata_from_ast(
    file_path: Path, category_hint: str = "other"
) -> Optional[Dict[str, Any]]:
    """
    Extract complete task metadata using only AST parsing (no imports).

    Args:
        file_path: Path to the task Python file
        category_hint: Suggested category from file path

    Returns:
        Dictionary with task metadata or None if parsing fails
    """
    source = get_file_source(file_path)
    if source is None:
        return None

    tree = parse_file(file_path)
    if tree is None:
        return None

    result = find_task_decorator(tree)
    if result is None:
        return None

    func_node, decorator_node = result

    # Extract decorator arguments
    decorator_kwargs = extract_decorator_kwargs(decorator_node)

    # Get task ID (type_name or function name)
    task_id = decorator_kwargs.get("type_name", func_node.name)

    # Get category
    category = decorator_kwargs.get("category", category_hint)

    # Get display name
    display_name = decorator_kwargs.get(
        "display_name", func_node.name.replace("_", " ").title()
    )

    # Get description
    description = extract_docstring(func_node)
    if not description:
        description = f"Task: {display_name}"

    # Extract is_collapsed flag
    is_collapsed = decorator_kwargs.get("is_collapsed", False)

    # Extract outputs
    outputs = decorator_kwargs.get("outputs", [])
    if isinstance(outputs, str):
        outputs = [outputs]

    # Extract output types
    output_types = decorator_kwargs.get("output_types", {})

    # Resolve previewable: use explicit value if set, otherwise auto-infer from output_types
    _non_previewable_types = {"object", "generator", "array"}
    if "previewable" in decorator_kwargs:
        previewable = bool(decorator_kwargs["previewable"])
    elif output_types:
        previewable = not any(
            v.lower() in _non_previewable_types for v in output_types.values()
        )
    else:
        previewable = False

    # Build outputs dict
    outputs_dict = {}
    for output_name in outputs:
        outputs_dict[output_name] = {
            "type": output_types.get(output_name, "any"),
            "description": f"Output: {output_name}",
        }

    # Extract input parameters
    parameters = extract_function_parameters(func_node)

    # Build inputs dict
    inputs_dict = {}
    for param_name, param_info in parameters.items():
        input_meta = {
            "type": param_info.get("type", "any"),
            "required": param_info.get("required", True),
            "description": f"Input parameter: {param_name}",
        }

        if "default" in param_info:
            input_meta["default"] = param_info["default"]

        inputs_dict[param_name] = input_meta

    # Extract dependencies
    imports = extract_imports(source)
    dependencies = infer_dependencies_from_imports(imports)

    metadata = {
        "id": task_id,
        "source_file": file_path.name,
        "category": category,
        "display_name": display_name,
        "description": description,
        "is_collapsed": is_collapsed,
        "previewable": previewable,
        "inputs": inputs_dict,
        "outputs": outputs_dict,
        "dependencies": {"python": dependencies, "system": []},
    }

    return metadata


# ============================================================================
# SETUP AND UTILITIES
# ============================================================================


def merge_with_existing(
    new_metadata: Dict[str, Any], existing_path: Path
) -> Dict[str, Any]:
    """
    Merge new metadata with existing, preserving manual edits.

    Preserved fields: version, author, contributors, tags, created_at, custom descriptions
    """
    if not existing_path.exists():
        return new_metadata

    try:
        with open(existing_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

        # Compare content BEFORE overwriting fields from existing, so that changes
        # to any field (author, contributors, tags, description, etc.) are detected.
        IGNORE_FOR_CONTENT = {"created_at", "updated_at"}
        existing_content = {
            k: v for k, v in existing.items() if k not in IGNORE_FOR_CONTENT
        }
        new_content = {
            k: v for k, v in new_metadata.items() if k not in IGNORE_FOR_CONTENT
        }
        content_unchanged = existing_content == new_content

        # Preserve manually-edited fields
        preserve_fields = [
            "version",
            "author",
            "contributors",
            "tags",
            "created_at",
            "category",
            "internal",
            "auto_schedule",
        ]
        for field in preserve_fields:
            if field in existing:
                new_metadata[field] = existing[field]

        # Preserve description if customized
        if "description" in existing:
            auto_desc_patterns = [
                f"Workflow: {new_metadata.get('display_name', '')}",
                f"Task: {new_metadata.get('display_name', '')}",
            ]
            if existing["description"] not in auto_desc_patterns:
                # Check if it looks manually edited (longer/more detailed)
                if len(existing["description"]) > len(new_metadata.get("id", "")) + 20:
                    new_metadata["description"] = existing["description"]

        # Preserve updated_at if content hasn't changed to avoid spurious diffs
        if content_unchanged:
            new_metadata["updated_at"] = existing.get(
                "updated_at", new_metadata["updated_at"]
            )

        return new_metadata

    except Exception as e:
        print(f"  ⚠️  Warning: Could not load existing metadata: {e}")
        return new_metadata


# ============================================================================
# TASK METADATA GENERATION
# ============================================================================


def find_task_files(tasks_dir: Path) -> List[Path]:
    """Find all Python task files (excluding __init__.py and __pycache__)"""
    task_files = []
    for root, dirs, files in os.walk(tasks_dir):
        dirs[:] = [d for d in dirs if d not in ["__pycache__", "_metadata"]]

        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                task_files.append(Path(root) / file)

    return sorted(task_files)


def infer_category_from_path(file_path: Path, tasks_dir: Path) -> str:
    """Infer category from file path (e.g., tasks/io/load_image.py -> io)"""
    try:
        rel_path = file_path.relative_to(tasks_dir)
        if len(rel_path.parts) >= 2:
            return rel_path.parts[0]
    except ValueError:
        pass
    return "other"


def extract_task_metadata(
    file_path: Path, project_root: Path, tasks_dir: Path
) -> Optional[Dict[str, Any]]:
    """Extract metadata from a task file using AST parsing (no imports required)"""
    try:
        # Infer category from file path
        category_hint = infer_category_from_path(file_path, tasks_dir)

        # Use AST parsing to extract metadata
        metadata = extract_task_metadata_from_ast(file_path, category_hint)

        if metadata is None:
            return None

        # Add versioning and authorship info
        metadata["version"] = "1.0.0"
        metadata["author"] = "FabricFlow Team"
        metadata["contributors"] = []
        metadata["created_at"] = datetime.now().strftime("%Y-%m-%d")
        metadata["updated_at"] = datetime.now().strftime("%Y-%m-%d")
        metadata["tags"] = (
            [metadata["category"]] if metadata["category"] != "other" else []
        )

        return metadata

    except Exception as e:
        print(f"  ⚠️  Warning: Could not extract metadata from {file_path.name}: {e}")
        return None


def validate_task_output_types(metadata: Dict[str, Any], file_path: Path) -> List[str]:
    """
    Validate output_types consistency for a task.

    Returns a list of warning strings (empty means valid).
    """
    warnings = []
    outputs = list(metadata.get("outputs", {}).keys())
    output_types = {
        k: v.get("type", "any") for k, v in metadata.get("outputs", {}).items()
    }

    # Warn if the task has outputs but none have a specific type declared
    if outputs and all(t == "any" for t in output_types.values()):
        warnings.append(
            f"  ⚠  {file_path.name}: has {len(outputs)} output(s) but no specific output_types declared "
            f"(all 'any') — add output_types={{...}} to @task decorator for preview support"
        )

    return warnings


def generate_task_metadata(
    task_file: Path, project_root: Path, tasks_dir: Path, output_dir: Path
) -> bool:
    """Generate metadata file for a single task"""
    metadata = extract_task_metadata(task_file, project_root, tasks_dir)
    if metadata is None:
        return False

    task_id = metadata["id"]
    category = metadata["category"]

    # Create category subdirectory
    category_dir = output_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    output_file = category_dir / f"{task_id}.json"
    metadata = merge_with_existing(metadata, output_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    rel_task = task_file.relative_to(project_root)
    rel_output = output_file.relative_to(project_root)
    print(f"  OK: {rel_task} -> {rel_output}")

    for warning in validate_task_output_types(metadata, task_file):
        print(warning)

    return True


# ============================================================================
# WORKFLOW METADATA GENERATION
# ============================================================================


def find_workflow_files(workflows_dir: Path) -> List[Path]:
    """Find all workflow JSON files (excluding _metadata directory)"""
    workflow_files = []
    for file in workflows_dir.glob("*.json"):
        if file.is_file():
            workflow_files.append(file)
    return sorted(workflow_files)


def infer_category_from_tasks(required_tasks: List[str]) -> str:
    """Infer workflow category based on the tasks it uses"""
    category_keywords = {
        "vision": ["yolo", "detect", "segment", "vision", "object_detection"],
        "llm": ["llm", "llama", "chat", "gpt", "generate", "inference"],
        "diffusion": ["diffusion", "stable_diffusion", "generate_image"],
        "document": ["pdf", "ocr", "document", "tesseract"],
        "video": ["video", "camera", "webcam", "streaming"],
        "audio": ["audio", "whisper", "transcribe"],
        "embedding": ["embed", "similarity", "vector"],
    }

    task_str = " ".join(required_tasks).lower()

    for category, keywords in category_keywords.items():
        if any(keyword in task_str for keyword in keywords):
            return category

    return "general"


def infer_workflow_dependencies(required_tasks: List[str]) -> Dict[str, List[str]]:
    """Infer Python and model dependencies based on required tasks"""
    python_deps = set()
    model_deps = []

    task_str = " ".join(required_tasks).lower()

    if any(kw in task_str for kw in ["yolo", "object_detection", "detect_objects"]):
        python_deps.update(
            ["ultralytics>=8.0.0", "opencv-python>=4.8.0", "numpy>=1.24.0"]
        )
        model_deps.append("yolo11n.pt")

    if any(
        kw in task_str for kw in ["diffusion", "stable_diffusion", "generate_image"]
    ):
        python_deps.update(
            ["diffusers>=0.20.0", "torch>=2.0.0", "transformers>=4.30.0"]
        )

    if any(kw in task_str for kw in ["llm", "llama", "chat", "inference"]):
        python_deps.update(["torch>=2.0.0", "transformers>=4.30.0"])

    if "onnx" in task_str:
        python_deps.add("onnxruntime>=1.15.0")

    if any(kw in task_str for kw in ["image", "load_image", "resize", "convert"]):
        python_deps.update(["opencv-python>=4.8.0", "numpy>=1.24.0"])

    if any(kw in task_str for kw in ["ocr", "pdf", "tesseract", "document"]):
        python_deps.update(["pytesseract>=0.3.10", "opencv-python>=4.8.0"])
        if "pdf" in task_str:
            python_deps.add("reportlab>=4.0.0")

    if any(kw in task_str for kw in ["embed", "similarity", "vector", "chromadb"]):
        python_deps.update(["chromadb>=0.5.0", "sentence-transformers>=2.2.0"])

    return {"python": sorted(list(python_deps)), "models": model_deps}


def infer_workflow_tags(
    workflow_id: str, required_tasks: List[str], category: str
) -> List[str]:
    """Generate relevant tags based on workflow content"""
    tags = [category]
    combined = f"{workflow_id.lower()} {' '.join(required_tasks).lower()}"

    if "yolo" in combined:
        tags.extend(["yolo", "object-detection"])
    if "diffusion" in combined or "stable_diffusion" in combined:
        tags.extend(["diffusion", "image-generation"])
    if "llm" in combined or "llama" in combined:
        tags.extend(["llm", "text-generation"])
    if "ocr" in combined:
        tags.append("ocr")
    if "video" in combined or "camera" in combined or "webcam" in combined:
        tags.append("video-processing")
    if "streaming" in combined:
        tags.append("streaming")
    if "detection" in combined:
        tags.append("detection")
    if "segmentation" in combined:
        tags.append("segmentation")
    if "document" in combined:
        tags.append("document-processing")

    tags.extend(["ml", "computer-vision"])
    return sorted(list(set(tags)))


def extract_workflow_metadata(
    file_path: Path, project_root: Path
) -> Optional[Dict[str, Any]]:
    """Extract metadata from a workflow JSON file"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        workflow_id = file_path.stem
        display_name = workflow_id.replace("_", " ").title()

        # Extract required tasks
        required_tasks = []
        if "tasks" in workflow:
            for task in workflow["tasks"]:
                if "task" in task:
                    task_type = task["task"]
                    if task_type not in required_tasks:
                        required_tasks.append(task_type)

        category = infer_category_from_tasks(required_tasks)
        dependencies = infer_workflow_dependencies(required_tasks)
        tags = infer_workflow_tags(workflow_id, required_tasks, category)

        metadata = {
            "id": workflow_id,
            "version": "1.0.0",
            "display_name": display_name,
            "description": workflow.get("description", f"Workflow: {display_name}"),
            "author": "FabricFlow Team",
            "contributors": [],
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "updated_at": datetime.now().strftime("%Y-%m-%d"),
            "tags": tags,
            "category": category,
            "required_tasks": sorted(required_tasks),
            "dependencies": dependencies,
        }

        # Extract inputs
        inputs = {}
        if "inputs" in workflow:
            for input_name, default_value in workflow["inputs"].items():
                input_type = "string"
                if isinstance(default_value, bool):
                    input_type = "bool"
                elif isinstance(default_value, int):
                    input_type = "int"
                elif isinstance(default_value, float):
                    input_type = "float"
                elif isinstance(default_value, dict):
                    input_type = "object"
                elif isinstance(default_value, list):
                    input_type = "array"

                inputs[input_name] = {
                    "type": input_type,
                    "required": False,
                    "default": default_value,
                    "description": f"Workflow input: {input_name.replace('_', ' ')}",
                }

        metadata["inputs"] = inputs

        # Extract outputs
        outputs = {}
        if "outputs" in workflow:
            for output_name, output_path in workflow["outputs"].items():
                outputs[output_name] = {
                    "type": "any",
                    "description": f"Workflow output: {output_name.replace('_', ' ')}",
                }

        metadata["outputs"] = outputs
        return metadata

    except Exception as e:
        print(
            f"  ⚠️  Warning: Could not extract workflow metadata from {file_path.name}: {e}"
        )
        return None


def generate_workflow_metadata(
    workflow_file: Path, project_root: Path, output_dir: Path
) -> bool:
    """Generate metadata file for a single workflow"""
    metadata = extract_workflow_metadata(workflow_file, project_root)
    if metadata is None:
        return False

    workflow_id = metadata["id"]
    output_file = output_dir / f"{workflow_id}.json"
    metadata = merge_with_existing(metadata, output_file)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")

    rel_workflow = workflow_file.relative_to(project_root)
    rel_output = output_file.relative_to(project_root)
    print(f"  OK: {rel_workflow} -> {rel_output}")
    return True


# ============================================================================
# MAIN FUNCTION
# ============================================================================


def main():
    """Main entry point for metadata generation"""
    parser = argparse.ArgumentParser(
        description="Generate metadata JSON files for FabricFlow tasks and workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all metadata (tasks + workflows)
  python generate_manifest.py
  
  # Generate only task metadata
  python generate_manifest.py --tasks-only
  
  # Generate only workflow metadata
  python generate_manifest.py --workflows-only
        """,
    )

    parser.add_argument(
        "--tasks-only",
        action="store_true",
        help="Generate only task metadata",
    )

    parser.add_argument(
        "--workflows-only",
        action="store_true",
        help="Generate only workflow metadata",
    )

    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent
    tasks_dir = project_root / "tasks"
    workflows_dir = project_root / "workflows"

    generate_tasks = not args.workflows_only
    generate_workflows = not args.tasks_only

    total_generated = 0

    # Generate task metadata
    if generate_tasks and tasks_dir.exists():
        print("Generating task metadata...")
        print()

        output_dir = tasks_dir / "_metadata"
        output_dir.mkdir(parents=True, exist_ok=True)

        task_files = find_task_files(tasks_dir)
        print(f"Found {len(task_files)} task file(s)")
        print()

        task_count = 0
        for task_file in task_files:
            if generate_task_metadata(task_file, project_root, tasks_dir, output_dir):
                task_count += 1

        total_generated += task_count
        print()
        print(f"Generated {task_count} task metadata file(s)")
        print()

    # Generate workflow metadata
    if generate_workflows and workflows_dir.exists():
        print("Generating workflow metadata...")
        print()

        output_dir = workflows_dir / "_metadata"
        output_dir.mkdir(parents=True, exist_ok=True)

        workflow_files = find_workflow_files(workflows_dir)
        print(f"Found {len(workflow_files)} workflow file(s)")
        print()

        workflow_count = 0
        for workflow_file in workflow_files:
            if generate_workflow_metadata(workflow_file, project_root, output_dir):
                workflow_count += 1

        total_generated += workflow_count
        print()
        print(f"Generated {workflow_count} workflow metadata file(s)")
        print()

    print("=" * 60)
    print(f"Total metadata files generated: {total_generated}")
    print(f"Output: tasks/_metadata/ and workflows/_metadata/")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
