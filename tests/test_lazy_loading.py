"""Unit tests for tasks package lazy loading behavior."""

from unittest.mock import Mock, call, patch
import pytest


@pytest.fixture(autouse=True)
def reset_task_state():
    """Reset tasks module state before each test."""
    import tasks
    
    original_loaded = tasks._loaded_modules.copy()
    original_failed = tasks._failed_modules.copy()
    original_available = dict(tasks._available_modules)
    
    yield
    
    tasks._loaded_modules.clear()
    tasks._loaded_modules.update(original_loaded)
    tasks._failed_modules.clear()
    tasks._failed_modules.update(original_failed)
    tasks._available_modules.clear()
    tasks._available_modules.update(original_available)


class TestLazyLoading:
    """Test lazy loading behavior of tasks package."""

    def test_importing_tasks_does_not_import_task_modules(self):
        """Test that importing tasks doesn't eagerly import task modules."""
        with patch("importlib.import_module") as mock_import:
            import importlib
            import tasks
            
            importlib.reload(tasks)
            
            task_module_imports = [
                call_args[0][0] for call_args in mock_import.call_args_list
                if call_args[0][0].startswith("tasks.") and call_args[0][0] != "tasks"
            ]
            
            assert len(task_module_imports) == 0, f"Eagerly imported: {task_module_imports}"

    def test_get_task_class_imports_only_needed_module(self):
        """Test that get_task_class only imports the specific module needed."""
        import tasks
        
        tasks._loaded_modules.clear()
        tasks._failed_modules.clear()
        tasks._available_modules.clear()
        tasks._available_modules["test_task"] = ["tasks.test.test_task"]
        
        mock_task = Mock(type_name="test_task")
        
        with patch("importlib.import_module") as mock_import:
            with patch.object(tasks, "_original_get_task_class") as mock_get_task:
                mock_get_task.side_effect = [ValueError("Unknown task"), mock_task]
                
                result = tasks._lazy_get_task_class("test_task")
                
                mock_import.assert_called_once_with("tasks.test.test_task")
                assert result == mock_task

    def test_get_task_class_does_not_reimport_loaded_modules(self):
        """Test that already-loaded modules are not re-imported."""
        import tasks
        
        tasks._loaded_modules.clear()
        tasks._failed_modules.clear()
        tasks._available_modules.clear()
        tasks._loaded_modules.add("tasks.test.test_task")
        tasks._available_modules["test_task"] = ["tasks.test.test_task"]
        
        mock_task = Mock(type_name="test_task")
        
        with patch("importlib.import_module") as mock_import:
            with patch.object(tasks, "_original_get_task_class") as mock_get_task:
                mock_get_task.return_value = mock_task
                
                result = tasks._lazy_get_task_class("test_task")
                
                mock_import.assert_not_called()
                assert result == mock_task

    def test_failed_modules_are_not_retried(self):
        """Test that modules that failed to import are not retried."""
        import tasks
        
        tasks._loaded_modules.clear()
        tasks._failed_modules.clear()
        tasks._available_modules.clear()
        tasks._failed_modules.add("tasks.test.broken_task")
        tasks._available_modules["broken_task"] = ["tasks.test.broken_task"]
        
        with patch("importlib.import_module") as mock_import:
            with patch.object(tasks, "_original_get_task_class") as mock_get_task:
                mock_get_task.side_effect = ValueError("Unknown task")
                
                with pytest.raises(ValueError):
                    tasks._lazy_get_task_class("broken_task")
                
                failed_calls = [c for c in mock_import.call_args_list 
                               if c == call("tasks.test.broken_task")]
                assert len(failed_calls) == 0

    def test_lazy_loading_with_duplicate_filenames(self):
        """Test lazy loading tries multiple paths for duplicate filenames."""
        import tasks
        
        tasks._loaded_modules.clear()
        tasks._failed_modules.clear()
        tasks._available_modules.clear()
        tasks._available_modules["utils"] = ["tasks.vision.utils", "tasks.llm.utils"]
        
        mock_task = Mock(type_name="format_string")
        
        with patch("importlib.import_module") as mock_import:
            with patch.object(tasks, "_original_get_task_class") as mock_get_task:
                mock_get_task.side_effect = [
                    ValueError("Unknown task"),
                    ValueError("Not in first module"),
                    mock_task
                ]
                
                result = tasks._lazy_get_task_class("utils")
                
                assert mock_import.call_count == 2
                mock_import.assert_any_call("tasks.vision.utils")
                mock_import.assert_any_call("tasks.llm.utils")
                assert result == mock_task

    def test_slow_task_logging(self):
        """Test that slow tasks (>100ms) are logged."""
        import tasks
        import time as time_module
        
        tasks._loaded_modules.clear()
        tasks._failed_modules.clear()
        tasks._available_modules.clear()
        tasks._available_modules["slow_task"] = ["tasks.test.slow_task"]
        
        mock_task = Mock(type_name="slow_task")
        
        with patch("importlib.import_module"):
            with patch.object(tasks, "_original_get_task_class") as mock_get_task:
                with patch.object(time_module, "perf_counter") as mock_time:
                    with patch.object(tasks.logger, "info") as mock_log:
                        mock_time.side_effect = [0.0, 0.2]
                        mock_get_task.side_effect = [ValueError("Unknown task"), mock_task]
                        
                        tasks._lazy_get_task_class("slow_task")
                        
                        assert mock_log.call_count == 1
                        args = mock_log.call_args
                        assert "Slow task load" in args[0][0]
                        assert "tasks.test.slow_task" in args[0][0]
                        assert args[1]["extra"]["duration"] == 0.2

    def test_fast_task_not_logged(self):
        """Test that fast tasks (<100ms) are not logged."""
        import tasks
        import time as time_module
        
        tasks._loaded_modules.clear()
        tasks._failed_modules.clear()
        tasks._available_modules.clear()
        tasks._available_modules["fast_task"] = ["tasks.test.fast_task"]
        
        mock_task = Mock(type_name="fast_task")
        
        with patch("importlib.import_module"):
            with patch.object(tasks, "_original_get_task_class") as mock_get_task:
                with patch.object(time_module, "perf_counter") as mock_time:
                    with patch.object(tasks.logger, "info") as mock_log:
                        mock_time.side_effect = [0.0, 0.05]
                        mock_get_task.side_effect = [ValueError("Unknown task"), mock_task]
                        
                        tasks._lazy_get_task_class("fast_task")
                        
                        mock_log.assert_not_called()

    def test_import_failure_logged_and_tracked(self):
        """Test that import failures are logged and tracked."""
        import tasks
        
        tasks._loaded_modules.clear()
        tasks._failed_modules.clear()
        tasks._available_modules.clear()
        tasks._available_modules["broken_task"] = ["tasks.test.broken_task"]
        
        with patch("importlib.import_module") as mock_import:
            with patch.object(tasks, "_original_get_task_class") as mock_get_task:
                with patch.object(tasks.logger, "warning") as mock_warn:
                    mock_import.side_effect = ModuleNotFoundError("No module named 'torch'")
                    mock_get_task.side_effect = ValueError("Unknown task")
                    
                    with pytest.raises(ValueError):
                        tasks._lazy_get_task_class("broken_task")
                    
                    assert mock_warn.call_count >= 1
                    first_call = mock_warn.call_args_list[0][0][0]
                    assert "Failed lazy load" in first_call
                    assert "broken_task" in first_call
                    assert "ModuleNotFoundError" in first_call
                    assert "No module named 'torch'" in first_call
                    assert "tasks.test.broken_task" in tasks._failed_modules
