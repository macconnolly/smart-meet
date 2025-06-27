# Code Style Conventions

## General Standards
- **Python Version**: 3.11+ with type hints required
- **Line Length**: 100 characters maximum
- **Formatter**: Black (configured in implementation)
- **Linter**: Flake8
- **Type Checker**: MyPy
- **Import Order**: Standard library → Third party → Local (alphabetical within groups)

## Naming Conventions
- **Classes**: PascalCase (e.g., `MemoryExtractor`, `ActivationEngine`)
- **Functions/Methods**: snake_case (e.g., `spread_activation`, `extract_memories`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `MAX_ACTIVATIONS`)
- **Private Methods**: Leading underscore (e.g., `_calculate_decay`)
- **Async Functions**: Prefix with action verb (e.g., `get_`, `create_`, `update_`)

## Code Organization
- **File Structure**: One class per file for major components
- **Module Structure**: Logical grouping (extraction/, cognitive/, storage/)
- **Dataclasses**: Use for data models with `@dataclass` decorator
- **Type Hints**: Required for all function parameters and returns
- **Docstrings**: Required for classes and public methods

## Async Patterns
- Use `async/await` for all I/O operations
- Batch operations where possible
- Use `asyncio.create_task()` for background operations
- Handle `CancelledError` in long-running tasks

## Error Handling
- Use specific exceptions over generic ones
- Log errors with context
- Return `Optional[T]` for nullable results
- Validate inputs early

## Testing Standards
- Test files: `test_<module>.py`
- Use pytest fixtures for setup
- Mock external dependencies
- Test async functions with `pytest-asyncio`
- Aim for >80% coverage

## Documentation
- Docstring format:
```python
def function_name(param: Type) -> ReturnType:
    """Brief description.
    
    Args:
        param: Description of parameter
        
    Returns:
        Description of return value
    """
```

## Performance Considerations
- Use generators for large datasets
- Cache expensive computations
- Batch database operations
- Profile before optimizing

## Specific Patterns
- **Repository Pattern**: For database access
- **Engine Pattern**: For complex algorithms
- **Factory Pattern**: For object creation
- **Async Context Managers**: For resource management

## Example Code Style
```python
from dataclasses import dataclass
from typing import List, Optional
import asyncio

@dataclass
class Memory:
    """Represents a cognitive memory."""
    id: str
    content: str
    importance_score: float = 0.5

class MemoryRepository:
    """Handles memory persistence."""
    
    async def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve memory by ID.
        
        Args:
            memory_id: Unique memory identifier
            
        Returns:
            Memory object or None if not found
        """
        # Implementation
        pass
```