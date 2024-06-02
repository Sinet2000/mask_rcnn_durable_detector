from enum import Enum
from typing import Dict, Type, TypeVar

T = TypeVar('T', bound=Enum)
    
def from_json_with_enum(data: Dict[str, T], enum_type: Type[T]) -> T:

    try:
        return enum_type(data)
    except ValueError:
        # Handle cases where the enum value is not found
        return enum_type.UNKNOWN  # Assuming UNKNOWN is defined in the enum, or you can handle it differently

def string_to_enum(enum_class: Type[T], string_value: str) -> T:
    try:
        string_value_lower = string_value.lower()
        for member in enum_class:
            if member.name.lower() == string_value_lower:
                return member
        raise ValueError  # If no match is found, raise ValueError
    except ValueError:
        return enum_class.UNKNOWN 