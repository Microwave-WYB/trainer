from typing import Optional


def unwrap[T](value: Optional[T]) -> T:
    if value is None:
        raise ValueError("Value is None")

    return value
