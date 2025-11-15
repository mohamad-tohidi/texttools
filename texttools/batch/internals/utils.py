from typing import Any


def export_data(data) -> list[dict[str, str]]:
    """
    Produces a structure of the following form from an initial data structure:
    [{"id": str, "text": str},...]
    """
    return data


def import_data(data) -> Any:
    """
    Takes the output and adds and aggregates it to the original structure.
    """
    return data
