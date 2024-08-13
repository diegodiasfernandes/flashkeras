import os

def count_directories_in_directory(path: str) -> int:
    if not os.path.isdir(path):
        raise ValueError(f"The path '{path}' is not a valid directory.")
    
    return sum(1 for item in os.listdir(path) if os.path.isdir(os.path.join(path, item)))