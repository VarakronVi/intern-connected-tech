# file_operations.py
import os

def read_file(file_path):
    """Read content from a text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_file(file_path, content):
    """Write content to a text file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)
