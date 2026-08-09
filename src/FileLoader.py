import json
from typing import Any, Dict, List, Optional
from Parser import Prompt, FunctionCall
from pydantic import ValidationError


class FileLoader:
    """Utility class to handle loading and validating input JSON files."""

    def load_file_data(
        self, file_name: str, type_file: str
    ) -> Optional[List[Dict[str, Any]]]:
        """Load JSON file content and validate structure using Pydantic."""
        # read json prompts file
        try:
            with open(file_name, 'r') as file:
                data = json.load(file)
        except FileNotFoundError as e:
            print(e)
            return None
        except json.JSONDecodeError as e:
            print(e)
            return None

        # here i wanna parse the data input (using pydantic)
        try:
            if not data:
                raise ValueError('no data exist')
            if type_file == 'input':
                for line in data:
                    Prompt(**line)

            elif type_file == 'functions_definition':
                for line in data:
                    FunctionCall(**line)
        except (ValidationError, ValueError) as e:
            print(e)
            return None
        return data
