import json
from Parser import Prompt, FunctionCall
from pydantic import ValidationError


class FileLoader:

    def load_file_data(self, file_name, type_file) -> list:
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

        # here i wanna parse the data input (using pydintic)
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
