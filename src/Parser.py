from pydantic import BaseModel,Field
import argparse


class Prompt(BaseModel):
    prompt: str = Field(min_length = 1)


class ValueType(BaseModel):
    type: str = Field(min_lenght = 1)


class FunctionCall(BaseModel):
    name: str= Field(min_lenght = 1)
    description: str= Field(min_lenght = 1)
    parameters: dict[str, ValueType]
    returns: ValueType


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description='parsing of arguments ...')

    def action(self):
        self.parser.add_argument(
            "--functions_definition", type=str, help="functions definition file")
        self.parser.add_argument(
            '--input', type=str, help='input file prompts'
        )
        self.parser.add_argument(
            '--output', type=str, help='output file'
        )
        self.parser.add_argument(
            '--model', type=str, help='model name'
        )
        return self.parser.parse_args()
