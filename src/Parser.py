import argparse
from typing import Dict
from pydantic import BaseModel, Field


class Prompt(BaseModel):
    """Schema representing a user prompt."""

    prompt: str = Field(min_length=1)


class ValueType(BaseModel):
    """Schema representing data types for function parameters and returns."""

    type: str = Field(min_length=1)


class FunctionCall(BaseModel):
    """Schema representing function definitions and metadata."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, ValueType]
    returns: ValueType


class Parser:
    """CLI argument parser for input options."""

    def __init__(self) -> None:
        """Initialize the ArgumentParser instance."""
        self.parser = argparse.ArgumentParser(
            description='parsing of arguments ...'
        )

    def action(self) -> argparse.Namespace:
        """Define arguments and parse command line input."""
        self.parser.add_argument(
            "--functions_definition",
            type=str,
            help="functions definition file"
        )
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
