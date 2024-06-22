import inspect
from enum import Enum
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import docstring_parser


class ChatgptParameter:
    def __init__(
        self,
        name: str,
        type: str,
        is_required: bool,
        description: Optional[str] = None,
        list_enum_value: Optional[List[str]] = None,
    ) -> None:
        self.name = name
        self.type = type
        self.is_required = is_required

        self.description = description
        self.list_enum_value = list_enum_value


class ChatgptTool:

    def __init__(
        self,
        callable: Callable,
        name: str,
        description: str,
        list_parameter: List[ChatgptParameter],
    ) -> None:
        self.callable = callable
        self.name = name
        self.description = description
        self.list_parameter = list_parameter
        self.list_secret_key = []

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.call(*args, **kwds)

    def call(self, *args, **kwds) -> Any:
        return self.callable(*args, **kwds)

    def to_tool_dict(self) -> dict:
        tool_dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }
        for parameter in self.list_parameter:
            tool_dict["function"]["parameters"]["properties"][parameter.name] = {
                "type": parameter.type,
            }
            if parameter.description:
                tool_dict["function"]["parameters"]["properties"][parameter.name]["description"] = parameter.description
            if parameter.list_enum_value:
                tool_dict["function"]["parameters"]["properties"][parameter.name]["enum"] = parameter.list_enum_value
            if parameter.is_required:
                tool_dict["function"]["parameters"]["required"].append(parameter.name)
        return tool_dict


def parse_docstring(callable: Callable) -> Tuple[str, Dict[str, str]]:
    docstring_parsed = docstring_parser.parse(inspect.getdoc(callable))  # type: ignore
    if docstring_parsed.short_description is None:
        raise ValueError("The short description in the docstring is required.")
    description = docstring_parsed.short_description

    # TODO check if it matches the signature
    dict_parameter_description = {}
    for (
        param
    ) in docstring_parsed.params:  # TODO there is also something strange that happens if 1 description is missing
        dict_parameter_description[param.arg_name] = param.description
    return description, dict_parameter_description


def get_parameter_type(annotation: type) -> Tuple[str, Optional[List[str]]]:
    if annotation == str:
        return "string", None
    if annotation == int:
        return "number", None
    if annotation == float:
        return "number", None
    if annotation == bool:
        return "boolean", None
    # check in annotation is a literal
    if hasattr(annotation, "__origin__") and annotation.__origin__ == Literal:
        return "string", [str(value) for value in annotation.__args__]
    if issubclass(annotation, Enum):
        enum_members = [member.value for member in annotation]
        return "string", enum_members
    raise ValueError(f"Unsupported parameter type: {annotation}")


def create_chatgpt_tool(callable: Callable) -> "ChatgptTool":
    name = callable.__name__
    signature = inspect.signature(callable)
    if signature.return_annotation == inspect.Signature.empty:
        raise ValueError("The return annotation is required.")
    if signature.return_annotation != str:
        raise ValueError("The return annotation must be a string.")

    description, dict_parameter_description = parse_docstring(callable)
    list_parameter: List[ChatgptParameter] = []
    for parameter in signature.parameters.values():
        if parameter.annotation == inspect.Signature.empty:
            raise ValueError("All parameters must have an annotation.")
        parameter_name = parameter.name
        parameter_type, parameter_list_enum_value = get_parameter_type(parameter.annotation)
        parameter_is_required = parameter.default == inspect.Parameter.empty
        parameter_description = dict_parameter_description.get(parameter_name, None)

        list_parameter.append(
            ChatgptParameter(
                name=parameter_name,
                type=parameter_type,
                is_required=parameter_is_required,
                description=parameter_description,
                list_enum_value=parameter_list_enum_value,
            )
        )
    return ChatgptTool(callable, name, description, list_parameter)
