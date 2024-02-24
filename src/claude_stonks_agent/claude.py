from dataclasses import dataclass
import boto3
from langchain_community.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from typing import Any, List, Sequence, Union
from xml.sax.saxutils import escape as xml_escape
from contextlib import contextmanager
from langchain_core.outputs import LLMResult
from langchain_core.prompt_values import PromptValue
from langchain_core.tools import BaseTool
from langchain.agents.agent import AgentOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from xml.etree import ElementTree as ET
from langchain_core.callbacks import BaseCallbackHandler, BaseCallbackManager
import re


def fix_prompt(prompt: str) -> str:
    """
    The prompt that the bedrock integration sends prefixes the AI response with 'AI: ' where as claude expects 'Assistant: '
    """
    ai_to_assistant = re.sub(r'^AI: ', 'Assistant: ', prompt, flags=re.MULTILINE)
    return ai_to_assistant


class ClaudeBedrock(Bedrock):
    """
    Derived version to fix some functionality for the claude model.
    """
    def __init__(self):
        super().__init__(
            client=boto3.client('bedrock-runtime'),
            model_id='anthropic.claude-v2:1',
            model_kwargs={
                'temperature': 0.1,
                'stop_sequences': ['\n\nHuman:', '</function_calls>']
            }
        )

    def generate_prompt(
            self, 
            prompts: List[PromptValue], 
            stop: List[str] | None = None, 
            callbacks: List[BaseCallbackHandler] | BaseCallbackManager | List[List[BaseCallbackHandler] | BaseCallbackManager | None] | None = None, 
            **kwargs: Any
        ) -> LLMResult:
        prompt_strings = [ p.to_string() for p in prompts ]
        fixed_prompt_strings = [ fix_prompt(p) for p in prompt_strings]
        result = super().generate(fixed_prompt_strings, stop=stop, callbacks=callbacks, **kwargs)
        return result


def create_llm() -> Bedrock:
    return ClaudeBedrock()


class XmlBuilder:
    def __init__(self):
        self._builder = StringBuilder()

    @contextmanager
    def tag_with_children(self, name: str):
        self._builder.append_line("<" + name + ">")
        try:
            yield
        finally:
            self._builder.append_line("</" + name + ">")

    def tag_with_text(self, name: str, text: str, escape: bool = True):
        open_tag = f'<{name}>'
        close_tag = f'</{name}>'
        content = xml_escape(text) if escape else text

        # if the value has multiple lines then put opening and closing on their own lines
        if '\n' in content:
            self._builder.append_line(open_tag)
            self._builder.append_line(content)
            self._builder.append_line(close_tag)
        else:
            self._builder.append_line(open_tag + content + close_tag)

    def __str__(self):
        return str(self._builder)


class StringBuilder:
    def __init__(self):
        self._buffer = []

    def append(self, value: str):
        self._buffer.append(value)

    def append_line(self, value: str):
        self._buffer.append(value)
        self._buffer.append('\n')

    def __str__(self):
        return ''.join(self._buffer)


def build_tools_description(tools: Sequence[BaseTool]) -> str:
    builder = XmlBuilder()

    with builder.tag_with_children('tools'):
        for tool in tools:
            build_tool_description(tool, builder)

    return str(builder)


def _clean_description(description: str) -> str:
    # strip the 'whatever(arg1, arg2) - ' prefix from descriptions
    args_list_suffix = ') - '
    args_list_end = description.find(args_list_suffix)

    if args_list_end > 0:
        return description[args_list_end + len(args_list_suffix):]
    else:
        return description


def build_tool_description(tool: BaseTool, builder: XmlBuilder = XmlBuilder()) -> str:
    with builder.tag_with_children("tool_description"):
        builder.tag_with_text('tool_name', tool.name)
        builder.tag_with_text('description', tool.description)

        if tool.args:
            with builder.tag_with_children('parameters'):
                for name, arg in tool.args.items():
                    with builder.tag_with_children('parameter'):
                        builder.tag_with_text('name', name)
                        builder.tag_with_text('type', arg['type'])

                        description = arg.get('description')
                        if description:
                            builder.tag_with_text('description', _clean_description(description))


    return str(builder)


def xml_to_dict(element):
    """Recursively convert an XML element into a dictionary."""
    # Base case: if the element has no children, return its text
    if not list(element):
        return element.text

    child_count = len(element)
    key_set = { child.tag for child in element }

    if child_count > 1 and len(key_set) == 1:
        # there's only 1 child tag name but mutliple elements with it, so treat the whole thing as an array on the parent
        return [ xml_to_dict(child) for child in element ]

    # Recursive case: build a dictionary from the element's children
    return {child.tag: xml_to_dict(child) for child in element}


def parse_xml(text: str):
    try:
        return ET.fromstring(text)
    except Exception as ex:
        print(f'ERROR: Failed to parse xml')
        print(ex)
        print(text)
        raise


def parse_xml_to_dict(text: str) -> dict:
    root = parse_xml(text)
    result = { root.tag: xml_to_dict(root) }
    return result


_open_function_calls = '<function_calls>'
_close_function_calls = '</function_calls>'


@dataclass
class AgentActions:
    actions: list[AgentAction]
    log: str


def extract_agent_actions(text: str) -> AgentActions:
    open_calls = text.find(_open_function_calls)

    # no function calls
    if open_calls < 0:
        return AgentActions(actions=[], log=text)

    close_calls = text.find(_close_function_calls)

    # autoclose function calls in case it was cut off by the stop sequences
    if open_calls >=0 and close_calls < 0:
        text += _close_function_calls

    close_calls = text.find(_close_function_calls)

    xml_text = text[open_calls:close_calls + len(_close_function_calls)]
    xml = parse_xml_to_dict(xml_text)

    calls = xml['function_calls']
    invoke = calls['invoke']
    tool_name = invoke['tool_name']

    parameters = invoke.get('parameters', {})
    if isinstance(parameters, str):
        # make sure it's just an empty string (something for calling a no arg function we'll received '\n')
        stripped_str = parameters.strip()
        if stripped_str:
            raise ValueError(f'Parameters are a string with a value: {stripped_str}')
        parameters = {}

    tool_input = { name: value for name, value in parameters.items() }

    actions = [ AgentAction(tool=tool_name, tool_input=tool_input, log='') ]

    # include any preamble in logs
    return AgentActions(actions=actions, log=text[0: close_calls + len(_close_function_calls)])



def format_tool_responses(
    intermediate_steps: list[tuple[AgentAction, str]],
) -> str:
    builder = XmlBuilder()

    with builder.tag_with_children('function_results'):
        for action, outcome in intermediate_steps:
            with builder.tag_with_children('result'):
                builder.tag_with_text('tool_name', action.tool)
                builder.tag_with_text('stdout', str(outcome), escape=False)

    return str(builder)
