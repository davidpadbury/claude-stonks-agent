from abc import ABC, abstractmethod
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage
from langchain_core.agents import AgentAction
from claude_stonks_agent.claude import AgentActions, format_tool_responses
from typing import Optional

def merge_messages(messages1: list[BaseMessage], messages2: list[BaseMessage]) -> list[BaseMessage]:
    result = list(messages1)

    for message in messages2:
        if result and isinstance(result[-1], AIMessage) and isinstance(message, AIMessage):
            result[-1] = AIMessage(content=result[-1].content + '\n\n' + message.content)
        else:
            result.append(message)

    return result


class AgentStep(ABC):
    """
    Separate each step of the agent's work into a distinct item that will make it easier to display the conversation
    """
    @staticmethod
    def steps_to_messages(steps: list['AgentStep']) -> list[BaseMessage]:
        result = []

        for step in steps:
            result = merge_messages(result, step.to_messages())

        return result

    st_role: str = 'agent'
    st_avatar: Optional[str] = None

    @abstractmethod
    def to_messages(self) -> list[BaseMessage]:
        pass

    def format_st_message(self) -> Optional[str]:
        return None

    def format_st_status_title(self) -> Optional[str]:
        return None
    
    def format_st_status_content(self) -> Optional[str]:
        return None

    def __str__(self) -> str:
        return type(self).__name__

    def __repr__(self) -> str:
        return type(self).__name__


class HumanInputStep(AgentStep):
    def __init__(self, input: str):
        self.input = input

    st_role: str = 'human'
    st_avatar: Optional[str] = '👩🏻‍🦰'

    def to_messages(self) -> list[BaseMessage]:
        return [HumanMessage(content=self.input)]

    def format_st_message(self) -> Optional[str]:
        return self.input

    def __repr__(self) -> str:
        return 'HumanInputStep("{}")'.format(self.input)


def format_tool_args(action: AgentAction) -> str:
    tool_input = action.tool_input

    match tool_input:
        case str():
            return tool_input
        case dict():
            return ', '.join([
                f'{key}={value}'
                for key, value in tool_input.items()
            ])


class ToolCallStep(AgentStep):
    """
    Request from the assistant to call a tool
    """
    def __init__(self, agent_actions: AgentActions):
        self.agent_actions = agent_actions

    display_in_history: bool = False
    st_role: str = 'assistant'
    st_avatar: Optional[str] = '🛠️'

    def to_messages(self) -> list[BaseMessage]:
        return [AIMessage(content=self.agent_actions.log)]

    def __repr__(self) -> str:
        tool_calls_text = ';'.join([
            f'{action.tool}({format_tool_args(action)})'
            for action in self.agent_actions.actions
        ])
        return 'ToolCallStep("{}")'.format(tool_calls_text)

    def format_st_status_title(self) -> str | None:
        first_action = self.agent_actions.actions[0]
        return first_action.tool

    def format_st_status_content(self) -> str | None:
        first_action = self.agent_actions.actions[0]
        return f'{first_action.tool}({format_tool_args(first_action)})'


class ToolCallResultStep(AgentStep):
    """
    Results of a call to a tool
    """
    def __init__(self, results: list[tuple[AgentAction, str]]):
        self.results = results

    display_in_history: bool = False
    st_role: str = 'assistant'
    st_avatar: Optional[str] = '️👾'

    def to_messages(self) -> list[BaseMessage]:
        formatted = format_tool_responses(self.results)
        return [AIMessage(content=formatted)]


    def format_st_status_title(self) -> str | None:
        (first_action, result) = self.results[0]
        return f'received {first_action.tool}({format_tool_args(first_action)})'

    def format_st_status_content(self) -> str | None:
        (first_action, result) = self.results[0]
        return f'{first_action.tool}({format_tool_args(first_action)})\n{result}'

    def __repr__(self) -> str:
        tool_outputs = ';'.join([
            f'{action.tool}({format_tool_args(action)}) -> {output}'
            for action, output in self.results
        ])
        return 'ToolCallResultStep("{}")'.format(tool_outputs)


class AgentOutcomeStep(AgentStep):
    """
    Final response from the agent
    """
    def __init__(self, output: str):
        self.output = output

    st_role: str = 'assistant'
    st_avatar: Optional[str] = '🤖'

    def to_messages(self) -> list[BaseMessage]:
        return [ AIMessage(content=self.output) ]

    def format_st_message(self) -> Optional[str]:
        return self.output

    def __repr__(self) -> str:
        return f'AgentOutcomeStep({self.output.strip()})'