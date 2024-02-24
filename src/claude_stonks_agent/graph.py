from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from typing import Annotated, TypedDict
from claude_stonks_agent.chains.main import create_chain as create_main_chain
import operator
from claude_stonks_agent.tools import create_alpha_vantage_tools
from claude_stonks_agent.alpha_vantage import AlphaVantageService
from claude_stonks_agent.claude import extract_agent_actions
from claude_stonks_agent.steps import AgentStep, HumanInputStep, ToolCallStep, ToolCallResultStep, AgentOutcomeStep


class AgentState(TypedDict):
    input: str
    steps: Annotated[list[AgentStep], operator.add]


def has_pending_functions(state: AgentState) -> str:
    last_step = state['steps'][-1]
    return 'run_tools' if isinstance(last_step, ToolCallStep) else END


def create_graph():
    alpha_vantage_tools = create_alpha_vantage_tools(AlphaVantageService.create())
    tool_executor = ToolExecutor(alpha_vantage_tools)
    main_chain = create_main_chain(alpha_vantage_tools)

    def run_entry(state: AgentState):
        # Store the original query as the first human input step
        return {
            'steps': [
                HumanInputStep(state['input']),
            ]
        }

    def run_main(state: AgentState):
        steps = state['steps']
        messages = AgentStep.steps_to_messages(steps)
        result = main_chain.invoke({
            'messages': messages,
        })
        pending_functions = extract_agent_actions(result)

        next_step = ToolCallStep(pending_functions) if pending_functions.actions else AgentOutcomeStep(result)

        return {
            'steps': [ next_step ]
        }

    def run_tools(state: AgentState):
        last_step = state['steps'][-1]

        if not isinstance(last_step, ToolCallStep):
            raise Exception('Shouldnt be here')

        pending_functions = last_step.agent_actions

        results = [
            (action, tool_executor.invoke(action))
            for action in pending_functions.actions
        ]

        return {
            'steps': [ ToolCallResultStep(results) ]
        }

    workflow = StateGraph(AgentState)

    workflow.add_node('entry', run_entry)
    workflow.add_node('main', run_main)
    workflow.add_node('tools', run_tools)

    workflow.set_entry_point('entry')

    workflow.add_edge('entry', 'main')
    workflow.add_edge('tools', 'main')
    workflow.add_conditional_edges(
        'main',
        has_pending_functions,
        {
            'run_tools': 'tools',
            END: END
        }
    )


    return workflow.compile()
