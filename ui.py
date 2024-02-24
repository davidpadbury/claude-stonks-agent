from dotenv import load_dotenv
load_dotenv()

from typing import Optional
import streamlit as st
import random
from claude_stonks_agent.graph import create_graph
from claude_stonks_agent.steps import AgentStep, HumanInputStep, ToolCallStep, ToolCallResultStep, AgentOutcomeStep
from langgraph.graph import END

st.set_page_config(
    page_title='Claude Stonks',
    page_icon='📉'
)
st.title('📉 Claude Stonks')

thinking_messages = [
    'Yoloing into the abyss...',
    'Consulting the meme lords...',
    'Leveraging diamond hands...',
    'Calculating risk for tendies...',
    'Analyzing stonk trajectories...',
    'Betting the farm on moonshots...',
    'Dipping toes in volatile waters...',
    'Listening for the money printers brrr...'
]

def thinking_message() -> str:
    return random.choice(thinking_messages)

def extract_latest_step(state: dict) -> Optional[AgentStep]:
    # will be using the node name but should have steps as the key of that
    first_key = next(iter(state.keys()), None)

    if not first_key:
        return None
    
    steps = state[first_key].get('steps', [])

    return steps[-1] if steps else None


def text_to_markdown(text: str) -> str:
    return text \
        .replace('\n', '\n\n') \
        .replace('$', '\\$')

graph = create_graph()
graph_config = { 'recursion_limit': 100, 'max_concurrency': 20 }

if 'chat_steps' not in st.session_state:
    st.session_state.chat_steps = []

def read_chat_steps() -> list[ToolCallStep]:
    return st.session_state.chat_steps

for step in read_chat_steps():
    message_text = step.format_st_message()

    if not message_text:
        continue

    with st.chat_message(step.st_role, avatar=step.st_avatar):
        st.markdown(text_to_markdown(message_text))


if prompt := st.chat_input('> '):
    with st.chat_message('user', avatar=HumanInputStep.st_avatar):
        st.markdown(prompt)

    request = {
        'input': prompt,
        'steps': read_chat_steps()
    }

    last_response: Optional[dict] = None
    with st.chat_message('assistant', avatar='🤖'):
        with st.status(thinking_message()) as status:
            for state in graph.stream(request, config=graph_config):
                last_response = state

                if END not in state:
                    latest_step = extract_latest_step(state)
                    if latest_step:
                        latest_status = latest_step.format_st_status_title()
                        latest_status_content = latest_step.format_st_status_content()
                        status.update(label = latest_status or thinking_message())
                        if latest_status_content:
                            st.write(latest_status_content)
                    else:
                        status.update(label = thinking_message())

                last_response = state

            status.update(label = '💎 Finished 💎')
            

        if last_response and END in last_response:
            steps = last_response[END]['steps']
            last_step = steps[-1]

            if isinstance(last_step, AgentOutcomeStep):
                st.markdown(text_to_markdown(last_step.format_st_message()))
            else:
                st.write(f'Finished with unexpected step: {last_step}')

            st.session_state.chat_steps = steps

