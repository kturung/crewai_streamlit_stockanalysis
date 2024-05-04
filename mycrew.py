from crewai import Agent, Task, Crew, Process
from functions import *
from langchain_groq import ChatGroq
import streamlit as st
from langchain_core.agents import AgentFinish, AgentAction
        

st.session_state.api_key = st.sidebar.text_input(
        label="Groq API KEY", placeholder="Your API KEY"
    )

if not st.session_state.api_key:
    st.info("Please enter your GROQ API KEY to start.")
    st.stop()

llm = ChatGroq(api_key=st.session_state.api_key, model="llama3-70b-8192", temperature=0, max_tokens=3000)

price_tool = GetHistoricalPriceCrew()
chart_tool = PlotLineChartBaseCrew()
manuel_tool = NoneToolCrew()

planner_price_tool = GetHistoricalPriceBase()
planner_chart_tool = PlotLineChartBase()


executer_tools = [price_tool, chart_tool, manuel_tool]
planner_function_definitions = get_openai_functions_definitions([planner_price_tool, planner_chart_tool])


def callback_processor(response):
  expander = None
  if type(response) == AgentFinish:
    placeholder = st.empty()
    with placeholder.container():
      expander = st.expander(f"**Agent Finished!**", expanded=True)
      expander.write(response.return_values['output'])
  else:
    agent_action_dict = response[0]
    agent_action = AgentAction(agent_action_dict[0].tool, agent_action_dict[0].tool_input  , agent_action_dict[0].log)
    placeholder = st.empty()
    with placeholder.container():
      expander = st.expander(f"**{agent_action.tool}**", expanded=False)
      expander.write(agent_action.log.replace('\n', '\n\n'))
      expander.write(f"Observation: {agent_action_dict[1]}")



planner = Agent(
  role='Planner',
  goal='Create a structured step by step plan to achive the user\'s request',
  backstory='A meticulous planner with an expertise in structuring tasks for stock analysis reports',
  verbose=True,
  allow_delegation=False,
  llm=llm,
  step_callback=callback_processor
)

executer = Agent(   
  role='Executer',
  goal='Execute the tasks as per the plan',
  backstory='An efficient executer with a knack for completing tasks in a timely manner',
  verbose=True,
  tools=executer_tools,
  allow_delegation=False,
  llm=llm,
  max_iter=11,
  step_callback=callback_processor
  )



planner_task = Task(
  description="""Create a structured step by step plan report for this user request:\n
      {question}\n\n"""
      f"""You need to utilize these tools:\n\n
      {str(planner_function_definitions).replace('{', '{{').replace('}', '}}')} \n\n
      Your plan should contain as few steps as possible, but each step should be clear and concise.\n\n
      Do not include tool parameters in the plan.
      """,
  expected_output="""A report in markdown strictly in this format:\n\n
  ## Plan\n
  ### User Request\n
  - User request summary here\n
  ### Step Number here\n
  - Description: Step description here.\n
  ### Step Number here\n
  - Description: Step description here.\n
  ...
  """,
  agent=planner
)


executer_task = Task(
  description="""Execute the steps as per the plan\n\n
      If the Tool is None for a step in the plan, you should use the manuel-processing-tool to process that step.\n\n
      Always refer to the Step Number in the plan on Thought: to know which step you are currently executing.\n\n""",
  expected_output="""I've completed all the tasks as per the plan.""",
  agent=executer,
  tools=executer_tools
)




# Form the crew with a sequential process

report_crew = Crew(

  agents=[planner, executer],

  tasks=[planner_task, executer_task],

  process=Process.sequential,

  verbose=True
)

def generate_response(user_prompt):
  response = report_crew.kickoff({
    "question": user_prompt
  })

  return response



