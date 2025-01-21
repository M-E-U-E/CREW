from crewai import LLM, Agent, Task, Crew
import warnings
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

llm = LLM(
    model="openai/gpt-4",
    temperature=0.7
)

# Event Planner Agent
planner = Agent(
    role="Event Planner",
    goal="Identify events based on location, date, and preferences",
    backstory="You're tasked with finding events in the user's area "
              "that match their preferences and timing. Use APIs or mock data.",
    allow_delegation=False,
    llm=llm,
    verbose=True
)

# Weather Forecaster Agent
forecaster = Agent(
    role="Weather Forecaster",
    goal="Provide accurate weather forecasts for specific locations and dates.",
    backstory="You're responsible for checking weather conditions for the events "
              "suggested by the Event Planner.",
    allow_delegation=False,
    llm=llm,
    verbose=True
)

# Activity Recommender Agent
recommender = Agent(
    role="Activity Recommender",
    goal="Suggest activities or events combining user preferences and weather data.",
    backstory="You work with the Event Planner and Weather Forecaster "
              "to recommend the best options to the user.",
    allow_delegation=False,
    llm=llm,
    verbose=True
)

# Define tasks
event_task = Task(
    description=(
        "Find events based on the following inputs:\n"
        "- Location: {location}\n"
        "- Date: {date}\n"
        "- Preferences: {preferences}\n"
        "Return a list of events matching these inputs."
    ),
    expected_output="A list of events with details (name, location, date, type).",
    agent=planner
)

weather_task = Task(
    description=(
        "Fetch weather conditions for the following:\n"
        "- Location: {location}\n"
        "- Date: {date}\n"
        "Return weather data (temperature, conditions, suitability for outdoor activities)."
    ),
    expected_output="Weather forecast data for the specified location and date.",
    agent=forecaster
)

recommendation_task = Task(
    description=(
        "Based on event details and weather data, provide activity recommendations:\n"
        "- Suggest events suitable for the weather.\n"
        "- Include alternatives for unsuitable conditions."
    ),
    expected_output="A list of tailored event recommendations for the user.",
    agent=recommender
)

# Crew setup
crew = Crew(
    agents=[planner, forecaster, recommender],
    tasks=[event_task, weather_task, recommendation_task],
    verbose=True
)

# Inputs for the bot
inputs = {
    "location": "bangkok",
    "date": "this weekend",
    "preferences": "outdoor, family-friendly"
}

# Execute the workflow
result = crew.kickoff(inputs=inputs)

from IPython.display import Markdown
Markdown(result.raw)
