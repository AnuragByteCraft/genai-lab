"""
In this demo we illustrate 3 worker agents and 1 traige agent. This is a router agentic pattern.
We also illustrate tool use.

IMPORTANT:

This demo breaks in mac with latest OpenAI Agent SDK on gemma-3-12b but works on gpt-oss-20b

"""
import asyncio
from datetime import datetime, timedelta

from agents import Agent, function_tool
from agents import Runner

from core.utils.openai_agent_sdk_settings import initialize_openai_agent_sdk

# import openai


# One of the distinguishing characteristics of an Agentic AI application is ability to invoke tools.
# Below we define a set of tools and agents for our demo. These use OpenAI Agent SDK.

# ------------------------------------ Function Tools -------------------------------------------------
@function_tool
def get_todays_date_day_and_time():
    """
    This provides a dictionary with information on date, day and time in 12 hour format.
    You can use this when you need all date, day, time.
    :return: a dict containing date/day/time, each as a str
    """
    print("I am in get_today_datetime")
    now = datetime.now()

    return {
        "date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A"),
        "time": now.strftime("%I:%M %p")  # 12-hour format with AM/PM
    }


@function_tool
def get_future_date(days_from_now: int):
    """
    Calculates a future date and its corresponding day of the week.
    Use this to find the date for 'tomorrow', 'day after tomorrow', or a specific number of days from now.
    For example, to find tomorrow's date, you would call this tool with the argument days_from_now=1.
    :param days_from_now: The number of days to add to today's date. Must be an integer.
    :return: A dict containing the future date and the day of the week.
    """
    print(f"Executing get_future_date() with days_from_now={days_from_now}")
    future_date = datetime.now() + timedelta(days=days_from_now)
    retval = {
        "date": future_date.strftime("%Y-%m-%d"),
        "day": future_date.strftime("%A"),
    }
    print("returning from get_future_date: ", retval)
    return retval

# -------------------------------------- Agent Definitions ----------------------------------------------------

history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly.",
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You are a math professor. Explain your reasoning at each step and include examples. Your output "
                 "should be printable in a text console.",
)

date_prompt = (
        "You are an expert assistant for all date and time queries. You have three tools. "
        "To solve a query about a FUTURE date (like 'tomorrow', 'next week', or 'the weekend'), "
        "you MUST follow a multi-step plan: "
        "1. First, call get_today_datetime() to determine the current day of the week. "
        "2. Second, based on the current day, calculate the integer number of days until the desired future date. "
        "3. Third, call the get_future_date() tool with the correct 'days_from_now' integer argument. You might need to call it multiple times for a date range like a weekend."
    )


today_agent = Agent(
    name="Today Agent",
    handoff_description="Expert on all date and time related queries, including calculating future dates.",
    instructions=date_prompt,
    tools=[get_todays_date_day_and_time, get_future_date]
)


coder_agent = Agent(
    name="Coder Agent",
    handoff_description="Expert in Python coding.",
    instructions="You are an expert Python coder. Given a task generate well structured and modular Python code.",
)

# triage_agent = Agent(
#     name="Triage Agent",
#     instructions="You determine which agent to use based on the user's question",
#     instructions_role="user",
#     handoffs=[history_tutor_agent, math_tutor_agent, today_agent, coder_agent],
#     handoff_description="Router agent that decides which specialist agent should answer.",
# )

triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's question",
    handoff_description="Router agent that decides which specialist agent should answer.",
    handoffs=[history_tutor_agent, math_tutor_agent, today_agent, coder_agent],
    tools=[]   # ← Add this!!!
)

# --------------------- main function that runs the orchestrator as Runner.run() ------------------------


async def main():
    result = await Runner.run(triage_agent, "Prove that the derivative of sin_theta squared is sin(2*theta)")
    print("result = ", result.final_output)

    result = await Runner.run(triage_agent, "When did Mahabharata war take place?")
    print("result = ", result.final_output)

    result = await Runner.run(triage_agent, "What is day after tomorrow's date?")
    print("result = ", result.final_output)

    result = await Runner.run(triage_agent, "What is the date of coming weekend?")
    print("result = ", result.final_output)

    result = await Runner.run(triage_agent, "I have a Wedding to attend on Nov 27. Usually I am busy with work on Thursdays. Will I be free to attend the wedding?")
    print("result = ", result.final_output)

    result = await Runner.run(triage_agent, "Implement quicksort algorithm and illustrate with an example.")
    print("result = ", result.final_output)

# -----------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':
    # We need to do this first in order to make our local LLM as default
    initialize_openai_agent_sdk()

    # OpenAI SDK runs as async and so invoke it as below
    asyncio.run(main())
