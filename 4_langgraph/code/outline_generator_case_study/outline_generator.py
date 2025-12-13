"""
Generates outline for a chapter given the slides content and transcripts information
"""
import os
import json

from core.llm.get_completion_client import get_completion_messages
from core.case_study_outline_generator.process_slides import get_contents_for_ppt


def create_outline_from_slides(contents=None, outline=None, feedback=None):
    system_prompt = """
    You are a book chapter outliner.

You will be given either:

1. A slide deck (JSON format), OR  
2. A previously generated outline and feedback from a critic.

---

When provided with a **slide deck**, generate a chapter outline with the following structure:
- chapterTitle
- sections (each with sectionHeading, subsections, and optional subsubsections)
- Each subsection should have a "content" field with a 1–2 sentence summary

When provided with an **outline and critic feedback**, revise the outline to:
- Address the critic’s concerns
- Improve structure and clarity
- Preserve as much valid content as possible

---

Always output a valid **strict JSON object**. Use **double quotes** for all keys and values. Do not include markdown, comments, or explanation.


    """
    if contents:
        prompt = "Slide Deck in JSON: \n" + contents + "\n"
    else:
        prompt = "Previous Outline: \n" + outline + "\n"
        prompt += "Feedback from critic: \n" + feedback + "\n"

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },

        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = get_completion_messages(messages)
    response = response.split("```json")[-1].split("```")[0]
    print(response)
    response = json.loads(response)

    return response


def outline_critic(outline:str) -> str:
    system_prompt = """
    You are a critical reviewer of book chapter outlines.

You will be given a chapter outline (in JSON format). Your task is to:
- Evaluate its structure, clarity, and coherence
- Point out missing, redundant, or confusing parts
- Identify opportunities to improve flow, grouping, or naming
- Be concise but constructive

Your output should be a plain-text critique, structured as a list of specific suggestions.

Do not regenerate the outline — just review and comment.
Provide only the feedback. No other explanation or comments are required.

    """
    prompt = "Outline to be reviewed: \n" + outline + "\n"

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },

        {
            "role": "user",
            "content": prompt,
        },
    ]

    response = get_completion_messages(messages)
    # response = response.split("```json")[-1].split("```")[0]
    print(response)
    # response = json.loads(response)

    return response


if __name__ == '__main__':
    ppt_folder = r"C:\home\ananth\trainings\adobe\genai_2025\agentic_ai_july_2025"
    name = "session1_beyond_parrots_and_calculators.pptx"
    ppt_file_name = os.path.join(ppt_folder, name)
    contents = get_contents_for_ppt(ppt_file_name)
    contents = json.dumps(contents)
    outline = create_outline_from_slides(contents)
    # print(outline)
