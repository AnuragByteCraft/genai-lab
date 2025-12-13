"""
This module implements a tool that gets the summaries - at file and folder levels
"""
import json

SUMMARIES_FILE = r"/Users/ananth/PycharmProjects/agentic_ai_nov_2025/core/summarizer/nanochat_folder_summary.json"

def get_folder_summaries():
    """
    Returns the folder level summaries for the nanochat repo
    :return: A dict of folder name and corresponding summary
    """
    with open(SUMMARIES_FILE, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    children = summaries["children"]

    folder_summaries = {}
    critical_file_summaries = {}

    important_folders = ["nanochat", "scripts", "rustbpe"]

    for child in children:
        name = child["path"]
        name1 = name.split("/")[-1]
        summary = child["summary"]
        if summary is not None and len(summary):
            folder_summaries[child["path"]] = summary

        if name1 in important_folders:
            children1 = child["children"]
            for i, child1 in enumerate(children1):
                file_name = child1["path"]
                child_summary = child1["summary"]
                # print(i, file_name)
                if child_summary is not None and len(child_summary):
                    critical_file_summaries[file_name] = child_summary

    results = {
        "folder_summaries": folder_summaries,
        "critical_file_summaries": critical_file_summaries,
    }

    return results


def get_file_summaries():
    summary_file_path = r"/Users/ananth/PycharmProjects/agentic_ai_nov_2025/core/summarizer/summary.json"
    with open(summary_file_path, "r", encoding="utf-8") as f:
        summaries = json.load(f)
    summaries = list(summaries.values())
    summaries = [summ["summary"] for summ in summaries]
    summaries = [summ for summ in summaries if summ is not None]
    summaries = "\n\n".join(summaries)
    return summaries


if __name__ == '__main__':
    print(get_file_summaries())

    # summ = get_folder_summaries()
    # folder_summaries = summ["folder_summaries"]
    # critical_file_summaries = summ["critical_file_summaries"]
    #
    # print(summ)
    #
    # for pth, summary in folder_summaries.items():
    #     print(pth, " => ", summary)
    #     print("-" * 100)
    #
    # for pth, summary in critical_file_summaries.items():
    #     print(pth, " => ", summary)
    #     print("-" * 100)



