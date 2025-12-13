"""
Given a slide deck in PPT form, process them
"""
import os
from langchain_community.document_loaders import UnstructuredPowerPointLoader


def get_docs(ppt_file):
    loader = UnstructuredPowerPointLoader(ppt_file, mode="elements")
    docs = loader.load()
    return docs


def get_slide_contents(doc):
    mydata = {
        "content": doc.page_content,
        "metadata": {
            "category_depth": doc.metadata.get("category_depth", None),
            "page_number": doc.metadata["page_number"],
            "category": doc.metadata["category"],
        }
    }

    return mydata


def get_slidedeck_contents(docs):
    contents = []

    for doc in docs:
        mydata = {
                    "content": doc.page_content,
                    "metadata": {
                        "category_depth": doc.metadata.get("category_depth", None),
                        "page_number": doc.metadata["page_number"],
                        "category": doc.metadata["category"],
                    }
        }
        contents.append(mydata)
    return contents


def get_contents_for_ppt(filename):
    docs = get_docs(filename)
    contents = get_slidedeck_contents(docs)
    return contents


if __name__ == '__main__':
    ppt_folder = r"C:\home\ananth\trainings\adobe\genai_2025\agentic_ai_july_2025"
    name = "session1_beyond_parrots_and_calculators.pptx"
    ppt_file_name = os.path.join(ppt_folder, name)
    docs = get_docs(ppt_file_name)
    contents = get_slidedeck_contents(docs)
    print(contents)

