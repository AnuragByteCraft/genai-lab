import os
import lmstudio as lms

model = lms.llm()


def image_chat(image_path, prompt):
    chat = lms.Chat()
    image_handle = lms.prepare_image(image_path)
    chat.add_user_message(prompt, images=[image_handle])
    prediction = model.respond(chat)
    return prediction.content


if __name__ == '__main__':
    DATAPATH = r"/Users/ananth/PycharmProjects/book_generator/ppt_images/chapter4"
    name = "image_002_slide_3"

    # --------------------------- Some sample queries for testing -----------------------------
    # text = "What are the categories of prompting shown in the image?"
    # text = "Explain the content of the image and suggest a good prompting technique for code generation."
    # text = "Extract all prompting techniques from the image along with their categories and return the output in JSON form."
    # ------------------------------------------------------------------------------------------

    fname = os.path.join(DATAPATH, name + ".png")
    print(fname)

    while True:
        text = input("Enter your prompt: ")
        if text.lower() in ["q", "quit", "exit", "bye"]:
            break
        preds = image_chat(fname, text)
        print(preds)