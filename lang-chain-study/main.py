# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import openai
import os

import torch
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from modelscope import AutoTokenizer, AutoModel, snapshot_download

model_dir = snapshot_download("ZhipuAI/chatglm3-6b", revision="v1.0.0")
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).float().to(device)
model = model.eval()

prompt = PromptTemplate(
    input_variables=["product"],
    template="What is a good name for a company that makes {product}?",
)


# os.environ["OPENAI_API_KEY"] = "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"


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    # llm = OpenAI(temperature=0.9)
    # text = "What would be a good company name for a company that makes colorful socks?"
    # print(llm(text))

    print(prompt.format(product="colorful socks"))

    # response, history = model.chat(tokenizer, "你好", history=[])
    # print(response)
    # response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
    # print(response)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
