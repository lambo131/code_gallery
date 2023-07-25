import os
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

class generator:
    def __init__(self):
        load_dotenv(find_dotenv())
        openai_api_key =  os.getenv("OPENAI_API_KEY")
        print("openai_api_key: ", openai_api_key)
        self.llm = OpenAI(temperature=0, openai_api_key=openai_api_key)

    def get_ans(self, dish):
        template = """
        generate a list of new, none-existing resturant name based on these dishes:
        {dish_names}
        """
        prompt = PromptTemplate(
            input_variables=["dish_names"],
            template=template,
        )
        final_prompt = prompt.format(dish_names=dish)
        return self.llm(final_prompt)
    #