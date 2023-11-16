from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain

from langchain_poc.config import get_openai_settings

_OPEN_AI_MODEL_NAME = "gpt-3.5-turbo"
_OPEN_AI_TEMPERATURE = 0  # range from 0 to 2. Default 1. Randomizing response from GPT if more the 0

information_placeholder = """
Farion graduated from the Philology School of the Lviv University in 1987 with honors[citation needed], while her name was entered in the book "Toiling glory of University"[citation needed]. During the college years she was a member of a Communist Party of the Soviet Union (the only student being in the Communist Party[3]).

In 1996 she defended her candidate dissertation. Since 2006 Farion became politically active balloting for People's Deputy of Ukraine mandate from the All-Ukrainian Union "Svoboda", of which she was a member since 2005. In 2006 Farion also successfully balloted to the regional council, while in 2010 she won in a majoritarian electoral district of Lviv.
"""

if __name__ == "__main__":
    summary_template = """
    give the information {information} about a person from I want you to create:
    1. a short summery
    2. two interesting facts about them
    """

    prompt_summery_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOpenAI(
        api_key=get_openai_settings().api_key,
        temperature=_OPEN_AI_TEMPERATURE,
        model_name=_OPEN_AI_MODEL_NAME
    )

    chain = LLMChain(llm=llm, prompt=prompt_summery_template)

    print(chain.run(information=information_placeholder))