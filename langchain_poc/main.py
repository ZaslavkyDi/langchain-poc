from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain


_OPEN_AI_MODEL_NAME = "gpt-3.5-turbo"
_OPEN_AI_TEMPERATURE = 0  # range from 0 to 2. Default 1. Randomizing response from GPT if more the 0

information_placeholder = """
Farion graduated from the Philology School of the Lviv University in 1987 with honors[citation needed], while her name was entered in the book "Toiling glory of University"[citation needed]. During the college years she was a member of a Communist Party of the Soviet Union (the only student being in the Communist Party[3]).

In 1996 she defended her candidate dissertation. Since 2006 Farion became politically active balloting for People's Deputy of Ukraine mandate from the All-Ukrainian Union "Svoboda", of which she was a member since 2005. In 2006 Farion also successfully balloted to the regional council, while in 2010 she won in a majoritarian electoral district of Lviv.

Among her scientific works are four monographs and 200 articles[citation needed]. During 1998–2004 Farion headed language commission of Prosvita. Since 1998 she initiated and organized the annual competition among students "Language is a foundation of your life". In 2004 Farion became a laureate of Oleksa Hirnyk Prize (Oleksa Hirnyk). Farion publicly advocates the memory of Stepan Bandera,[4] unity of the Ukrainian West and East based on a statist thinking.

In the 2012 parliamentary election Farion was elected into parliament after winning a constituency in Lviv Oblast.[5]

In the 2014 parliamentary election Farion again tried to win a constituency seat in Lviv, but failed this time having finished third in her constituency with approximately 16% of votes.[6]

In the July 2019 Ukrainian parliamentary election Farion again failed to return to parliament after finishing fifth with 10.35% of the vote in electoral district 116 in Lviv Oblast.[7]

In November 2023, she had a clash with Maksym Zhorin [uk] and Bohdan Krotevych after claiming that she cannot call the Russian-speaking fighters of the Azov Brigade Ukrainians.[8] During this time, she also failed, and later refused, to blur the name of her supporter from Russian-occupied Crimea on a screenshot of his letter of gratitude, causing him to be arrested by Russian authorities.[9] This caused public outrage, including protests of students of the Lviv Polytechnic Institute, but the institute refused to fire her[10]. On 15 November 2023, the Security Service of Ukraine has opened an investigation against her on the counts of discrimination, insulting the dignity of a serviceman, violation of confidentiality of correspondence, and breach of inviolability of private life,[11] and she was relieved of her position in the Lviv Polytechnic.[10]
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
        temperature=_OPEN_AI_TEMPERATURE,
        model_name=_OPEN_AI_MODEL_NAME
    )

    chain = LLMChain(llm=llm, prompt=prompt_summery_template)

    print(chain.run(information=information_placeholder))