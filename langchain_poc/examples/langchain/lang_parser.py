from langchain.output_parsers import PydanticOutputParser, ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field, field_validator

from langchain_poc.examples.langchain.base import BaseExample


class LangParser(BaseExample):
    email_response = """
        Here's our itinerary for our upcoming trip to Europe.
        We leave from Denver, Colorado airport at 8:45 pm, and arrive in Amsterdam 10 hours at Schipol Airport.
        We'll grab a ride to our airbnb and maybe stop somewhere for breakfast before
        taking a nap.
        
        Some sightseeing will follow for a couple of hours.
        We will then go shop for gifts to bring back to our children and friends.
        
        The next morning, at 7:45am we'll drive to to Belgium, Brussels. 
        While in Brussels we want to explore the city to its fullest.
    """

    email_template = """
        From the following email, extract the following information:
        
        leave_time: when are they leaving for vacation to Europe. 
        If there's an actual time written, use it, if not write unknown.
        
        leave_from: where are they leaving from, the airport or city name and state if available.
        
        cities_to_visit: extract the cities they are going to visit.
        If there are more than one, put them in square brackets like '["cityone", "citytwo"].
        
        email: {email}
    """

    def run_simple_example(self) -> None:
        email_template_with_output_format = (
            self.email_template
            + """
        Format the output as JSON with the following keys: 
        leave_time 
        leave_from 
        cities_to_visit
        """
        )

        prompt_template = ChatPromptTemplate.from_template(email_template_with_output_format)

        messages = prompt_template.format_messages(email=self.email_response)
        response = self.chat_model(messages)

        print(response.content)

    def run_structured_parser_example(self) -> None:
        leave_time_schema = ResponseSchema(
            name="leave_time",  # field name that we want to see in response payload
            description=(  # what is supposed to do or have
                "When they are leaving. It's usually a numberical time of the day."
                "If not available write n/a."
            ),
        )
        leave_from_schema = ResponseSchema(
            name="leave_from",
            description=(
                "Where are they leaving from. It's a airport, city, state or province."
                "If not available write n/a."
            ),
        )
        cities_to_visit_schema = ResponseSchema(
            name="cities_to_visit",
            description=(  # what is supposed to do or have
                "The cities, towns they are going to visit during they trip. This need to be in a Python list."
                "If not available write empty Python list."
            ),
        )

        # the order is important
        response_schemas = [
            leave_time_schema,
            leave_from_schema,
            cities_to_visit_schema,
        ]

        # setup output parser
        output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

        format_instructions = output_parser.get_format_instructions()
        email_template_with_instruction = self.email_template + "{format_instructions}"

        email_prompt = ChatPromptTemplate.from_template(email_template_with_instruction)
        messages = email_prompt.format_messages(
            email=self.email_response, format_instructions=format_instructions
        )

        response: BaseMessage = self.chat_model(messages)
        print(type(response.content))
        print(response.content)

        output_dict = output_parser.parse(response.content)
        print(type(output_dict))
        print(output_dict)

    def run_pydantic_parser_example(self) -> None:
        class VacationInfoSchema(BaseModel):
            leave_time: str = Field(
                description="When they are leaving. It's usually a numberical time of the day"
            )
            leave_from: str = Field(
                description="Where are they leaving from. It's a airport, city, state or province"
            )
            cities_to_visit: list[str] = Field(
                escription=(
                    "The cities, towns they are going to visit during they trip. "
                    "This need to be in a Python list"
                )
            )
            num_people: int = Field(
                description="This is integer for a number of people on this trip"
            )

            @field_validator("num_people", mode="after")
            def check_num_people(cls, num_people: int) -> int:
                if num_people <= 0:
                    raise ValueError("Badly formatted field.")
                return num_people

        # pydantic setup parser and inject the instruction
        pydantic_parser = PydanticOutputParser(pydantic_object=VacationInfoSchema)
        format_instructions: str = pydantic_parser.get_format_instructions()

        email_template = """
            From the following email, extract the following information regarding this trip.
            
            email: {email}
            
            {format_instructions}
        """
        prompt_template = ChatPromptTemplate.from_template(email_template)
        messages = prompt_template.format_messages(
            email=email_template, format_instructions=format_instructions
        )

        response = self.chat_model(messages)
        print(type(response.content))
        print(response.content)

        vacation_item: VacationInfoSchema = pydantic_parser.parse(response.content)
        print(vacation_item)
