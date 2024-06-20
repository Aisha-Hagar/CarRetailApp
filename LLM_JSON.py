from langchain.chat_models import AzureChatOpenAI
import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_structured_output_runnable
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, dotenv_values 
import yaml

class LLM:
    
    def __init__(self):
        #Load OpenAI LLM
        load_dotenv() 
        
        os.environ["OPENAI_API_TYPE"] = os.getenv("OPENAI_API_TYPE")
    
        os.environ["OPENAI_API_BASE"] = os.getenv("OPENAI_API_BASE")
        
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
        
        self.chat = AzureChatOpenAI(
        
            openai_api_version="2024-02-01",
        
            azure_deployment="deployment-testing",

            temperature=0
        
        )
        
        #Define json schema
        self.car_schema = {
            "type": "function",
            "function": {
                "name": "car_attributes",
                "description": "Record some identifying attributes of a car.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "car": {
                            "description": "The attributes of the car",
                            "type": "object",
                            "properties": {
                                "color": {
                                    "description": "The color of the car",
                                    "type": "string"
                                },
                                "brand": {
                                    "description": "The brand name of the car",
                                    "type": "string"
                                },
                                "model": {
                                    "description": "The model name of the car",
                                    "type": "string"
                                },
                                "manufactured_year": {
                                    "description": "The year the car was manufactured",
                                    "type": "integer"
                                },
                                "motor_size_cc": {
                                    "description": "The motor size of the car",
                                    "type": "integer"
                                },
                                "tires": {
                                    "description": "The attributes of the tires",
                                    "type": "object",
                                    "properties": {
                                        "type": {
                                            "description": "The tires type either new or used",
                                            "type": "string"
                                        },
                                        "manufactured_year": {
                                            "description": "The year the tires were manufactured",
                                            "type": "integer"
                                        }
                                    },
                                    "required": [
                                        "type",
                                        "manufactured_year"
                                    ]
                                },
                                "windows": {
                                    "description": "The type of car windows",
                                    "type": "string"
                                },
                                  "notices": {
                                      "description": "Notices or remarks or aditional information",
                                      "type": "array",
                                      "items":
                                       {
                                           "type": "object",
                                           "properties": {
                                               "type": {
                                                   "description": "The type of notice or remark or aditional information",
                                                    "type": "string"
                                                    },
                                               "description": {
                                                   "description": "The notice or remark or aditional information",
                                                   "type": "string"
                                                   }
                                               },
                                           "required": [
                                               "type",
                                               "description"
                                               ]
                                           }
                                      },
                                  "price": {
                                      "description": "Price of the car",
                                      "type": "object",
                                      "properties": {
                                          "amount": {
                                              "description": "The price of the car",
                                              "type": "integer"
                                              },
                                          "currency": {
                                              "description": "The currency of price of the car",
                                              "type": "string"
                                              }
                                          },
                                      "required": [
                                          "amount",
                                          "currency"
                                          ]
                                      }
                                  },
                              "required": [
                                  "body_type",
                                  "color",
                                  "brand",
                                  "model",
                                  "manufactured_year",
                                  "motor_size_cc",
                                  "tires",
                                  "windows",
                                  "notices",
                                  "price"
                                  ]
                              }
                          },
                      "required": [
                          "car"
                          ]
                      }
                  }
              }

        #Reference: https://python.langchain.com/v0.1/docs/use_cases/extraction/quickstart/
        self.template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert extraction algorithm."
                    "Given a description of a car, exctract a dictionary that contains the car attributes."
                    "Only extract relevant information from the car description text. "
                    "If there you are not given a car descrption return null for all attributes."
                    "Strictly use the given description only without making up any information."
                    "If you do not know the value of an attribute you are asked to extract, "
                    "return null for the attribute's value.",
                ),
                ("human", "{user_input}"),
            ]
        )

        
        """
        To extract structured output, the method used in this code 'create_structured_output_runnable' is stated as deprecated in the LangChain docs and replaced with a method called with_structured_output. However, at the time of writing this code, this method hadn't been implemented for OpenAI yet.
        """
        self.structured_llm = create_structured_output_runnable(
            self.car_schema,
            self.chat,
            mode="openai-tools",
            enforce_function_usage=True,
            return_single=True
        )
        

    def llm_response(self,text):
        
        prompt = self.template.invoke({"user_input": text})
        print(prompt)
        llm_output = self.structured_llm.invoke(prompt)
        
        return llm_output
