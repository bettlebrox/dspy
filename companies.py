from functools import partial
import os
from typing import Any
import dspy
import dspy.evaluate
from dspy.teleprompt import BootstrapFewShot
from dspy.predict.avatar import Avatar, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import BrowserbaseLoader
from dotenv import load_dotenv
from openinference.instrumentation.dspy import DSPyInstrumentor
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

load_dotenv()
endpoint = "https://app.phoenix.arize.com/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(
    SimpleSpanProcessor(span_exporter=span_otlp_exporter)
)
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()


# Set up the LM.
model = dspy.OpenAI(model="gpt-4o")
dspy.settings.configure(lm=model)


trainset = [
    dspy.Example(
        criteria=[
            "b2b SaaS company",
        ],
        list_of_matches=["Slack", "Salesforce", "Hubspot", "Shopify", "Zendesk"],
    ),
    dspy.Example(
        criteria=[
            "recently funded company",
        ],
        list_of_matches=[
            "AstraZeneca",
            "Argo Blockchain",
            "Eastman Chemical Company",
            "HydroMind",
        ],
    ),
    dspy.Example(
        criteria=[
            "Fortune 500 company",
        ],
        list_of_matches=[
            "Walmart",
            "Amazon",
            "Apple",
            "Saudi Armco",
            "Chine Natural Petroleum",
            "Shell",
        ],
    ),
]
trainset = [example.with_inputs("criteria") for example in trainset]


class FindCompany(dspy.Signature):
    criteria: list[str] = dspy.InputField()
    list_of_matches: list[str] = dspy.OutputField()


class BrowserbaseTool:
    def run(self, website: str, **kwargs: Any) -> str:
        """Retrieve website on browserbase and parse result."""
        self.browserbaseLoader = BrowserbaseLoader(
            [website], text_content=True, api_key=os.getenv("BROWSERBASE_API_KEY")
        )
        return self.browserbaseLoader.load()


tools = [
    Tool(
        tool=GoogleSerperAPIWrapper(),
        name="Google Search",
        desc="Search the web for relevant websites.",
    ),
]
researcher = BootstrapFewShot(dspy.evaluate.answer_exact_match)
complied_researcher = researcher.compile(
    Avatar(
        FindCompany,
        tools=tools,
        verbose=True,
    ),
    trainset=trainset,
)

answer = complied_researcher(
    criteria=[
        "open role for VP of Sales",
        "ecommerce company",
    ]
)
print(answer)
