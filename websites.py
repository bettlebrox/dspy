from dotenv import load_dotenv

load_dotenv()
import os
import dspy
import dspy.evaluate
from dsp.utils import dotdict
from typing import List, Union, Optional
from dspy.teleprompt import BootstrapFewShot
from openinference.instrumentation.dspy import DSPyInstrumentor
from dspy.predict.avatar import Avatar, Tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import BrowserbaseLoader
from opentelemetry import trace as trace_api
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk import trace as trace_sdk
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from wrapt.patches import wrap_object

endpoint = "https://app.phoenix.arize.com/v1/traces"
resource = Resource(attributes={})
tracer_provider = trace_sdk.TracerProvider(resource=resource)
span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
tracer_provider.add_span_processor(
    SimpleSpanProcessor(span_exporter=span_otlp_exporter)
)
trace_api.set_tracer_provider(tracer_provider=tracer_provider)
DSPyInstrumentor().instrument()


class GoogleSerperRMClient(dspy.Retrieve):
    name: str = "GoogleSearch"
    desc: str = "Search the web for relevant websites."

    def __init__(self, k: int):
        super().__init__(k=k)
        self.wrapper = GoogleSerperAPIWrapper(k=k, gl="ie")

    def forward(self, query: Union[str, List[str]]) -> dspy.Prediction:
        results = self.wrapper.results(query.lstrip('"').rstrip('"'))
        passages = [dotdict(result) for result in results["organic"]]
        for passage in passages:
            passage.long_text = passage.snippet
        return dspy.Prediction(passages=passages)


class BrowserbaseRMClient(dspy.Retrieve):
    name: str = "Browserbase"
    desc: str = "Retrieve website on browserbase and parse result."

    def __init__(self, k: int):
        super().__init__(k=k)

    def forward(self, websites: Union[str, List[str]]) -> dspy.Prediction:
        """Retrieve website on browserbase and parse result."""
        self.browserbaseLoader = BrowserbaseLoader(
            [websites.lstrip('"').rstrip('"')],
            text_content=True,
            api_key=os.getenv("BROWSERBASE_API_KEY"),
        )
        return dspy.Prediction(passages=self.browserbaseLoader.load())


# Set up the LM.
model = dspy.OpenAI(model="gpt-4o")
dspy.settings.configure(lm=model, rm=GoogleSerperRMClient(k=5))

trainset = [
    dspy.Example(
        criteria="b2b SaaS company",
        answer=[
            "[Marketer Milk](https://www.marketermilk.com/blog/saas-companies)",
            "[BuiltIn](https://builtin.com/articles/top-b2b-saas-companies)",
            "[Exploding Topics](https://explodingtopics.com/blog/b2b-saas-startups)",
            "[Growth list](https://growthlist.co/b2b-saas-startups/)",
            "[Datamation](https://www.datamation.com/cloud/saas-companies)",
        ],
    ),
    dspy.Example(
        criteria="recently funded company",
        answer=[
            "[TechCrunch](https://techcrunch.com/tag/funding/)",
        ],
    ),
    dspy.Example(
        criteria="irish startup",
        answer=[
            "[Startups.ie](https://startups.ie/)",
            "[Start in Ireland](https://www.startinireland.com/)",
            "[Dogpatch Labs](https://dogpatchlabs.com/)",
        ],
    ),
]
trainset = [example.with_inputs("criteria") for example in trainset]


class WebsiteSearch(dspy.Signature):
    """Search the web for relevant websites."""

    criteria: str = dspy.InputField(
        desc="Search criteria for websites that list companies."
    )
    answer: list[str] = dspy.OutputField(
        desc="Links to websites that list companies matching the criteria in markdown format."
    )


def validate_answer(example, pred, trace=None):
    return True


optimizer = BootstrapFewShot(validate_answer)

compiled = optimizer.compile(
    dspy.ReAct(WebsiteSearch, tools=[GoogleSerperRMClient(k=5)]),
    trainset=trainset,
)
compiled.save("compiled.json")
answer = compiled(criteria="medtech companies")
print(answer.answer)
