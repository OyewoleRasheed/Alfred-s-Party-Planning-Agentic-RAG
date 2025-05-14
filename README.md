# Alfred-Agentic-RAG🦇🎉

*A Retrieval-Augmented Agent with Web Search & Custom Knowledge Base*

This project replicates and **extends** the [Hugging Face Agents Course - Retrieval Agents Notebook](https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/retrieval_agents.ipynb). It demonstrates a hybrid **Web + Knowledge Base Retrieval Agent** designed to help Alfred (the Wayne Manor butler) plan a luxury superhero-themed party!

The agent follows this process:

1. **Analyzes the Request**: Identifies key elements like party theme, entertainment, catering, and decoration.
2. **Performs Retrieval**: Searches live web results via DuckDuckGo for up-to-date luxury event ideas.
3. **Synthesizes Information**: Combines results into an actionable plan.
4. **Stores for Future Reference**: Uses a custom semantic search tool over a private knowledge base for instant access to prior curated ideas.

---

## 🚀 Features

* 🔎 **Live Web Search** with DuckDuckGo
* 📚 **Custom Knowledge Base** powered by semantic search (BM25)
* 🤖 **Retrieval-Augmented Generation (RAG)** for synthesizing results
* 🛠️ **Extendable Tools** using the SmolAgents framework
* 🦸‍♂️ **Domain-Specific Retrieval** (e.g., superhero-themed party planning)

---

## 🛠️ Installation

Install the required libraries:

```bash
!pip install -U smolagents
!pip install duckduckgo-search
!pip install langchain langchain-community
```

---

## 📚 Usage

### 1. Login to Hugging Face Hub

```python
from huggingface_hub import notebook_login
notebook_login()
```

### 2. Create Custom Knowledge Base Tool

```python
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from smolagents import Tool, CodeAgent, HfApiModel

# Simulated knowledge base about luxury superhero-themed party planning
party_ideas = [
    {"text": "A superhero-themed masquerade ball with luxury decor, including gold accents and velvet curtains.", "source": "Party Ideas 1"},
    {"text": "Hire a professional DJ who can play themed music for superheroes like Batman and Wonder Woman.", "source": "Entertainment Ideas"},
    {"text": "For catering, serve dishes named after superheroes, like 'The Hulk's Green Smoothie' and 'Iron Man's Power Steak.'", "source": "Catering Ideas"},
    {"text": "Decorate with iconic superhero logos and projections of Gotham and other superhero cities around the venue.", "source": "Decoration Ideas"},
    {"text": "Interactive experiences with VR where guests can engage in superhero simulations or compete in themed games.", "source": "Entertainment Ideas"}
]

source_docs = [Document(page_content=doc['text'], metadata={"source": doc['source']}) for doc in party_ideas]

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs_processed = text_splitter.split_documents(source_docs)

# Custom retriever tool
class PartyPlanningRetrieverTool(Tool):
    name = "party_planning_retriever"
    description = "Retrieves party planning ideas for Alfred's luxury superhero-themed party."

    inputs = {
        "query": {
            "type": "string",
            "description": "Party planning or superhero party query."
        }
    }
    output_type = "string"

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(docs, k=5)

    def forward(self, query: str) -> str:
        docs = self.retriever.invoke(query)
        return "\nRetrieved ideas:\n" + "".join(
            [f"\n\n===== Idea {str(i)} =====\n" + doc.page_content for i, doc in enumerate(docs)]
        )

# Instantiate retriever tool
party_planning_retriever = PartyPlanningRetrieverTool(docs_processed)
```

### 3. Initialize Agent & Run Query

```python
# Create the code agent
agent = CodeAgent(tools=[party_planning_retriever], model=HfApiModel())

# Example query
response = agent.run("Find ideas for a luxury superhero-themed party, including entertainment, catering, and decoration options.")

print(response)
```

---

## 🔥 How It Works

* **PartyPlanningRetrieverTool**: Performs **semantic search** over Alfred’s private knowledge base using BM25 ranking.
* **Web Retrieval** (Optional): For fresh ideas, DuckDuckGoSearchTool can be added to query live results.
* **CodeAgent**: Synthesizes and formats ideas into a cohesive plan using Hugging Face models.

---

## ✅ Example Use Cases

* 🦸‍♂️ **Superhero Party Planning**: Generate luxury event plans with custom themes.
* 🌐 **Live + Knowledge Base Hybrid Retrieval**: Mix up-to-date web results with curated knowledge.
* 📝 **Personalized Event Planning**: Store & reuse ideas for future Wayne Manor events.

---

## 📖 Learn More

This work builds on the Hugging Face Agents Course:
📓 [Retrieval Agents Notebook](https://huggingface.co/agents-course/notebooks/blob/main/unit2/smolagents/retrieval_agents.ipynb)
🎓 [Course Unit: Retrieval Agents](https://huggingface.co/learn/agents-course/unit2/smolagents/retrieval_agents)
