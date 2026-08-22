"""
Multi-Agent Content Generation System
Author: AI Solutions Architect (ChatGPT)
Description:
  - Implements a multi-agent content generation pipeline using a LangGraph-style workflow and CrewAI-style crew/task orchestration.
  - This is an opinionated, runnable reference implementation that uses:
      * a simple LangGraph-like directed graph implemented with classes (so you can map to real LangGraph SDK easily)
      * a Crew/CrewMember/Task abstraction resembling CrewAI to define agents & tasks
      * integrations (placeholder wrappers) for OpenAI, SerpAPI, and keyword tools
  - The system: TopicResearcher -> ContentWriter -> SEOOptimizer -> (optional) Editor -> OutputWriter

Notes:
  - This file is intentionally self-contained and simulative. Replace placeholder API calls with your own credentials and SDK calls.
  - If you want a direct mapping to the real LangGraph and CrewAI SDKs, I included clear comments where to swap in real API calls.

How to use (quick):
  1. Install dependencies you need, e.g. pip install openai requests python-dotenv
  2. Set environment variables: OPENAI_API_KEY, SERPAPI_KEY (optional), YOAST_API_KEY (optional)
  3. Run: python multi_agent_langgraph_crewai_system.py
  4. Output file will be written to ./outputs/<topic_slug>_final.txt

This implementation focuses on clarity and handoffs (data format between agents). It writes logs and outputs.

"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Callable, Optional
import time
import json
import os
import re
from pathlib import Path
import hashlib
import textwrap

# --- Helpers -----------------------------------------------------------------

def slugify(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s[:60]


def now_ts() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


# --- Agent definitions -------------------------------------------------------

@dataclass
class AgentSpec:
    role: str
    goal: str
    backstory: str
    tools: List[str] = field(default_factory=list)
    name: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            self.name = self.role.replace(" ", "_").lower()


# --- Simple LangGraph-like Graph Implementation -------------------------------

class LGNode:
    def __init__(self, id: str, func: Callable[[Dict[str, Any]], Dict[str, Any]], spec: AgentSpec):
        self.id = id
        self.func = func
        self.spec = spec
        self.out_edges: List[str] = []

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"[LGNode] Running node {self.id} (agent: {self.spec.name})")
        return self.func(data)


class LangGraph:
    def __init__(self):
        self.nodes: Dict[str, LGNode] = {}
        self.edges: Dict[str, List[str]] = {}

    def add_node(self, node: LGNode):
        self.nodes[node.id] = node
        self.edges[node.id] = []

    def add_edge(self, src: str, dst: str):
        if src not in self.nodes or dst not in self.nodes:
            raise KeyError("src/dst node not found")
        self.edges[src].append(dst)
        self.nodes[src].out_edges.append(dst)

    def run_from(self, start_id: str, initial_data: Dict[str, Any]) -> Dict[str, Any]:
        # Simple BFS execution following edges in order; for more complex workflows use real LangGraph
        visited = set()
        results = {}
        queue = [start_id]
        data_for_node = {start_id: initial_data}

        while queue:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            node = self.nodes[node_id]
            input_data = data_for_node.get(node_id, {})
            out = node.run(input_data)
            results[node_id] = out
            # propagate to successors
            for succ in self.edges.get(node_id, []):
                # merge handoff: succ receives previous output under 'handoff_from_<node_id>' key
                merged = data_for_node.get(succ, {}).copy() if data_for_node.get(succ) else {}
                merged[f"handoff_from_{node_id}"] = out
                # also preserve a 'latest' pointer
                merged['latest'] = out
                data_for_node[succ] = merged
                queue.append(succ)
        return results


# --- CrewAI-like abstractions ------------------------------------------------

@dataclass
class Task:
    id: str
    description: str
    input_keys: List[str]
    output_keys: List[str]
    execute: Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass
class CrewMember:
    id: str
    spec: AgentSpec
    tasks: List[Task] = field(default_factory=list)

    def perform(self, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        matching = [t for t in self.tasks if t.id == task_id]
        if not matching:
            raise KeyError(f"Task {task_id} not found for crew member {self.id}")
        task = matching[0]
        print(f"[CrewMember] {self.id} performing task {task.id}")
        return task.execute(context)


@dataclass
class Crew:
    id: str
    members: List[CrewMember]

    def assign(self, member_id: str, task_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        mem = next((m for m in self.members if m.id == member_id), None)
        if not mem:
            raise KeyError(f"Crew member {member_id} not found")
        return mem.perform(task_id, context)


# --- Tool integrations (placeholders / wrappers) ------------------------------

# NOTE: Replace these with real SDK calls (openai, serpapi, requests to yoast, etc.)

class OpenAIClient:
    def __init__(self, api_key: Optional[str] = None):
        # placeholder wrapper; user should configure openai.api_key or use official SDK
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        # Minimal safe fallback for environments lacking OpenAI credentials: echo simplified output
        if not self.api_key:
            print("[OpenAIClient] WARNING: No API key found. Returning simulated text.")
            return f"SIMULATED OUTPUT FOR PROMPT START:\n{prompt[:300]}\n... (no API key)"
        try:
            import openai
            openai.api_key = self.api_key
            resp = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                n=1,
            )
            return resp.choices[0].text.strip()
        except Exception as e:
            print(f"[OpenAIClient] error calling OpenAI: {e}")
            return f"ERROR: {e}"


class SerpAPIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.key = api_key or os.getenv('SERPAPI_KEY')

    def search(self, query: str, num: int = 10) -> List[Dict[str, Any]]:
        if not self.key:
            print("[SerpAPIClient] WARNING: No SerpAPI key provided. Returning simulated search results.")
            return [{'title': f'Simulated result for {query}', 'snippet': 'An example snippet', 'link': 'https://example.com'}]
        try:
            import requests
            params = {"q": query, "api_key": self.key}
            resp = requests.get("https://serpapi.example/search", params=params, timeout=10)
            return resp.json().get('organic_results', [])
        except Exception as e:
            print(f"[SerpAPIClient] error: {e}")
            return []


class KeywordToolClient:
    def analyze(self, seed_terms: List[str]) -> Dict[str, Any]:
        # Simulate keyword analysis. In real life call Ahrefs/SEMrush/API.
        out = {}
        for t in seed_terms:
            out[t] = {
                'search_volume': 1000 + len(t) * 10,
                'difficulty': max(1, min(80, len(t) * 3)),
                'suggested_variations': [t + ' guide', t + ' tips', t + ' 2025']
            }
        return out


class SEOAnalyzer:
    def optimize(self, content: str, target_keywords: List[str]) -> Dict[str, Any]:
        # Very small heuristic optimizer: ensures keywords appear in title, meta, and body appropriately
        title = content.split('\n')[0][:70]
        meta = (content[:160].replace('\n', ' '))[:155]
        # count keywords
        counts = {k: len(re.findall(re.escape(k), content, flags=re.IGNORECASE)) for k in target_keywords}
        suggestions = []
        for k, c in counts.items():
            if c == 0:
                suggestions.append(f"Add '{k}' once in intro and once in conclusion.")
            elif c < 2:
                suggestions.append(f"Consider using '{k}' one more time.")
        score = max(0, 100 - sum(5 for v in counts.values() if v == 0))
        return {'title': title, 'meta': meta, 'keyword_counts': counts, 'suggestions': suggestions, 'score': score}


# --- Agent Implementations (node functions) ----------------------------------

# Shared clients (replace with DI in production)
OPENAI = OpenAIClient()
SERPAPI = SerpAPIClient()
KWTOOL = KeywordToolClient()
SEOTOOL = SEOAnalyzer()


# Node: Topic Researcher

def topic_researcher_node(in_data: Dict[str, Any]) -> Dict[str, Any]:
    # expects: { 'topic': str, 'client_brief': Optional[str] }
    topic = in_data.get('topic') or in_data.get('client_brief') or 'AI in Marketing'
    print(f"[TopicResearcher] Researching topic: {topic}")

    # 1) Run web search to collect top sources
    search_q = f"{topic} latest trends 2025"
    search_results = SERPAPI.search(search_q, num=10)

    # 2) Generate seed keywords via OpenAI summarization of search results (or keyword tool)
    seed_prompt = f"Given the following topic: '{topic}', suggest 10 high-value seed keywords and short intent labels:\nSearch results snippets:\n"
    for r in (search_results or [])[:5]:
        seed_prompt += f"- {r.get('title')} - {r.get('snippet')}\n"

    ai_resp = OPENAI.generate(seed_prompt, max_tokens=300)
    # parse simple lines from AI response
    seed_keywords = []
    for line in ai_resp.split('\n'):
        m = re.search(r"([A-Za-z0-9\s\-']{3,})", line)
        if m:
            kw = m.group(1).strip().lower()
            if kw and len(kw) > 2:
                seed_keywords.append(kw)
    seed_keywords = seed_keywords[:10] if seed_keywords else [topic.lower(), topic.lower() + ' trends']

    # 3) Use keyword tool to analyze
    kw_analysis = KWTOOL.analyze(seed_keywords)

    out = {
        'topic': topic,
        'search_query': search_q,
        'search_results': search_results,
        'seed_keywords': seed_keywords,
        'kw_analysis': kw_analysis,
        'research_ts': now_ts()
    }
    return out


# Node: Content Writer

def content_writer_node(in_data: Dict[str, Any]) -> Dict[str, Any]:
    # expects handoff_from_topic_researcher
    research = in_data.get('handoff_from_topic_researcher') or in_data.get('latest') or {}
    topic = research.get('topic', 'AI in Marketing')
    seed_keywords = research.get('seed_keywords', [topic])
    top_keywords = seed_keywords[:5]

    brief = in_data.get('client_brief') or f"Write an authoritative, 900-1200 word blog post about {topic}."
    prompt = textwrap.dedent(f"""
    You are a professional content writer. Produce a blog post draft with the following constraints:
    - Topic: {topic}
    - Target keywords: {', '.join(top_keywords)}
    - Voice: helpful, professional, approachable
    - Structure: Title, short meta description (max 155 chars), H1, intro, 3-6 subheadings, conclusion, CTA.
    - Include suggested internal link ideas and suggested images (descriptions).

    Research context (summaries):
    {json.dumps(research.get('search_results', [])[:5], indent=2) if research.get('search_results') else 'No sources provided.'}

    Draft:
    """)

    generated = OPENAI.generate(prompt, max_tokens=900)

    out = {
        'topic': topic,
        'seed_keywords': seed_keywords,
        'target_keywords': top_keywords,
        'draft': generated,
        'writer_ts': now_ts()
    }
    return out


# Node: SEO Optimizer

def seo_optimizer_node(in_data: Dict[str, Any]) -> Dict[str, Any]:
    # expects handoff_from_content_writer
    writer_out = in_data.get('handoff_from_content_writer') or in_data.get('latest') or {}
    content = writer_out.get('draft', '')
    keywords = writer_out.get('target_keywords', [])

    print(f"[SEOOptimizer] Optimizing content for keywords: {keywords}")
    seo_report = SEOTOOL.optimize(content, keywords)

    # optionally request OpenAI to rewrite title/meta
    title_prompt = f"Given the following draft, produce a catchy SEO title (<=70 chars) and meta description (<=155 chars):\n\n{content[:1500]}"
    seo_ai = OPENAI.generate(title_prompt, max_tokens=120)

    out = {
        'optimized': {
            'seo_report': seo_report,
            'ai_title_meta': seo_ai,
        },
        'optimized_content': content,
        'optimizer_ts': now_ts()
    }
    return out


# Node: Editor (optional manual or automated)

def editor_node(in_data: Dict[str, Any]) -> Dict[str, Any]:
    # expects handoff_from_seo_optimizer
    opt = in_data.get('handoff_from_seo_optimizer') or in_data.get('latest') or {}
    content = opt.get('optimized_content', '')
    seo = opt.get('optimized', {})

    # Simple automated edit: shorten paragraphs > 200 chars and fix passive voice prompt
    edit_prompt = f"Edit & proofread the following article. Make it clearer, fix grammar, keep SEO keywords, and improve readability. Output only the final article.\n\n{content[:4000]}"
    edited = OPENAI.generate(edit_prompt, max_tokens=900)

    out = {
        'edited_content': edited,
        'editor_ts': now_ts(),
        'seo': seo
    }
    return out


# Node: Output Writer

def output_writer_node(in_data: Dict[str, Any]) -> Dict[str, Any]:
    # expects handoff_from_editor or handoff_from_seo_optimizer
    source = in_data.get('handoff_from_editor') or in_data.get('handoff_from_seo_optimizer') or in_data.get('latest') or {}
    # Try several fields
    content = source.get('edited_content') or source.get('optimized_content') or source.get('draft') or ''
    topic = source.get('topic') or in_data.get('topic') or 'content'
    keywords = source.get('target_keywords') or []

    out_dir = Path('outputs')
    out_dir.mkdir(exist_ok=True)
    fname = f"{slugify(topic)}_{now_ts()}.txt"
    path = out_dir / fname

    meta = {
        'topic': topic,
        'keywords': keywords,
        'generated_at': now_ts()
    }

    with open(path, 'w', encoding='utf-8') as f:
        f.write(json.dumps(meta) + '\n\n')
        f.write(content)

    print(f"[OutputWriter] Wrote output to {path}")
    return {'path': str(path), 'meta': meta}


# --- System Assembly: Define agents, LangGraph nodes, Crew tasks ----------------

# Agent specs
TOPIC_RESEARCHER = AgentSpec(
    role='Topic Researcher',
    goal='Discover trending topics, gather sources, and produce seed keywords.',
    backstory='Experienced market researcher who scrapes the web for signals and keywords.',
    tools=['SerpAPI', 'OpenAI', 'KeywordTool']
)

CONTENT_WRITER = AgentSpec(
    role='Content Writer',
    goal='Draft high-quality, audience-appropriate content using research.',
    backstory='Professional content writer with SEO awareness and structure-first approach.',
    tools=['OpenAI']
)

SEO_OPTIMIZER = AgentSpec(
    role='SEO Optimizer',
    goal='Enhance content for discoverability, meta tags and keyword placement.',
    backstory='SEO specialist who translates keywords into on-page signals.',
    tools=['SEOTool', 'OpenAI']
)

EDITOR = AgentSpec(
    role='Editor',
    goal='Proofread and polish the content to publication quality.',
    backstory='Editor focusing on clarity, voice, and brand guidelines.',
    tools=['OpenAI']
)

OUTPUT_WRITER = AgentSpec(
    role='Output Writer',
    goal='Persist content to storage and produce CMS-ready artifacts.',
    backstory='Ops specialist who prepares content files and metadata for publishing.',
    tools=['Filesystem']
)

# Create LangGraph nodes
lg = LangGraph()
lg.add_node(LGNode('topic_researcher', topic_researcher_node, TOPIC_RESEARCHER))
lg.add_node(LGNode('content_writer', content_writer_node, CONTENT_WRITER))
lg.add_node(LGNode('seo_optimizer', seo_optimizer_node, SEO_OPTIMIZER))
lg.add_node(LGNode('editor', editor_node, EDITOR))
lg.add_node(LGNode('output_writer', output_writer_node, OUTPUT_WRITER))

# Edges / transitions: topical flow
lg.add_edge('topic_researcher', 'content_writer')
lg.add_edge('content_writer', 'seo_optimizer')
lg.add_edge('seo_optimizer', 'editor')
lg.add_edge('editor', 'output_writer')

# CrewAI-like setup
# Tasks mapped to nodes
research_task = Task(
    id='research_task',
    description='Perform topic research and generate seed keywords and sources.',
    input_keys=['topic', 'client_brief'],
    output_keys=['seed_keywords', 'search_results'],
    execute=lambda ctx: topic_researcher_node(ctx)
)

write_task = Task(
    id='write_task',
    description='Write draft based on research.',
    input_keys=['handoff_from_topic_researcher', 'client_brief'],
    output_keys=['draft'],
    execute=lambda ctx: content_writer_node(ctx)
)

seo_task = Task(
    id='seo_task',
    description='Optimize draft for SEO.',
    input_keys=['handoff_from_content_writer'],
    output_keys=['optimized_content', 'seo_report'],
    execute=lambda ctx: seo_optimizer_node(ctx)
)

edit_task = Task(
    id='edit_task',
    description='Edit and proofread the optimized draft.',
    input_keys=['handoff_from_seo_optimizer'],
    output_keys=['edited_content'],
    execute=lambda ctx: editor_node(ctx)
)

output_task = Task(
    id='output_task',
    description='Write final file to disk and return path.',
    input_keys=['handoff_from_editor'],
    output_keys=['path'],
    execute=lambda ctx: output_writer_node(ctx)
)

# Crew members
crew_members = [
    CrewMember(id='topic_researcher', spec=TOPIC_RESEARCHER, tasks=[research_task]),
    CrewMember(id='content_writer', spec=CONTENT_WRITER, tasks=[write_task]),
    CrewMember(id='seo_optimizer', spec=SEO_OPTIMIZER, tasks=[seo_task]),
    CrewMember(id='editor', spec=EDITOR, tasks=[edit_task]),
    CrewMember(id='output_writer', spec=OUTPUT_WRITER, tasks=[output_task]),
]

crew = Crew(id='innovate_marketing_crew', members=crew_members)


# --- High-level Orchestrator --------------------------------------------------

class Orchestrator:
    def __init__(self, graph: LangGraph, crew: Crew):
        self.graph = graph
        self.crew = crew

    def run_topic_pipeline(self, topic: str, client_brief: Optional[str] = None) -> Dict[str, Any]:
        initial = {'topic': topic, 'client_brief': client_brief}
        # Run LangGraph from topic_researcher
        lg_results = self.graph.run_from('topic_researcher', initial)

        # For each node output, optionally invoke Crew assignments (simulating CrewAI)
        # 1) Research already executed as node; we can call crew to confirm
        research_result = lg_results.get('topic_researcher')
        crew_output_research = self.crew.assign('topic_researcher', 'research_task', {**initial})

        # Handoffs: build context for writer
        writer_context = {'handoff_from_topic_researcher': research_result, 'client_brief': client_brief}
        crew_output_write = self.crew.assign('content_writer', 'write_task', writer_context)

        seo_context = {'handoff_from_content_writer': crew_output_write}
        crew_output_seo = self.crew.assign('seo_optimizer', 'seo_task', seo_context)

        edit_context = {'handoff_from_seo_optimizer': crew_output_seo}
        crew_output_edit = self.crew.assign('editor', 'edit_task', edit_context)

        out_context = {'handoff_from_editor': crew_output_edit, 'topic': topic}
        crew_output_final = self.crew.assign('output_writer', 'output_task', out_context)

        return {
            'research': research_result,
            'draft': crew_output_write,
            'seo': crew_output_seo,
            'edited': crew_output_edit,
            'output': crew_output_final
        }


# --- Example execution (main) -------------------------------------------------

def main_example():
    print("Starting multi-agent pipeline (simulation).")
    orchestrator = Orchestrator(lg, crew)
    topic = "AI in Marketing"  # sample; replace with user input
    client_brief = "Create a long-form blog post aimed at CMOs about adopting AI in digital marketing."
    result = orchestrator.run_topic_pipeline(topic, client_brief)
    print('\n--- Pipeline completed ---')
    print(json.dumps({k: (v if isinstance(v, dict) else str(v)) for k,v in result.items()}, indent=2, default=str))
    print("Final output path:", result['output'].get('path'))


if __name__ == '__main__':
    main_example()

# --- End of file --------------------------------------------------------------
