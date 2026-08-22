import os
from dotenv import load_dotenv
from retriever import load_competitor_df, df_to_documents, build_index
from agent import CompetitiveAnalysisAgent
from ui import build_ui

def main():
        # Load env load_dotenv()
        load_dotenv()
        cohere_key = os.getenv("COHERE_API_KEY")
        if not cohere_key:
            raise RuntimeError("COHERE_API_KEY not set. Create a .env with COHERE_API_KEY=...")

        # Build index
        csv_path = os.path.join("data", "competitors.csv")
        df = load_competitor_df(csv_path)
        docs = df_to_documents(df)
        index = build_index(docs, cohere_api_key=cohere_key)

        # Create agent
        agent = CompetitiveAnalysisAgent(index=index, cohere_api_key=cohere_key)

        # UI
        ui = build_ui(on_ask=agent.reason_and_act, on_history=lambda: agent.get_history(n=10))
        ui.launch()

if __name__ == "__main__":
        main()