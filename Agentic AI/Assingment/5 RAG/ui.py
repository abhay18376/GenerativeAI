import gradio as gr
from typing import Callable, List, Tuple

def build_ui(on_ask: Callable[[str], str], on_history: Callable[[], List[Tuple[str,str]]]):
    with gr.Blocks(title="Competitive Analysis Agent") as demo:
        gr.Markdown("## 🧠 Agentic RAG: Competitive Analysis\nAsk about competitors, marketing strategies, SWOT, comparisons, and more.")
        with gr.Row():
            query = gr.Textbox(label="Your query", placeholder="e.g., Compare the marketing strategies of AlphaTech and BetaSense")
        with gr.Row():
            ask_btn = gr.Button("Ask")
            history_btn = gr.Button("Show Recent History")
            clear_btn = gr.Button("Clear")

        answer = gr.Textbox(label="Answer", lines=12)
        history = gr.Dataframe(headers=["User", "Assistant"], row_count=(5, "dynamic"))

        def _ask(q):
            if not q.strip():
                return "Please enter a question."
            return on_ask(q)

        def _show_history():
            return on_history()

        ask_btn.click(_ask, inputs=[query], outputs=[answer])
        history_btn.click(_show_history, outputs=[history])
        clear_btn.click(lambda: ("", []), outputs=[answer, history])
    return demo
    
    def build_ui(on_ask: Callable[[str], str], on_history: Callable[[], List[Tuple[str,str]]]):
        with gr.Blocks(title="Competitive Analysis Agent") as demo:
            gr.Markdown("## 🧠 Agentic RAG: Competitive Analysis\nAsk about competitors, marketing strategies, SWOT, comparisons, and more.")
            with gr.Row():
                query = gr.Textbox(label="Your query", placeholder="e.g., Compare the marketing strategies of AlphaTech and BetaSense")
            with gr.Row():
                ask_btn = gr.Button("Ask")
                history_btn = gr.Button("Show Recent History")
                clear_btn = gr.Button("Clear")

            answer = gr.Textbox(label="Answer", lines=12)
            history = gr.Dataframe(headers=["User", "Assistant"], row_count=(5, "dynamic"))

            def _ask(q):
                if not q.strip():
                    return "Please enter a question."
                return on_ask(q)

            def _show_history():
                return on_history()

            ask_btn.click(_ask, inputs=[query], outputs=[answer])
            history_btn.click(_show_history, outputs=[history])
            clear_btn.click(lambda: ("", []), outputs=[answer, history])
        return demo