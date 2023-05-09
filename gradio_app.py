import logging

import gradio as gr
import pandas as pd
from buster.busterbot import Buster

import cfg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# initialize buster with the config in cfg.py (adapt to your needs) ...
buster: Buster = Buster(cfg=cfg.buster_cfg, retriever=cfg.retriever)


def check_auth(username: str, password: str) -> bool:
    """Basic auth, only supports a single user."""
    # TODO: update to better auth
    is_auth = username == cfg.username and password == cfg.password
    logger.info(f"Log-in attempted. {is_auth=}")
    return is_auth


def format_sources(matched_documents: pd.DataFrame) -> str:
    if len(matched_documents) == 0:
        return ""

    sourced_answer_template: str = (
        """📝 Here are the sources I used to answer your question:<br>""" """{sources}<br><br>""" """{footnote}"""
    )
    source_template: str = """[🔗 {source.title}]({source.url}), relevance: {source.similarity:2.1f} %"""

    matched_documents.similarity = matched_documents.similarity * 100
    sources = "<br>".join([source_template.format(source=source) for _, source in matched_documents.iterrows()])
    footnote: str = "I'm a bot 🤖 and not always perfect."

    return sourced_answer_template.format(sources=sources, footnote=footnote)


def add_sources(history, response):
    documents_relevant = response.documents_relevant

    if documents_relevant:
        # add sources
        formatted_sources = format_sources(response.matched_documents)
        history.append([None, formatted_sources])

    return history


def user(user_input, history):
    """Adds user's question immediately to the chat."""
    return "", history + [[user_input, None]]


def chat(history):
    user_input = history[-1][0]

    response = buster.process_input(user_input)

    history[-1][1] = ""

    for token in response.completion.completor:
        history[-1][1] += token

        yield history, response


block = gr.Blocks(
    css="#chatbot .overflow-y-auto{height:500px}",
    theme=gr.themes.Default(primary_hue="violet", secondary_hue="fuchsia"),
)

with block:
    with gr.Row():
        gr.Markdown("<h3><center>Buster 🤖: A Question-Answering Bot for your documentation</center></h3>")

    chatbot = gr.Chatbot()

    with gr.Row():
        question = gr.Textbox(
            label="What's your question?",
            placeholder="Ask a question to AI stackoverflow here...",
            lines=1,
        )
        submit = gr.Button(value="Send", variant="secondary").style(full_width=False)

    examples = gr.Examples(
        examples=[
            "How can I run a job with 2 GPUs?",
            "What kind of GPUs are available on the cluster?",
            "What is the $SCRATCH drive for?",
        ],
        inputs=question,
    )

    gr.Markdown("This application uses GPT to search the docs for relevant info and answer questions.")

    gr.HTML("️<center> Created with ❤️ by @jerpint and @hadrienbertrand")

    response = gr.State()

    submit.click(user, [question, chatbot], [question, chatbot], queue=False).then(
        chat, inputs=[chatbot], outputs=[chatbot, response]
    ).then(add_sources, inputs=[chatbot, response], outputs=[chatbot])
    question.submit(user, [question, chatbot], [question, chatbot], queue=False).then(
        chat, inputs=[chatbot], outputs=[chatbot, response]
    ).then(add_sources, inputs=[chatbot, response], outputs=[chatbot])

block.queue(concurrency_count=16)
block.launch(debug=True, share=False, auth=check_auth, auth_message="Request access from an admin.")
