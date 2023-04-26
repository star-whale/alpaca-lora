import os
from finetune import train
from generate import init_model, evaluate
from transformers import LlamaForCausalLM, LlamaTokenizer
from starwhale import evaluation, pass_context, Context, dataset
from starwhale.api import model, experiment
from starwhale.api.service import api
import gradio
import torch
from pathlib import Path

ROOTDIR = Path(__file__).parent
base_model = "decapoda-research/llama-7b-hf"


@evaluation.predict
def ppl(data: dict, **kw):
    instruction = data["instruction"]
    init_model(base_model=str(ROOTDIR/"models"), lora_weights=str(ROOTDIR/"lora-alpaca") if os.path.exists(ROOTDIR/"lora-alpaca") else "tloen/alpaca-lora-7b")
    return " ".join(evaluate(instruction=instruction))


@experiment.fine_tune()
def fine_tune() -> None:
    context = Context.get_runtime_context()
    ds_key_selectors = {
        "webqsp": {
            "rawquestion": "instruction",
            "parses[0].Answers[0].EntityName": "output",
        },
        "grailqav1": {"question":"instruction","answer[0].entity_name": "output"},
        "graph_questions_testing": {"question":"instruction","answer[0]": "output"},
        "z_bench_common": {"prompt": "instruction", "gpt4": "output"},
        "mkqa": {"query": "instruction", "answers.en[0].text": "output"},
    }
    sw_dataset = dataset(
        context.dataset_uris[0], readonly=True, create="forbid"
    )
    sw_dataset = sw_dataset.with_loader_config(
        field_transformer=ds_key_selectors.get(sw_dataset.name, None)
    )
    train(sw_train_dataset=sw_dataset, base_model="decapoda-research/llama-7b-hf", output_dir=str(ROOTDIR/"lora-alpaca"))
    build_package(ROOTDIR)


@api(gradio.Text(), gradio.Text())
def online_eval(question: str) -> str:
    return ppl({"instruction": question})


if not os.path.exists(ROOTDIR/"models"):
    hgmodel = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    tokenizer.save_pretrained(ROOTDIR/"models")
    hgmodel.save_pretrained(ROOTDIR/"models")


def build_package(ROOTDIR):
    model.build(
    workdir=ROOTDIR,
    name="llama-7b-hf",
    modules=[ppl, fine_tune, online_eval],
)

if __name__ == "__main__":
    build_package(ROOTDIR)
