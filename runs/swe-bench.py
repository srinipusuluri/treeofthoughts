import argparse
from tot.methods.bfs import solve
from tot.tasks.swe import SWETask
from datasets import load_dataset

print("Downloading dataset...")
train_dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split = "dev", cache_dir='datasets_cache')

args = argparse.Namespace(
    backend='mixtral-8x7b-32768',
    temperature=0.2, 
    task='swe', 
    naive_run=False,
    prompt_sample='cot', 
    method_generate='sample', 
    method_evaluate='vote', 
    method_select='greedy', 
    n_generate_sample=1, 
    n_evaluate_sample=3, 
    n_select_sample=5)

print("Solving...")
task = SWETask(train_dataset)
i = 10
ys, infos = solve(args, task, i, to_print=False)
print("Solution:")
print(SWETask.parse_diff_block(ys[0]))

print("Expected solution:")
print(train_dataset[i]["patch"])