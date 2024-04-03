import argparse
from tot.methods.bfs import solve
from tot.tasks.swe import SWETask
from datasets import load_dataset

train_dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split = "test")

args = argparse.Namespace(
    backend='gpt-4', 
    temperature=0.1, 
    task='swe', 
    naive_run=False, 
    prompt_sample=None, 
    method_generate='propose', 
    method_evaluate='vote', 
    method_select='greedy', 
    n_generate_sample=1, 
    n_evaluate_sample=3, 
    n_select_sample=5)

task = SWETask(train_dataset)

ys, infos = solve(args, task, 1)
print(ys[0])
    