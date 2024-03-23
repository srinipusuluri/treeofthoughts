import argparse
from tot.methods.bfs import solve
from tot.tasks.swe import SWETask

args = argparse.Namespace(backend='gpt-4', 
                          temperature=0.7, 
                          task='swe', 
                          naive_run=False, 
                          prompt_sample=None, 
                          method_generate='propose', 
                          method_evaluate='vote', 
                          method_select='greedy', 
                          n_generate_sample=1, 
                          n_evaluate_sample=3, 
                          n_select_sample=5)

task = SWETask()
ys, infos = solve(args=args, task=task, idx=1)
print(ys[0])