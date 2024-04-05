import re, os
from tot.tasks.base import Task
from tot.prompts.swe import *
from tot.models import gpt, groq

class SWETask(Task):
    """
    Input (x)   : a problem statement
    Output (y)  : a patch generation
    Reward (r)  : # TODO
    """
    def __init__(self, dataset):
        super().__init__()
        self.data = dataset
        self.steps = 2
        self.stops = ['\nPatch:\n', None]

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        return self.data[idx]["problem_statement"]
    
    def test_output(self, idx: int, output: str):
        output = output.split('Patch:\n')[-1]
        prompt = score_prompt + output
        api_base = os.getenv("OPENAI_API_BASE", "")
        if api_base == 'https://api.groq.com/openai/v1':
            score_output = groq(prompt, n=3, model='mixtral-8x7b-32768')
        else:
            score_outputs = gpt(prompt, n=5, model='gpt-4')
        scores = []
        for score_output in score_outputs:
            print("score_output: ",score_output)
            pattern = r".*correctness score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        print('------------')
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:
        prompt = vote_prompt
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list:
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best choice is .*(\d+).*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                if vote in range(n_candidates):
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Patch:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Patch: 1:\n{ys[0]}\n\nPatch: 2:\n{ys[1]}\n'
        return prompt
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):
        if 'more correct patch is 1' in compare_output:
            return 0
        elif 'more correct patch is 2' in compare_output:
            return 1
        elif 'two patches are equally correct' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
    
    @staticmethod
    def parse_diff_block(text: str):
        """Extracts the first block of unified diff format.

        Args:
            text (str): The large text containing one or more diff blocks.

        Returns:
            str: The first block between `diff and ` if found, otherwise None.
        """

        start_pattern = r"```diff"
        end_pattern = r"```"

        in_diff_block = False
        diff_block = []

        for line in text.splitlines():
            if start_pattern in line:  
                in_diff_block = True
                continue  # Skip the line with the start marker

            if in_diff_block:
                if end_pattern in line:
                    in_diff_block = False
                    break  # End of the diff block
                else:
                    diff_block.append(line)

        return "\n".join(diff_block) if diff_block else None
