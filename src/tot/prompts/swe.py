standard_prompt = '''
Given the problem statement of a github issue write a correct git patch to solve it. The problem statement is the following: {input}
'''

cot_prompt = '''
Given the problem statement of a github issue write a correct git patch to solve it.
Make a plan then write. Your output should be of the following format:

Plan:
Your plan here.

Patch:
Your patch here.

The problem statement is the following: {input}
'''


vote_prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where {s} the integer id of the choice.
'''

compare_prompt = '''Briefly analyze the degree of correctness of the following two patches. Conclude in the last line "The more correct patch is 1", "The more correct patch is 2", or "The two patches are equally correct".
'''

score_prompt = '''Analyze the following patch, then at the last line conclude "Therefore the correctness score is {s}", where s is an integer from 1 to 10.
'''