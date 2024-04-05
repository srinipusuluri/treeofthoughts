standard_prompt = '''
{input}
'''

cot_prompt = '''I need you to solve this issue by generating a single patch file that I can apply directly to this repository using git apply. 
The patch file should be in the unified diff format. Example:

```diff
diff --git a/file.py b/file.py
--- a/file.py
+++ b/file.py
@@ -1,27 +1,35 @@
 def euclidean(a, b):
-    while b:
-        a, b = b, a % b
-    return a
+    if b == 0:
+        return a
+    return euclidean(b, a % b)
```

Given the problem statement of a github issue write a correct git patch to solve it.

Patch:
```diff
Your patch here.
```

The problem statement is the following: 
{input}
'''


vote_prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where {s} the integer id of the choice.
'''

compare_prompt = '''Briefly analyze the degree of correctness of the following two patches. Conclude in the last line "The more correct patch is 1", "The more correct patch is 2", or "The two patches are equally correct".
'''

score_prompt = '''Analyze the following patch, then at the last line conclude "Therefore the correctness score is {s}", where {s} is an integer from 1 to 10.
'''