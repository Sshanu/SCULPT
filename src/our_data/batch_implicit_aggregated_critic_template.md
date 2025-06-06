<|im_start|>system
# Task Overview  
Evaluate a set of prompts, predictions, and ground truths. Provide detailed feedback on each case and group related feedback into clusters based on common patterns or prompt references.

## Step-by-Step Instructions:

1. **Read the Input Prompt Thoroughly**:
   - Begin by carefully reading the entire input prompt along with its specific details. Make sure to understand the task at hand, including any requirements or constraints provided.

2. **General Feedback**:
    Provide comprehensive feedback on the input prompt to enhance its effectiveness in each of the following areas:
    * Contextual Errors: Identify specific inaccuracies or mistakes that may lead to misunderstandings.
    * Incorrect or Irrelevant Examples: Highlight any incorrect or misplaced examples within the prompt. Note that no section should contain more than 5-6 examples.
    * Gaps in Information: Point out any missing details or context that could clarify the task for the user, ensuring they have all necessary information.
    * Potential Improvements: Suggest ways to improve the prompt for better clarity and impact. This could include simplifying language, adding relevant examples, or outlining a clear sequence of steps. Ensure the prompt is efficient, concise, and free from redundant information.
    * Grammar and Syntax: Note any spelling or grammatical errors that could cause confusion, as well as poorly constructed sentences that may obscure the intended meaning.
    * Prompt Length: The prompt should be concise. Provide feedback around optimizing its length while maintaining clarity.
    * Other Issues: Identify any other areas where the prompt could be improved.

3. **Batch Feedback**:
   Based on `Batch Evaluations` where the model has generated wrong predictions, provide feedback to improve the performance of the prompt by following these steps:
    * Understanding the Context (prediction_explanation):
       - Start by clearly stating the input, expected output (ground_truth), and model’s actual prediction. Example format: `Input: '<input text>'`, `Expected Output: '<ground_truth>'`, `Prediction: '<prediction>'`.
       - Analyze why the model generated this prediction by identifying specific words, phrases, or contextual cues from the input.
       - Highlight the sections of the input or prompt that likely influenced the prediction using `prompt_references`.
    
    * Analysis and Feedback (`prompt_feedback`):
      The feedback should include the details about each of these steps:
      - `prediction_analysis`: Always include a clear analysis comparing the model's prediction with the expected output. Mention what the correct label should have been and highlight any discrepancies.
      - `prompt_examination`: Always analyze the prompt step-by-step, identifying specific sections that may have caused the error (e.g., unclear instructions, ambiguous wording). Explain how these issues led to the incorrect label.
      - `improvement_suggestions`: Provide multi-step feedback outlining all possible actions to address identified issues and explain how these changes will result in the correct label. Possible actions can include:
         - Rephrasing unclear instructions.
         - Removing redundancy.
         - Adding clarity or details.
         - Revising tone or structure for better flow.
         - Modifying examples: Remove bad examples, add or refine better examples, ensuring no section exceeds 5-6 examples.
    
    * Prompt references (`prompt_references`):
      Include references to the specific parts of the prompt that may have contributed to errors.
    
    * Cluster Feedback:
       - Group related feedback into {number_of_clusters} clusters based on patterns such as:
         - Shared sections of the prompt that influenced the predictions.
         - Expected output `ground_truth`.
         - Similar types of input data or prediction behavior.
       - Each cluster should include:
         - A list of explanations for the inputs in the cluster.
         - A specific list of feedback relevant to the cluster.
         - Clear `prompt_references` pointing to sections that could be revised or improved.

## Input Format:

**Example Prompt**

```json
{"<Heading 1>":{"body": "<body>","<Heading 1.1>":{"body": "<body>",...},"<Heading 1.2>":{"body": "<body>","Examples":["<example 1>",....],"<Heading 1.2.1>":{"body": "<body>","1.": {"body": "<instruction>","Examples":["<example 1>","<example 2>",.....]},"2.":...},"<Heading 1.2.2>":{"body": "<body>"}...}...},"<Heading 2>":{"body": "<body>"}...}
```

**Example Batch Evaluations**

```json
{"prompt": "The current prompt being used.","input_data": [{"id": "<unique id>","input": "<input text>","prediction": "<output generated by the model>","ground_truth": "<correct output>"},...]}
```

## Output Format:
The output must consists of a list of maximum {number_of_clusters} clusters, each identified by a unique `id`.

```json
[{"id": "general_feedback", "prompt_feedback": [{"prompt_examination":"<prompt_examination>", "improvement_suggestion": ["<improvement_suggestion>", ...]}, ...], "prompt_references": ["<prompt_reference>", ...]}, {"id": "cluster_1","prediction_explanation": ["<detailed explanation for example 1 in cluster 1>",...], "prompt_feedback": {"prediction_analysis": "<prediction_analysis>", "prompt_examination": "<prompt_examination>", "improvement_suggestions": ["<improvement_suggestions for cluster 1>", ...]}, "prompt_references": ["Heading 1> Heading 1.2> Heading 1.2.1> body","Heading 1> Heading 1.2> Heading 1.2.1> 2.> body","Heading 1> Heading 1.2> body","Heading 2> body"]},...]
```

## Constraints:
- Each section of the prompt should not contain more than 5-6 examples.
- Ensure the prompt is optimized in length without sacrificing clarity, with simplified language, relevant examples, and a clear sequence of steps. Avoid redundancy, and maintain efficiency and conciseness throughout. 
<|im_end|>
<|im_start|>user
**Input Prompt**

```json
{parsed_prompt}
```

**Batch Evaluations**

```json
{batch_evaluation}
```

## Prompt Optimization Reminders:
- Ensure Clarity and Conciseness: The prompt must be optimized in length without sacrificing clarity. Use simplified language for better understanding.
- Establish a Clear Sequence: The prompt should have a logical flow, outlining a clear sequence of steps for the task.
- Avoid Redundancy: Eliminate any redundant information to enhance efficiency and conciseness.
- Example Relevance: All examples included must be directly relevant to the prompt's objectives.

## Hard Constraints:
- Detailed Feedback: Provide comprehensive feedback in multiple steps, including suggestions for rephrasing, compacting information, ensuring relevance, and logical arrangement.
- Strict Example Limit: Each section of the prompt must contain no more than 5-6 examples. Adherence to this limit is mandatory to ensure clarity and maintain focus.
- `prompt_references` must have the very accurate with all the nested headings.
<|im_end|>
<|im_start|>assistant
```json