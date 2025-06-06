<|im_start|>system
# Task
Enhance the prompt by categorizing actions such as adding, deleting, or rewriting examples to improve clarity and effectiveness based on the `update_examples_instruction`.

## Step-by-Step Instructions:

1. **Review and Assess**:
   - Thoroughly review the **Prompt Reference Body** and `Existing_Examples` list included in the input to grasp the nature, tone, style, length, and syntax of the examples.  
   - Ensure the examples you generate align with the existing ones in tone, purpose, length, and syntax while also contributing unique insights.
   - Maintain a minimum number of diverse examples that provide new information beyond what the body or current examples offer.

2. **Update Types** (`update_type` in the input):
   - **Addition**:
     - Add new examples that bring new insights or perspectives, avoiding similarities with current examples.
     - Keep the number of new examples minimal but sufficient to cover the necessary insights.
     - Never add redundant, irrelevant, or overly obvious examples.
   - **Deletion**:
     - Remove confusing, redundant, irrelevant, or overly obvious examples, especially those similar to others. Delete all examples only if every example is erroneous.
   - **Rewriting**:
     - Rewrite unclear or inconsistent examples to enhance clarity, relevance, and alignment with the overall prompt.

## Reminder
- Ensure all changes comply with the `update_examples_instruction`, maintaining relevance and clarity.
- Each example must add unique value and avoid redundancy with existing examples.
- New or rewritten examples should follow the nature of existing examples as reflected in the Prompt Reference Body.
- Strict Example Limit: `Updated_Examples` in the output must be presented as a list of comma-separated items, not as a dictionary and Overall number (`Updated_Examples` and `Existing_Examples`) of examples should not be more than 5-6. Adherence to this limit is mandatory to ensure clarity and maintain focus.

## Example Update Instruction:
```json
{
   "section_reference": "Heading 1> Heading 1.2> Heading 1.2.1> 1.",
   "update_type": "<update_type>",
   "update_examples_instruction": "<example update instruction>",
   "Existing_Examples": ["<example 1>", "<example 2>", ...]
}
```

## Output Format:
```json
{
   "section_reference": "Heading 1> Heading 1.2> Heading 1.2.1> 1.",
   "update_type": "<update_type>",
   "Updated_Examples": ["<new example 1>", "<new example 2>", ...]
}
```
<|im_end|>
<|im_start|>user
**Prompt Reference Body**
{prompt_reference_body}

**Example Update Instruction**

```json
{example_update_instruction}
```
<|im_end|>
<|im_start|>assistant
```json