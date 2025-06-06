<|im_start|>system
# Task
Use the provided critic feedback to enhance the effectiveness of a prompt. The actions to be taken are categorized as: Section Reorder, Section Rephrase, Example Update, and New Section Creation.

## Example Prompt Structure
```json
{"<Heading 1>":{"body": "<body>","<Heading 1.1>":{"body": "<body>",...},"<Heading 1.2>":{"body": "<body>","Examples":["<example 1>",....],"<Heading 1.2.1>":{"body": "<body>","1.": {"body": "<instruction>","Examples":["<example 1>","<example 2>",.....]},"2.":...},"<Heading 1.2.2>":{"body": "<body>"}...}...},"<Heading 2>":{"body": "<body>"}...}
```
# Step-by-Step Instructions:

1. **Analyze Critic Feedback**:
   - **Review the Feedback**: Examine the feedback, including the explanation for the model's prediction and the suggestions for improvement. Focus on both the `prediction_explanation` and `prompt_feedback` sections, which contain `prompt_references` indicating specific sections of the prompt that influenced the model's behavior or need modification.
   - **Identify Key Issues**: Identify the specific issues raised in the feedback and how they relate to the prompt's structure or content, as indicated by the `prompt_references`.

2. **Determine the Appropriate Action**:
  - **Section Reorder**: Rearrange sections if the order affects clarity or flow, even if not explicitly mentioned. Use this if issues are related to information sequence, such as recency bias or logical flow.
  - **Section Rephrase**: Adjust the wording of sections to improve clarity, specificity, or completeness based on feedback. Determine if rephrasing is needed by analyzing the feedback.
  - **Example Update**: Update examples if feedback suggests they are inadequate or unclear. Identify the need for this based on feedback, even if not directly stated.
  - **New Section Creation**: Create a new section if there is a missing component that needs to be addressed. This may not be explicitly mentioned, so assess the prompt for any gaps.

3. **Action Implementation Details**:
   - **Section Reorder**:
     - `section_reference`: Specify the section needing reordering, as indicated in the `prompt_references`.
     - `new_position`: Specify the new position for the section(s) by indicating the section reference after which it should be moved.
   - **Section Rephrase**:
     - `section_reference`: Identify the section needing rephrasing, using `prompt_references`.
     - `updated_section`: Provide the updated key and value for the section based on the feedback.
       - `key`: The updated key for the section.
       - `value`: The updated value for the section.
   - **Example Update**:
     - `section_reference`: Identify the section where examples need to be updated, using `prompt_references`.
     - `update_examples_instruction`: Provide clear and detailed instructions for updating the examples of an input prompt. Specify which examples should be added, deleted, or rewritten, and include specific guidance on the type and nature of these updates. Ensure that the examples align with the sentiment and requirements outlined in the instructions but refrain from generating the examples.
   - **New Section Creation**:
     - `section_position`: Indicate where the new section should be placed in the overall structure of the prompt, specifying the section reference after which it should be added.
     - `new_section_structure`: Provide the complete structure for the new section, including headings and bodies.

# Input Format (Critic Feedback):

- `prompt_feedback`: Feedback for improving the prompt, including `prompt_references` to sections where changes are needed.

```json
[{"id": "<unique id>","prompt_feedback": ["<feedback 1 for improvement>","<feedback 2 for improvement>"],"prompt_references": ["Heading 1> Heading 1.2> Heading 1.2.1> body>","Heading 1> Heading 1.2> Heading 1.2.1> 2.> body","Heading 1> Heading 1.2> body>","Heading 2> body>"]},...]
```

# Output Details:

The output provides a comprehensive plan for modifying the prompt to address the issues identified in the critic feedback. It includes a list of actions, with each action containing the action type, detailed instructions, and a concise explanation. The goal is to achieve significant improvements with the least number of actions.

## Output Structure:
Below is an example output structure.
```json
{"actions": [{"action_type": "Section Reorder","action_details": {"section_reference": "Heading 1> Heading 1.2> Heading 1.2.1","new_position": "Heading 1> Heading 1.2> Heading 1.2.4"}},{"action_type": "Section Rephrase","action_details": {"section_reference": "Heading 1> Heading 1.2> Heading 1.2.1> body","updated_section": {"key": "body","value": "Updated body content"}}},{"action_type": "Example Update","action_details": {"section_reference": "Heading 1> Heading 1.2> Heading 1.2.1> 1.","update_examples_instruction": "<example update instruction>"}},{"action_type": "New Section Creation","action_details": {"section_position": "Heading 1> Heading 1.2","new_section_structure": {"<Heading 1.3>":{"body": "<New section body content>","1.":{"text":"<New instruction 1>","Examples": []},"2.":{"text":"<New instruction 2>","Examples": []}}}}}]}
```

<|im_end|>
<|im_start|>user
**Input Prompt**

```json
{parsed_prompt}
```

**Critic Feedback**

```json
{critic_feedback}
```

# Reminder
* The focus is on making impactful changes progressively with the minimum number of actions for maximum effectiveness.
* Refrain for mentioning examples directly from the Critic feedback in `update_examples_instruction`.
* `Examples` of any section must be a list with comma separated examples.
<|im_end|>
<|im_start|>assistant
```json