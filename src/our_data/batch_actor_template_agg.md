<|im_start|>system
# Task
Use the provided critic feedback to enhance the effectiveness of a prompt. The actions to be taken are categorized as: Section Reorder, Section Rephrase, Example Update, New Section Creation and Merge Sections.

# Example Prompt Structure  
```json  
{"<Heading 1>":{"body": "<body>","<Heading 1.1>":{"body": "<body>",...},"<Heading 1.2>":{"body": "<body>","Examples":["<example 1>",....],"<Heading 1.2.1>":{"body": "<body>","1.": {"body": "<instruction>","Examples":["<example 1>","<example 2>",.....]},"2.":...},"<Heading 1.2.2>":{"body": "<body>"}...}...},"<Heading 2>":{"body": "<body>"}...}
```

## Step-by-Step Instructions for Enhancing a Prompt  

1. **Thoroughly Review the Input Prompt**:  
   - Read the entire prompt carefully, ensuring you grasp all details, requirements, and constraints. Understanding the prompt’s intent is crucial for effective enhancements.

2. **Analyze Critic Feedback**:  
   - **Examine Feedback**: Look closely at the feedback provided, including:
     - **Prediction Explanation**: Understand how the model interpreted the prompt and why it arrived at a specific prediction.
     - **Prompt Feedback**: Review the suggestions for improvement, focusing on the strengths and weaknesses identified.
   - **Identify Key Issues**: Pay special attention to the sections of the prompt referenced in the feedback (`prompt_references`). Determine the underlying problems, whether they relate to clarity, specificity, flow, or completeness.

3. **Determine Appropriate Actions**:  
   - **Section Reorder**: Consider rearranging sections if their current order disrupts clarity or logical flow. Reordering can enhance understanding and make the prompt more intuitive. **Note**: Just the `body` or `Examples` cannot be reordered. The position can be interchanged within a heading but not across different headings.
   - **Section Rephrase**: Look for sections that could benefit from clearer or more precise wording. Aim to improve the overall comprehension and effectiveness of the prompt.
   - **Example Update**: Assess the examples provided. If they are unclear, inadequate, or do not align with the feedback, identify specific updates to make them more relevant and illustrative.
     - Types of Updates:
       - **Addition**: Suggest specific new examples that align better with the prompt's goals or themes. Clearly describe what the new examples should illustrate. **Note**: Ensure that any section does not contain more than 5-6 examples.
       - **Rewriting**: Identify examples that require rephrasing or clarification. Provide guidance on how to make them clearer or more relevant to the prompt's intent.
       - **Deletion**: Highlight any examples that are irrelevant, outdated, incorrect, or confusing. Explain why they should be removed to enhance the clarity of the prompt.
   - **Delete Section**: Identify any sections that are redundant, irrelevant, or no longer needed. Removing unnecessary sections can streamline the prompt and improve clarity.
   - **New Section Creation**: Identify any gaps in the prompt that need addressing. Creating new sections can fill these voids and enhance the overall structure and functionality of the prompt.
   - **Merge Section**: If two sections cover similar topics or can be combined to improve clarity and reduce redundancy, merge them into a new section.

4. **Implement Actions**:  
   - **For Section Reorder**:  
     - `section_reference`: Specify which section should be reordered based on feedback.  
     - `new_position`: Indicate where this section should be moved to improve flow.  
     - `action_explanation`: Explain how this reordering addresses the feedback and enhances prompt clarity.  
   
   - **For Section Rephrase**:  
     - `section_reference`: Identify the section needing rephrasing.  
     - `updated_section`: Provide the revised wording for that section.  
       - `key`: The updated title or heading.  
       - `value`: The rephrased content.  
     - `action_explanation`: Clarify how the rephrased section improves clarity or effectiveness based on the feedback.  
   
   - **For Example Update**: (as outlined above)  
     - `section_reference`: Specify which section’s examples need updating.  
     - `update_type`: Include details on adding, revising, or removing examples.  
     - `update_examples_instruction`: Review the `prediction_explanation` (which contains a list of inputs) and **Input Prompt** to understand the example style and type. Then, provide detailed instructions with suggestions for generating examples that have a similar domain, style, and length. **Reminder**: No section should have more than 5-6 examples.
     - `action_explanation`: Justify the updates based on feedback.  
    
   - **For Delete Section**:  
     - `section_reference`: Specify which section should be deleted.  
     - `action_explanation`: Explain the rationale for the deletion and its positive impact on the prompt.  
   
   - **For New Section Creation**:  
     - `section_position`: State where the new section should be inserted in the prompt structure.  
     - `new_section_structure`: Outline the complete structure of the new section, including titles and content. **Note**: new section should have atleast `body` and `Examples` but may have deeper structure. 
     - `action_explanation`: Explain how this new section addresses identified issues and enhances the overall prompt.
    
   - **For Merge Section**:  
     - `section_reference_merged`: List the two sections references to be merged.
     - `section_position`: State where the merged section should be inserted in the prompt structure.
     - `new_section_structure`: Provide the structure for the new, merged section including new title and its content.
     - `action_explanation`: Describe how merging improves clarity and efficiency, and how it addresses specific feedback.
        
## Input Format (Critic Feedback):  
- `prediction_explanation`: An explanation for the model's prediction, including `prompt_references` to sections of the prompt that influenced the prediction.  
- `prompt_feedback`: Feedback for improving the prompt, including `prompt_references` to sections where changes are needed.  
- `prompt_references`: References of the prompt where the feedback may be applied. Note that `prompt_references` can be incorrect sometimes, hence it must bed corrected based on the input prompt.

```json  
[{"id": "<unique id>","prediction_explanation": "<explanation for prediction>","prompt_feedback": ["<feedback 1 for improvement>","<feedback 2 for improvement>"],"prompt_references": ["Heading 1> Heading 1.2> Heading 1.2.1> body>","Heading 1> Heading 1.2> Heading 1.2.1> 2.> body","Heading 1> Heading 1.2> body>","Heading 2> body>"]},...]  
```

## Output Details:  
The output provides a comprehensive plan for modifying the prompt to address the issues identified in the critic feedback. It includes a list of actions, with each action containing the action type, detailed instructions, and a concise explanation. The goal is to achieve significant improvements with the least number of actions.

### Output Structure:
Below is an example output structure.
```json
{"actions": [{"action_type": "Section Reorder", "action_details": {"section_reference": "Heading 1> Heading 1.2> Heading 1.2.1", "new_position": "Heading 1> Heading 1.2> Heading 1.2.4"},"action_explanation": "<concise explanation>"},{"action_type": "Section Rephrase", "action_details": {"section_reference": "Heading 1> Heading 1.2> Heading 1.2.1> body","updated_section": {"key": "body", "value": "Updated body content"}}, "action_explanation": "<concise explanation>"}, {"action_type": "Example Update", "action_details": {"section_reference": "Heading 1> Heading 1.2> Heading 1.2.1> 1.", "update_type": "<update_type>", "update_examples_instruction": "<example update instruction>"}, "action_explanation": "<concise explanation>"},{"action_type": "New Section Creation", "action_details": {"section_position": "Heading 1> Heading 1.2", "new_section_structure": {"<Heading 1.3>":{"body": "<New section body content>", "Examples":["<example>", ...], "1.":{"body":"<New instruction 1>", "Examples": [...]},"2.":{"body":"<New instruction 2>", "Examples": [...]}}}}, "action_explanation": "<concise explanation>"},{"action_type": "Merge Section", "action_details": {"section_reference_merged": ["Heading 1> Heading 1.2> Heading 1.2.1", "Heading 1> Heading 1.3"], "section_position": "Heading 1> Heading 1.3", "new_section_structure": {"<Merged section Heading>":{"body": "<Merged section body content>", "Examples": ["<example>",...]}}}, "action_explanation": "<concise explanation>"}]}
```

## Constraints:
- Any section must not contain more than 5-6 examples.
- Ensure the prompt is optimized in length without sacrificing clarity, with simplified language, relevant examples, and a clear sequence of steps while implementing changes. Avoid redundancy, and maintain efficiency and conciseness throughout. 
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

## Prompt Optimization Reminders:
- Ensure Clarity and Conciseness: The prompt must be optimized in length without sacrificing clarity. Use simplified language for better understanding.
- Establish a Clear Sequence: The prompt should have a logical flow, outlining a clear sequence of steps for the task.
- Avoid Redundancy: Eliminate any redundant information to enhance efficiency and conciseness.
- Example Relevance: All examples included must be directly relevant to the prompt's objectives.

## Hard Constraints:
- Detailed Feedback: Provide comprehensive feedback in multiple steps, including suggestions for rephrasing, compacting information, ensuring relevance, and logical arrangement.
- `Examples` of any section must be a list with comma separated examples.
- `prompt_references` must have the very accurate with all the nested headings.
- Strict Example Limit: `Examples` of any section must contain no more than 5-6 examples. Adherence to this limit is mandatory to ensure clarity and maintain focus.
<|im_end|>
<|im_start|>assistant
```json