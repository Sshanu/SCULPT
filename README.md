# SCULPT

This repository contains the code for SCULPT: Systematic Tuning of Long Prompts.

## Running the Methods

To set up the project and run the different methods included in this repository, follow these steps:

1.  Install the required dependencies using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

2.  Run the desired method using the provided shell scripts and specify the task name as an argument:

    *   To run the SCULPT method:
        ```bash
        ./run_sculpt.sh <task_name>
        ```
    *   To run the Protegi method:
        ```bash
        ./run_protegi.sh <task_name>
        ```
    *   To run the OPRO method:
        ```bash
        ./run_opro.sh <task_name>
        ```
    *   To run the APEX method:
        ```bash
        ./run_apex.sh <task_name>
        ```
    *   To run the APE method:
        ```bash
        ./run_ape.sh <task_name>
        ```
    *   To run the LongAPE method:
        ```bash
        ./run_longape.sh <task_name>
        ```

    Replace `<task_name>` with the desired task (e.g., `formal_fallacies`, `causal_judgment`, `disambiguation_qa`, `salient_translation`, `go_emotions`, `beaver_tails`).

## Citation

If you use this code or the concepts from our work, please cite our paper published in the main conference of ACL 2025:

**SCULPT: Systematic Tuning of Long Prompts**
arXiv: https://arxiv.org/pdf/2410.20788

```bibtex
@article{kumar2024sculpt,
  title={SCULPT: Systematic Tuning of Long Prompts},
  author={Kumar, Shanu and Venkata, Akhila Yesantarao and Khandelwal, Shubhanshu and Santra, Bishal and Agrawal, Parag and Gupta, Manish},
  journal={arXiv preprint arXiv:2410.20788},
  year={2024}
}