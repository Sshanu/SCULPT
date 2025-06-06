# SCULPT

This repository contains the code for SCULPT: Systematic Tuning of Long Prompts.

## Running the Methods

To run the different methods included in this repository, use the provided shell scripts:

*   To run the SCULPT method:
    ```bash
    ./run_sculpt.sh
    ```
*   To run the Protegi method:
    ```bash
    ./run_protegi.sh
    ```
*   To run the OPRO method:
    ```bash
    ./run_opro.sh
    ```
*   To run the APEX method:
    ```bash
    ./run_apex.sh
    ```
*   To run the APE method:
    ```bash
    ./run_ape.sh
    ```
*   To run the LongAPE method:
    ```bash
    ./run_longape.sh
    ```

Please ensure you have the necessary dependencies installed. You may need to install the `openai` library if you haven't already:

```bash
pip install openai
```

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