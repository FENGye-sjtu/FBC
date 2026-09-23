"""Compute-matched Qwen2.5 NF4 v5/v6 CoT prompt pair in one model process."""

from trainingfree.run import main


if __name__ == "__main__":
    root = "/data_cinema/zqc/CVPR/workstreams/faithact75_20260923"
    common = [
        "--model", "/data_cinema/zqc/CVPR/models/Qwen2.5-VL-7B-Instruct",
        "--dataset", "/data_cinema/zqc/CVPR/data/vlmevalkit/cases.jsonl",
        "--image-root", "/data_cinema/zqc/CVPR/data/realworldqa",
        "--methods", "cot", "--offset", "0", "--limit", "128",
        "--device", "cuda:0", "--quantization", "nf4",
        "--max-pixels", "1003520", "--max-new-tokens", "512",
    ]
    shared = {}
    main(common + ["--cot-style", "evidence_layout_v5",
                   "--output", root + "/runs/qwen25_nf4_prompt_pair_dev128_v5"], shared)
    main(common + ["--cot-style", "discriminative_evidence_v6",
                   "--output", root + "/runs/qwen25_nf4_prompt_pair_dev128_v6"], shared)
