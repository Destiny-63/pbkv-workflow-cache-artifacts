# PBKV Workflow Cache

Research artifacts for studying KV-cache eviction in dynamic multi-agent workflows. The repository includes HoVer-MA trace collection, a CPU cache-replay simulator, and the code and weights for a step-level GraphSAGE role predictor.

## Run the HoVer replay

From a fresh checkout with Python 3.10 or later, install NumPy and run the validation split at the middle cache-pressure setting:

```bash
git clone https://github.com/Destiny-63/pbkv-workflow-cache-artifacts.git
cd pbkv-workflow-cache-artifacts
python -m pip install 'numpy>=1.24'
PYTHONHASHSEED=0 python -m pbkv_sim.run_g1 \
  --split-name val --only-mid --jobs 1 \
  --policies he_step --predictors markov --out replay-output
```

The command writes one JSON summary per admission seed, with 32 concurrent workflows and capacity ratio 0.5. Each summary includes the cache budget, token hits, and refetched tokens. The ratio applies to the gap between peak locked tokens and peak cache size. This example runs offline on the shipped traces and needs only a CPU.

## Data and code

- [HoVer-MA traces](data/hover_rolefirst/) include 1,500 paired workflow IDs across default and high-difficulty settings, with 22,462 model calls. The [split mapping](results/split.json) assigns IDs to train, validation, and test in counts of 1,000, 200, and 300. Trace records retain role order, token counts, prefix-segment hashes, and timing relative to the start of each workflow.
- [Collection](collect/) includes the HoVer collectors, prompt templates, transcript handling, and SWE benchmark utilities. Install the [full dependencies](requirements.txt) and run `python collect/hover_ma_v3.py --help` for collection options. Fresh collection requires a user-supplied SGLang endpoint and matching tokenizer. The collector requests hidden states from the serving backend.
- [Replay and prediction](pbkv_sim/) include cache eviction policies, Markov role prediction, and the training and inference entry points. Run `python -m pytest -q pbkv_sim/tests collect/tests` after installing the full dependencies.

## Load the SWE predictor

The [checkpoint](models/pbkv-reimpl-seed0.pt) is the seed-0 model selected by validation loss on the available SWE development subset. It predicts three future roles over the SWE role vocabulary, including `END`. With PyTorch and NumPy installed, this CPU example checks the checkpoint interface using a synthetic input:

```python
import torch
from vendor.pbkv_0213e13.chain_predictor import GraphSAGEChainPredictor

predictor = GraphSAGEChainPredictor("models/pbkv-reimpl-seed0.pt", device="cpu")
probabilities = predictor.predict_distributions(
    torch.zeros(predictor.emb_dim), [], "Planner"
)
print(predictor.agent_labels, probabilities.shape)
```

For actual inference, supply the current step's matching post-norm last-prefill-token embedding and the preceding role history. Training requires the corresponding recorded hidden states and an input split. See `python -m pbkv_sim.train_predictor --help` and `python -m pbkv_sim.infer_predictor --help` for the corpus interfaces.

## Measurement scope

The [SWE test summary](docs/cache-replay-test-results.md) covers offline replay on the available development subset. The ETA variants in that comparison use recorded future timing. The next-access reference uses the complete future access sequence. The reported token counts measure offline cache behavior. The raw SWE traces and hidden states are absent from this release, so the bundled HoVer traces support the quickstart above and the SWE tables serve as a historical summary.

## Attribution and licenses

- Upstream architecture and policy source are pinned to PBKV commit `0213e132`. [PBKV source](https://github.com/Hyzheng-code/PBKV/tree/0213e132efd5c49eb736bc08d032d4207cdcedf6) · [PBKV paper](https://arxiv.org/abs/2605.06472)
- The released trace metadata comes from workflow runs based on HoVer. [HoVer paper](https://aclanthology.org/2020.findings-emnlp.309/) · [HoVer dataset card](https://huggingface.co/datasets/vincentkoc/hover-parquet)
- [Original code](LICENSE) is MIT licensed, the [vendored PBKV implementation](vendor/pbkv_0213e13/LICENSE) retains Apache-2.0, and the [trace metadata](data/LICENSE) is CC BY 4.0.
