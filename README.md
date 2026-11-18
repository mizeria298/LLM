# LLM

Two independent Poetry environments, one per machine:

| env | where | Python | PyTorch |
|-----|-------|--------|---------|
| `envs/legacy`  | old PC (GTX 970) | 3.10 | 2.7.1 + cu118 |
| `envs/cluster` | cluster (A100)   | 3.12 | >=2.9 + cu128 |

```bash
poetry -P envs/cluster install --no-root   # or envs/legacy
source envs/cluster/.venv/bin/activate     # Windows: envs\legacy\.venv\Scripts\activate
```
