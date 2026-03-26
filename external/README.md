# External Code

Use this directory for upstream codebases that we do not want to rewrite into this repo by hand.

For OGBench, the intended location is:

- `external/ogbench`

Bootstrap it with:

```bash
bash scripts/bootstrap_ogbench.sh
```

That bootstrap helper also reapplies our tracked local OGBench patch so the checkout keeps:

- `--dataset_dir`
- `--wandb_mode=online|offline|disabled`
