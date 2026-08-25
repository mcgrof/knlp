# Upstream pull request: fix the released reproduction path

Three defects stop the released AgentFloor tree from reproducing its own paper
or running its own primary sweep. Each is a one-liner. The branch is prepared,
verified, and waiting on GitHub credentials to push.

## Branch

`fix-released-reproduction-path`, three commits on upstream's tip `f172b34`,
built and verified at `prune:~/agentfloor`.

```
018f54f  Give providers without per-run state the documented reset_run_state no-op
1eef1f2  Resolve the task set from the repository root, not the runs directory
2ed2f84  Skip undefined diagnostics when bootstrapping confidence intervals
```

Commit style follows the project's own: imperative sentence-case subject, prose
body, no trailers.

## Pushing it

There is no `gh` binary and no GitHub token on either machine, so this could not
be pushed automatically. Once a credential exists:

```sh
ssh prune
cd ~/agentfloor
gh repo fork rkarmaka/AgentFloor --remote=false --clone=false
git remote add fork git@github.com:<you>/AgentFloor.git
git push fork fix-released-reproduction-path
gh pr create --repo rkarmaka/AgentFloor \
  --head <you>:fix-released-reproduction-path \
  --title "Fix the released reproduction path" \
  --body-file UPSTREAM_PR_BODY.md
```

The PR body is in [UPSTREAM_PR_BODY.md](UPSTREAM_PR_BODY.md).

## Verification performed

The scoring fix was verified from a clean detached worktree at the branch tip:
the released corpus loads 16,542 runs, filters to 12,330 in the paper subset,
and returns the published per-tier table.

The two sweep fixes were verified by running the sweep itself on a single 48 GB
card through ollama: the tool-support probe reports native tool calling, tasks
execute with no errors, and the scorer returns pass rates consistent with the
published per-tier rates. Before the fixes the sweep could not complete a single
run.
