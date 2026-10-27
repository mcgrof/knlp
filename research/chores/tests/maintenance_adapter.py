"""Disposable native-Git qualification adapter, never a production writer.

The fixture owns a bare repository, an in-memory standing grant and test
producer. It deliberately has no CLI, remote transport or credential path.
Policy/source authentication and hostile-code isolation need a real adapter.
"""

import hashlib
import json
from pathlib import Path
import re
import subprocess


def git(repo, *args, data=None):
    return (
        subprocess.run(
            [
                "/usr/bin/git",
                "-c",
                "core.hooksPath=/dev/null",
                "-c",
                "core.fsmonitor=false",
                "--git-dir=" + str(repo),
                *args,
            ],
            input=data,
            capture_output=True,
            check=True,
            timeout=5,
            env={
                "PATH": "/usr/bin:/bin",
                "GIT_CONFIG_NOSYSTEM": "1",
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_NO_LAZY_FETCH": "1",
                "GIT_AUTHOR_NAME": "Fixture",
                "GIT_AUTHOR_EMAIL": "fixture@example.invalid",
                "GIT_COMMITTER_NAME": "Fixture",
                "GIT_COMMITTER_EMAIL": "fixture@example.invalid",
            },
        )
        .stdout.decode()
        .strip()
    )


def commit(repo, text, parent=None):
    blob = git(repo, "hash-object", "-w", "--stdin", data=text.encode())
    manual = git(repo, "hash-object", "-w", "--stdin", data=b"Fixture manual\n")
    docs = git(
        repo,
        "mktree",
        data=(
            f"100644 blob {blob}\tguide.md\n100644 blob {manual}\tmanual.md\n"
        ).encode(),
    )
    tree = git(repo, "mktree", data=f"040000 tree {docs}\tdocs\n".encode())
    args = ["commit-tree", tree] + (["-p", parent] if parent else [])
    return git(repo, *args, data=b"Disposable maintenance fixture\n")


class LocalGitAdapter:
    def __init__(self, repo, policy_digest):
        self.repo = Path(repo)
        self.policy_digest = policy_digest
        self.worker_generation = self.lease_generation = 1
        self.revoked = False
        self.tested = set()
        self.human_required = False
        self.human_decision = None

    def check(self, revision):
        # Fixed trusted assertion, not a command read from candidate source.
        text = git(self.repo, "show", revision + ":docs/guide.md")
        links = re.findall(r"\]\(([a-z-]+\.md)\)", text)
        passed = bool(links)
        for link in links:
            try:
                git(self.repo, "cat-file", "-e", revision + ":docs/" + link)
            except subprocess.CalledProcessError:
                passed = False
        if passed:
            self.tested.add(revision)
        return passed

    def merge(self, head, target, policy_digest, *, lose_response=False):
        if self.revoked or self.worker_generation != self.lease_generation:
            raise ValueError("standing grant revoked or worker fenced")
        if policy_digest != self.policy_digest or head not in self.tested:
            raise ValueError("trusted grant and producer receipt required")
        if self.human_required and self.human_decision != (head, target, policy_digest):
            raise ValueError("current human decision required by standing grant")
        identity = json.dumps([head, target, policy_digest], separators=(",", ":"))
        key = hashlib.sha256(identity.encode()).hexdigest()
        receipt = "refs/chores/receipts/" + key
        # A receipt is committed atomically with the branch update. A lost
        # response is reconciled here before any second update is attempted.
        try:
            previous = git(self.repo, "rev-parse", "--verify", receipt)
        except subprocess.CalledProcessError:
            previous = None
        if previous:
            if previous != head:
                raise ValueError("receipt identity conflict")
            return {"receipt": receipt, "merged_revision": previous, "recovered": True}
        if git(self.repo, "diff", "--name-only", target, head) != "docs/guide.md":
            raise ValueError("candidate outside the fixture grant")
        entries = git(self.repo, "ls-tree", "-r", head).splitlines()
        if len(entries) != 2 or any(
            not entry.startswith("100644 blob ") for entry in entries
        ):
            raise ValueError("fixture accepts only regular prose files")
        git(self.repo, "merge-base", "--is-ancestor", target, head)
        transaction = (
            "start\n"
            f"verify refs/heads/worker {head}\n"
            f"update refs/heads/main {head} {target}\n"
            f"create {receipt} {head}\n"
            "prepare\ncommit\n"
        )
        git(self.repo, "update-ref", "--stdin", data=transaction.encode())
        if lose_response:
            raise TimeoutError("simulated reply lost after commit")
        return {"receipt": receipt, "merged_revision": head, "recovered": False}
