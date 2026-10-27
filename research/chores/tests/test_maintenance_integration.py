"""Actual disposable Git refs, test failures, repair and replay recovery."""

from concurrent.futures import ThreadPoolExecutor
import subprocess

import pytest

from maintenance_adapter import LocalGitAdapter, commit, git
from test_adoption import assess, documents, maintenance, refresh_policy, value


@pytest.fixture
def integration(tmp_path, documents):
    repo = tmp_path / "disposable.git"
    subprocess.run(
        ["/usr/bin/git", "init", "--bare", str(repo)], capture_output=True, check=True
    )
    target = commit(repo, "See [manual](missing.md).\n")
    head = commit(repo, "See [manual](manual.md).\n", target)
    git(repo, "update-ref", "refs/heads/main", target)
    git(repo, "update-ref", "refs/heads/worker", head)
    d = documents
    for obj in [
        d[1],
        d[2]["results"][0],
        value(d, "candidate"),
        value(d, "review"),
        value(d, "protections"),
        value(d, "effects"),
    ]:
        obj.update(head=head, target=target)
    d[2]["results"][0]["tested_revision"] = value(d, "candidate")["tested_revision"] = (
        head
    )
    value(d, "policy")["baseline_revision"] = target
    adapter = LocalGitAdapter(repo, maintenance.digest(d[0]))
    return repo, target, head, adapter, d


def test_repair_merge_and_postmerge_verification(integration):
    repo, target, head, adapter, d = integration
    assert not adapter.check(target)
    assert adapter.check(head)
    assert assess(d)["status"] == "ready"
    receipt = adapter.merge(head, target, d[1]["policy_digest"])
    assert git(repo, "rev-parse", "refs/heads/main") == head
    d[1]["next_action"] = "verify"
    value(d, "effects").update(
        state="merged",
        merged_revision=receipt["merged_revision"],
        receipt=receipt["receipt"],
    )
    assert adapter.check(receipt["merged_revision"])
    value(d, "effects")["state"] = "verified"
    assert assess(d)["status"] == "complete"


@pytest.mark.parametrize("changed_ref", ["main", "worker"])
def test_atomic_refs_reject_changed_target_or_candidate(integration, changed_ref):
    repo, target, head, adapter, d = integration
    assert adapter.check(head)
    changed = commit(repo, "Other unrelated maintenance.\n", target)
    git(repo, "update-ref", "refs/heads/" + changed_ref, changed)
    with pytest.raises(subprocess.CalledProcessError):
        adapter.merge(head, target, d[1]["policy_digest"])
    assert not git(repo, "for-each-ref", "refs/chores/receipts")


def test_lost_reply_and_duplicate_workers_have_one_receipt(integration):
    repo, target, head, adapter, d = integration
    adapter.check(head)
    with pytest.raises(TimeoutError):
        adapter.merge(head, target, d[1]["policy_digest"], lose_response=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(
            pool.map(
                lambda _: adapter.merge(head, target, d[1]["policy_digest"]), range(2)
            )
        )
    assert all(r["recovered"] for r in receipts)
    assert (
        len(
            git(
                repo, "for-each-ref", "--format=%(refname)", "refs/chores/receipts"
            ).splitlines()
        )
        == 1
    )


def test_advisory_ready_cannot_replace_native_producer_and_grant(integration):
    _, target, head, adapter, d = integration
    assert assess(d)["status"] == "ready"
    with pytest.raises(ValueError):
        adapter.merge(head, target, d[1]["policy_digest"])
    adapter.check(head)
    adapter.lease_generation += 1
    with pytest.raises(ValueError):
        adapter.merge(head, target, d[1]["policy_digest"])
    adapter.worker_generation = adapter.lease_generation
    adapter.revoked = True
    with pytest.raises(ValueError):
        adapter.merge(head, target, d[1]["policy_digest"])


def test_concurrent_first_writers_commit_at_most_once(integration):
    repo, target, head, adapter, d = integration
    adapter.check(head)

    def attempt(_):
        try:
            return adapter.merge(head, target, d[1]["policy_digest"])
        except subprocess.CalledProcessError:
            # A transaction that loses the race cannot publish a second ref;
            # its caller must reconcile, just as after an uncertain response.
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert any(results)
    assert git(repo, "rev-parse", "refs/heads/main") == head
    assert (
        len(
            git(
                repo, "for-each-ref", "--format=%(refname)", "refs/chores/receipts"
            ).splitlines()
        )
        == 1
    )


def test_human_resume_and_failed_postmerge_stay_open(integration):
    _, target, head, adapter, d = integration
    d[0]["classes"]["documentation"]["human_review"] = "before_merge"
    refresh_policy(d)
    adapter.policy_digest = d[1]["policy_digest"]
    adapter.human_required = True
    assert assess(d)["status"] == "needs_human"
    adapter.check(head)
    with pytest.raises(ValueError):
        adapter.merge(head, target, d[1]["policy_digest"])
    value(d, "review").update(decision="approved", approved_actions=["merge"])
    adapter.human_decision = (head, target, d[1]["policy_digest"])
    assert assess(d)["status"] == "ready"
    adapter.check(head)
    receipt = adapter.merge(head, target, d[1]["policy_digest"])
    d[1]["next_action"] = "verify"
    value(d, "effects").update(
        state="post_merge_failed", merged_revision=head, receipt=receipt["receipt"]
    )
    d[2]["results"][0].update(outcome="failed", assertions={"links-resolve": "failed"})
    report = assess(d)
    assert report["status"] == "blocked" and report["handoff"]["route"]
