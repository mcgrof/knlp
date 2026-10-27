# Repository test plan

This is an unfilled proposal. No command or resource is approved by copying
this file. Record the owner's decisions and exact source evidence before
running proposed code. Avoid secrets, device serials and private host details
in public copies; keep actual destructive-target identity in a private grant.

## Identity and scope

Repository, upstream contribution policy, base/head/merge-base hashes:

Changed paths and affected interfaces:

Required build, lint, format, unit, integration and regression gates:

Command provenance (file and commit), dependencies and why each check applies:

## Test matrix

Add one row per required check. Do not infer that the examples were executed.

| Check | Exact argv and working directory | CPU / QEMU / bare metal / real GPU | Required configuration and assertions | Resource/grant reference | Result and evidence hash |
| --- | --- | --- | --- | --- | --- |
| To be selected | Not yet approved | Not yet assigned | Not yet established | None | Not run |

For each row record time, CPU/RAM/storage limits, network/dependency policy,
kernel and modules where relevant, guest image, device features, GPU runtime
and memory requirements, output bounds, timeout, cleanup and recovery.
“Pass” requires the named assertions at the exact revision. Use separate
states for fail, skip, unavailable and not run. A changed base, head, test
command, dependency, kernel or device configuration can invalidate coverage.

## Safety and resource assignment

Isolation and credential exclusion, including child processes and plugins:

Task-owned scratch paths; no production mounts or physical disks by default:

For a destructive test: approved target, current identity recheck, exclusive
ownership, recoverable backup/restore, post-test verification and operator:

For cloud/GPU work: approved allocation class, budget and deadline, resource
lease, cancellation, failure teardown, artifact retention and billing receipt:

For kernel/QEMU work: guest config, supported emulated features, console,
bounded boot/attach test and recovery; physical checks that remain required:

## Decision and handoff

Base/head comparison and existing environmental failures:

Required but unavailable checks and their effect on acceptance:

Next bounded request to the owner or hardware operator:

Approver, approval reference, validity window and cancellation route:
