# Generic maintenance profiles

These profiles describe test questions, not prospective adopters or qualified
deployments. Discover the exact commands from the selected repository's trusted
instructions, build files and CI. A language or filename alone cannot establish
a change's effects. Keep project identities and private access details local.

## Portable systems library

Routine checks can cover builds, unit contracts, sanitizers, failure paths and
the supported application binary interface (ABI). An ABI is the interface seen
by compiled consumers. Device transport, protocol and firmware-sensitive
changes need separately declared device coverage. Mock responses cannot prove
a physical device behaves correctly. An emulator is acceptable only for named
features supported by the chosen device model. Reset, firmware and configuration
actions require a specific device/operator grant.

## Kernel integration

Separate offline tools and builds from kernel admission, loading, attachment,
event meaning, loss reporting, detach and cleanup. A disposable QEMU guest
needs the selected configuration, debug metadata, modules, workload and console
recovery. A guest or container with the wrong kernel cannot stand in for the
required one. Real driver/device paths can require physical hardware. Tracing
other users, replaying operations and changing a host kernel are not default
test permissions. Preserve a patch while its exact runtime test is delegated.

## Compiler or build tool

Check clean and incremental builds, dependency tracking, deterministic output,
cache invalidation, parallel execution and supported toolchains/platforms.
Dependency/build actions execute code, so isolate them from credentials and
host state. A public status repository is not implementation source or evidence
that these checks pass. Performance claims require controlled comparisons
rather than a successful build. API, compatibility and release choices can
remain with a maintainer while ordinary repairs proceed under standing rules.

## Storage workload tool

Separate parser/job contracts, bounded jobs on newly created task-owned files,
supported emulated-device features and physical-device tests. A protocol path
in QEMU does not establish physical timing or firmware behavior. Raw writes,
format, discard and zoned-device operations require an explicitly allocated
disposable target, current identity/exclusivity checks and operator-approved
recovery. Never select an arbitrary disk. A cloud container may not expose
the raw device or host APIs needed by a test.

## GPU-dependent checks

A real GPU is necessary only when the assertion concerns device execution,
memory behavior or a GPU runtime/API. Declare vendor/runtime features, required
memory, fallback detection and resource limits. CPU mocks do not satisfy that
coverage, and a GPU allocation does not imply bare-metal or physical storage
access. Keep grant/lease identity, teardown and total billing in private operator
records. No local or rented GPU is needed merely to review a patch.

For each profile, map changed behavior to assertions, accepted fidelity and
required evidence. Mixed changes inherit all applicable tests and the most
restrictive permissions. Keep unavailable and optional coverage visible.
