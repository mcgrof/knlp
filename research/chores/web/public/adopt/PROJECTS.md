# Starting profiles for four projects

These are candidate adoption profiles, not endorsements, completed project
tests or permission to change the repositories. Inspect the selected revision
and contribution rules again before building a local plan. Hardware is a
property of a test, not a reason to run every check on an expensive machine.
None of these starting profiles requires a GPU merely for code review.

## libcxlmi: userspace device-management library

[libcxlmi](https://github.com/computexpresslink/libcxlmi) implements a Compute
Express Link (CXL) management interface. Its
[test workflow](https://github.com/computexpresslink/libcxlmi/blob/d2dd30c0dbbaf1e2d48c2c477fe20f8834a131fe/.github/workflows/test.yml)
builds with Meson and runs mock tests with address/undefined-behavior
sanitizers and Valgrind. Candidate commands include `meson setup build`,
`meson compile -C build` and `./build/tests/mock-tests`, with the workflow's
explicit options and dependencies. These commands were inspected, not run
as part of this guide. Also inspect the separate build, ABI and analysis
workflows; ABI means the binary interface seen by library consumers.

Separate mock command/response behavior from actual transport, device and
firmware behavior. A QEMU profile must name the available CXL device model
and management operation; do not assume every management transport or command
is emulated. Real-device tests need a dedicated device and an operator grant
that distinguishes observation from configuration, reset and firmware actions.
Ask a human to decide compatibility promises, unsupported hardware coverage
and any state-changing operation. A mock pass cannot close those questions.

## ebpf-syscall: kernel instrumentation and replay

[ebpf-syscall](https://github.com/SamsungDS/ebpf-syscall) combines userspace
tools with extended Berkeley Packet Filter (eBPF) kernel programs. Its
[build file](https://github.com/SamsungDS/ebpf-syscall/blob/f620df63614e494a13f5102cfe5a82202d7fafc8/Makefile)
and [README](https://github.com/SamsungDS/ebpf-syscall/blob/f620df63614e494a13f5102cfe5a82202d7fafc8/README.md)
are the starting evidence, not an assumed generic `make test` target. The
build can fetch and compile dependencies; that needs a reviewed provisioning
step. Offline parsing tests and build checks do not prove verifier admission,
attachment, event collection or replay effects on a running kernel.

Use a disposable guest with the required kernel configuration and BPF Type
Format (BTF) metadata for the selected attach points. Bound the workload and
verify attach, event meaning, lost-event reporting, detach and cleanup. Driver
or storage-specific events can need modules, emulated devices or real hardware
that an ordinary guest lacks. Keep syscall replay and tracing of other users
out of the default grant. A human selects privileged access, trace-data
handling, a real-device target or a host/kernel change. Do not infer runtime
coverage from a compiler exit code.

## Rush: public status now, implementation later

[rush-status](https://github.com/rushbuild/rush-status) is a public status and
statistics repository for a build engine. It is not the engine's source tree.
Its [published README](https://github.com/rushbuild/rush-status/blob/1aa0846a5fa82fe79816d014d6467d80c68251d8/README.md)
can guide a status-repository profile, not a claimed engine test suite.
Use separate policy scopes for public status maintenance and any separately
authorized implementation work. Do not fabricate private test commands or
infer implementation readiness from published graphs.

For a future build-engine profile, discover language/toolchain checks,
dependency execution, deterministic outputs, cache correctness and invalidation,
parallelism and filesystem semantics from the actual source and CI. Tests
can execute arbitrary build actions; isolate them from credentials and host
state. Use CPU correctness tests first. Bare-metal performance and portability
claims need their own controls. The human decides access to unreleased source,
API direction, compatibility, release readiness and which private results may
appear in the public status repo. This is a proposed profile, not a statement
that Rush uses this starter.

## fio: storage workload tool with device-specific tests

[fio](https://github.com/axboe/fio) exercises I/O workloads. Its
[general CI](https://github.com/axboe/fio/blob/80eb2516037c627d77cd7e8a428cb534b830f9b6/.github/workflows/ci.yml)
and [QEMU workflow](https://github.com/axboe/fio/blob/80eb2516037c627d77cd7e8a428cb534b830f9b6/.github/workflows/qemu.yml)
separate build/test variants and emulated NVMe, integrity and zoned-device
cases. Discover applicable options from `t/run-fio-tests.py` and the selected
workflow revision. Do not run device jobs simply because they live under a
test directory. Consult contribution guidance and the
[fio mailing list](https://lore.kernel.org/fio/) as well as GitHub discussions.

Start with bounded jobs using newly created task-owned files. A QEMU guest
with disposable backing images can cover supported command and error paths;
that does not establish physical-device performance or firmware behavior.
Raw block writes, format, discard/trim, passthrough and zoned-device operations
need explicit test-specific grants, exclusive ownership, identity checks and
recovery. A cloud GPU instance is irrelevant unless a selected test actually
needs its GPU, and may not expose suitable raw storage. Ask an operator for
the exact resource, not “any available disk.”

## Promote one profile at a time

For each project, produce a separate policy proposal, evidence-linked test
matrix and human-delegation choices. A shared template does not share
credentials, merge rights, hardware grants or a ledger across organizations.
Select a small real incoming change, agree on the acceptance test, and inspect
the first result before admitting recurring work. Keep missing hardware and
uncertain policy visible instead of changing them into skipped green checks.
