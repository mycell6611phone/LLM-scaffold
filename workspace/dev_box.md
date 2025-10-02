0) Goal and scope

Build a turnkey box.

User types requests in a chat UI.

Controller plans work.

Local LLM crew executes inside a VM.

No human confirmations.

Failure is rare. Catastrophic faults recover fast.

Memory learns per task. Only high quality items enter shared memory.

1) High level architecture
[User Chat UI on Host]
        |
        v
[Controller LLM + Bridge API on Host]
        |
   policy check
        |
        v
[VM: Execution Plane]
  - Local LLM crew
  - Tool runners
  - Build and test
  - Memory gate
  - Data store
        |
   health probes
        |
        v
[Host Recovery Daemon]
  - Tripcodes
  - Snapshots
  - Golden image
0) Goal and scope

Build a turnkey box.

User types requests in a chat UI.

Controller plans work.

Local LLM crew executes inside a VM.

No human confirmations.

Failure is rare. Catastrophic faults recover fast.

Memory learns per task. Only high quality items enter shared memory.

1) High level architecturePillars:

Single hardware target.

Blue green promotion.

Debate gates on plans and memory.

Policy enforced tools.

Immutable base with fast restore.

2) Hardware and OS baseline

Bill of materials

CPU: 16c or better.

RAM: 64 GB or better.

GPU: single 24 GB VRAM if needed.

Storage: NVMe 2 TB. Separate disks for OS and data if possible.

NIC: 10 GbE optional.

Host OS

Linux LTS.

Secure boot.

TPM 2.0.

Filesystem: btrfs or ZFS.

VM

KVM with libvirt.

qcow2 golden image. Overlay per Green.

VFIO GPU passthrough if used.

3) Control pipeline

State machine:
IDLE -> PLAN -> DEBATE_PLAN -> SIMULATE -> VERIFY -> PROMOTE
        ^                                 |
        |-------------- ABORT <-----------|

Rules:

Never mutate Blue.

Build Green from golden.

Promote only on full pass.

Abort on any failed check. No rollback needed in normal flow.

4) Policies and capability tiers

Tiers:

T0 read only compute.

T1 writes in workspace inside VM.

T2 system changes in VM after snapshot.

T3 destructive or sensitive. Tripcode required.

Preconditions T1 plus:

Health green.

Disk space headroom.

Policy match.

Dry run diff exists.

Snapshot if tier is T2 or higher.

Postconditions:

Tests pass.

Health probe green for N seconds.

Policy monitor shows no violations.

5) Debate gates

Plan debate

Two local LLMs plus a judge.

Two rounds.

Decision space: ACCEPT REJECT ERROR.

Early stop on agreement.

Tie break by judge rubric.

Fail closed on any ERROR.

Rubric:

Policy compliance 0.5

Simplicity 0.2

Predicted test pass 0.3

Memory debate

Gate all CandidateMemory.

PII and secret detectors.

Structural checks.

Accept on consensus or high score with no red flags.

Store provenance.

6) Memory and knowledge

Stores:

Short term task cache in VM.

Long term memory in host data partition.

Vector index for retrieval.

Audit log with hashes.

Write path:
Trace -> Sanitize -> Debate -> Validators -> Quarantine -> Admit
Provenance fields:

Model versions.

Tool chain IDs.

Data hashes.

Time and policy snapshot.

Revocation:

CRL list.

Fast purge by hash prefix.

7) Learning with LoRA adapters

Layout:

Frozen base model.

Per skill LoRA adapters.

Router picks adapter by intent.

Canary adapter for training.

Promote only after eval pass.

Training path:

Collect sanitized traces.

Debate and validators.

Build SFT batches.

QLoRA micro updates.

Early stop on plateau.
syntax = "proto3";
package orchestrator;

enum CapabilityTier { T0=0; T1=1; T2=2; T3=3; }

message ToolPolicy {
  string name = 1;
  CapabilityTier tier = 2;
  repeated string allowed_args = 3;
  repeated string denied_args = 4;
  uint32 cpu_millicores = 5;
  uint32 mem_mb = 6;
  uint32 io_write_mb = 7;
  bool net_allowed = 8;
  repeated string net_allowlist = 9;
  uint32 rate_limit_per_min = 10;
  uint32 cooldown_sec = 11;
  repeated string fs_roots = 12;
}

message PlanPolicy {
  uint32 max_steps = 1;
  uint32 max_parallel = 2;
  CapabilityTier snapshot_for_tier_ge = 3;
  CapabilityTier tripcode_for_tier_ge = 4;
}

message Step { string tool = 1; map<string,string> args = 2; }
message Plan { repeated Step steps = 1; string intent = 2; }

message SimulateReq { Plan plan = 1; }
message SimulateRes { bool ok = 1; repeated string diffs = 2; string report = 3; }

message SnapshotReq { string label = 1; }
message SnapshotRes { bool ok = 1; string snapshot_id = 2; }

message ApplyReq { Plan plan = 1; string snapshot_id = 2; }
message ApplyRes { bool ok = 1; string run_id = 2; repeated string logs = 3; }
12) Debate and scaffold code

Fallback facade

memory_debate.py exports CandidateMemory ModelInterface DebateConsensusEngine.

Re export real classes from memeryloop.py when present.

Provide working scaffolds otherwise.

Core behavior in scaffolds:

Heuristic evaluation.

Multi round debate.

Consensus test.

Debate log capture.

Status labels: ACCEPTED REJECTED RETAINED ERROR_DEBATE.

13) Build and promotion

Green build steps inside VM:

Checkout workspace.

codegen_t0 emits patch.

apply_patch_t1 applies in scope.

container_build_t1 builds image with pinned base.

service_run_t1 starts container on a shadow port.

Health probe on /healthz for N seconds.

Log scan for policy words.

If pass then promote. Promotion flips a symlink or a supervisor target.

Artifact rules:

Content addressed images.

Record digest and SBOM.

Store in local registry on host.

14) Observability

Metrics:

Plan typeability rate.

Debate agreement rate.

Simulate latency.

Verify pass rate.

Promote success rate.

Health probe SLO.

Tool error counts by tier.

Logs:

Append only.

Signed on host.

Include plan hash and artifact digests.

Traces:

Step spans with timing.

Resource budgets.

Policy decisions.

Dashboards:

Build funnel.

Error budgets.

Adapter learning curves.

15) Security model and threat highlights

Surface:

Chat UI.

Bridge API.

VM guest.

Recovery daemon.

Controls:

Mutual TLS on bridge.

HSM or TPM backed keys.

Signature on tripcodes.

Rate limits.

VM network allowlist.

No host mounts.

Threats and responses:

Prompt injection in code. Use policy checker and static analyzers.

Secret exfiltration. Use redactors and broker leases.

Drift. Use debate and validators.

Supply chain. Pin digests. Use SBOM.

Persistence on host. Keep host immutable. VM only.

16) Test harness

Seed tasks:

Hello service on port 8080.

CSV to SQLite to report.

Nightly cron.

Fault injections:

Tool crash.

Denied egress.

Disk full.

Failing tests.

Stuck process.

Tripcode during apply.

Expected results:

Plan rejected or verify failed before promotion.

No change on Blue.

On catastrophic fault host reverts VM fast.

Success gates:

Promotion success ≥ 99.5 percent.

Post promotion errors zero in first 10 minutes.

No direct writes to Blue.

Deterministic artifacts across runs.

17) Deployment and updates

Image pipeline:

Build golden image.

Bake fixed toolchain.

Prewarm models and wheels.

Sign image.

Ship with hardware.

Update flow:

Build new golden.

Run full harness.

Push to customer boxes.

Promote on next maintenance window.

Keep N previous images for quick revert.

18) User experience

Chat flow:

User types request.

Controller returns plan preview and status.

No confirmation required.

On policy mismatch controller refuses with reason.

On success show result and endpoint.

Support flow:

Export signed diagnostics on failure.

Optional tripcode from support to lock down or revert.

19) Cross install sharing

Method:

Ship adapter deltas after eval pass.

Use secure aggregation if averaging.

Keep same base model and tokenizer across installs.

Never ship raw user data.

20) Key configs to freeze

Kernel and drivers.

CUDA stack if GPU.

Compilers.

Container runtime.

Model checkpoints.

Tokenizers and prompts.

Policy regexes.

Port and path maps.

21) Next concrete steps

Freeze BOM and OS version.

Build golden VM.

Implement Bridge gRPC.

Implement tool gateway with policies.

Integrate memory_debate.py into plan and memory gates.

Wire health probes and promotion switch.

Stand up test harness with chaos.

Train first LoRA adapters from seed traces.

Run soak for one week on loop.
message VerifyReq { string run_id = 1; }
message VerifyRes { bool ok = 1; repeated string checks = 2; }

message PromoteReq { string run_id = 1; }
message PromoteRes { bool ok = 1; }

message RollbackReq { string snapshot_id = 1; }
message RollbackRes { bool ok = 1; }

message Tripcode { string payload = 1; /* ts|code|sig */ }
message TripcodeRes { bool ok = 1; string code = 2; string detail = 3; }

Safety:

Stop learning on metric drop.

No live adapter mutation.

Runtime composition only.

Optional DP if sharing cross installs.

Practical knobs:

r 8 to 16.

alpha 16 to 32.

lr 5e-5 to 1e-4.

epochs 1 to 2 per update.

Promotion criteria:

Pass rate up or equal with lower latency.

No new policy violations.

Seeded probes stable.

8) Recovery and safety

VM centric fail safe

Host UI outside VM.

All workers inside VM.

Immutable golden.

Writable overlay for Green.

Restore by dropping overlay or snapshot revert.

Tripcode protocol

Signed codes from controller.

HMAC with bridge key.

Cooldowns.

Append only audit.

Codes:

RECOVERY:REVERT_SNAPSHOT::<id>

RECOVERY:REIMAGE_TO_GOLDEN

RECOVERY:LOCKDOWN

RECOVERY:DIAGNOSTIC::<profile>

Watchdogs

Health heartbeat from VM.

Missed beats cause revert.

Systemd hardening on host

No new privileges.

Strict filesystem protection.

Capability bound set minimal.

9) Networking and secrets

Network on host:

eBPF or iptables.

VM egress allowlist.

No host mounts into VM.

Optional proxy with mTLS.

Secrets:

Host secret broker.

Short lived leases.

Pass via virtio serial.

Never bake into images.

10) Tool catalog and contracts

Example tools:

codegen_t0.

apply_patch_t1.

config_update_t2.

service_run_t1.

container_build_t1.

db_migrate_t2.

factory_reset_t3.

Tool contract:

Typed args only.

Args match regex allowlist.

FS scope explicit.

CPU RAM IO caps fixed.

Outputs include logs checksums artifacts.

Determinism:

Pin tool versions.

Pin container digests.

Set seeds.

Record hashes.

11) Bridge API surface

Use gRPC with Protobuf. No JSON is canonical.

Messages12) Debate and scaffold code

Fallback facade

memory_debate.py exports CandidateMemory ModelInterface DebateConsensusEngine.

Re export real classes from memeryloop.py when present.

Provide working scaffolds otherwise.

Core behavior in scaffolds:

Heuristic evaluation.

Multi round debate.

Consensus test.

Debate log capture.

Status labels: ACCEPTED REJECTED RETAINED ERROR_DEBATE.

13) Build and promotion

Green build steps inside VM:

Checkout workspace.

codegen_t0 emits patch.

apply_patch_t1 applies in scope.

container_build_t1 builds image with pinned base.

service_run_t1 starts container on a shadow port.

Health probe on /healthz for N seconds.

Log scan for policy words.

If pass then promote. Promotion flips a symlink or a supervisor target.

Artifact rules:

Content addressed images.

Record digest and SBOM.

Store in local registry on host.

14) Observability

Metrics:

Plan typeability rate.

Debate agreement rate.

Simulate latency.

Verify pass rate.

Promote success rate.

Health probe SLO.

Tool error counts by tier.

Logs:

Append only.

Signed on host.

Include plan hash and artifact digests.

Traces:

Step spans with timing.

Resource budgets.

Policy decisions.

Dashboards:

Build funnel.

Error budgets.

Adapter learning curves.

15) Security model and threat highlights

Surface:

Chat UI.

Bridge API.

VM guest.

Recovery daemon.

Controls:

Mutual TLS on bridge.

HSM or TPM backed keys.

Signature on tripcodes.

Rate limits.

VM network allowlist.

No host mounts.

Threats and responses:

Prompt injection in code. Use policy checker and static analyzers.

Secret exfiltration. Use redactors and broker leases.

Drift. Use debate and validators.

Supply chain. Pin digests. Use SBOM.

Persistence on host. Keep host immutable. VM only.

16) Test harness

Seed tasks:

Hello service on port 8080.

CSV to SQLite to report.

Nightly cron.

Fault injections:

Tool crash.

Denied egress.

Disk full.

Failing tests.

Stuck process.

Tripcode during apply.

Expected results:

Plan rejected or verify failed before promotion.

No change on Blue.

On catastrophic fault host reverts VM fast.

Success gates:

Promotion success ≥ 99.5 percent.

Post promotion errors zero in first 10 minutes.

No direct writes to Blue.

Deterministic artifacts across runs.

17) Deployment and updates

Image pipeline:

Build golden image.

Bake fixed toolchain.

Prewarm models and wheels.

Sign image.

Ship with hardware.

Update flow:

Build new golden.

Run full harness.

Push to customer boxes.

Promote on next maintenance window.

Keep N previous images for quick revert.

18) User experience

Chat flow:

User types request.

Controller returns plan preview and status.

No confirmation required.

On policy mismatch controller refuses with reason.

On success show result and endpoint.

Support flow:

Export signed diagnostics on failure.

Optional tripcode from support to lock down or revert.

19) Cross install sharing

Method:

Ship adapter deltas after eval pass.

Use secure aggregation if averaging.

Keep same base model and tokenizer across installs.

Never ship raw user data.

20) Key configs to freeze

Kernel and drivers.

CUDA stack if GPU.

Compilers.

Container runtime.

Model checkpoints.

Tokenizers and prompts.

Policy regexes.

Port and path maps.

21) Next concrete steps

Freeze BOM and OS version.

Build golden VM.

Implement Bridge gRPC.

Implement tool gateway with policies.

Integrate memory_debate.py into plan and memory gates.

Wire health probes and promotion switch.

Stand up test harness with chaos.

Train first LoRA adapters from seed traces.

Run soak for one week on loop.
