import asyncio
import sys
import types as _pytypes
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_VROOT = str(_REPO / "forks" / "verifiers")

sys.path.insert(0, _VROOT)

for _pkg, _subdir in [
    ("prime_rl", str(_REPO / "src" / "prime_rl")),
    ("prime_rl.orchestrator", str(_REPO / "src" / "prime_rl" / "orchestrator")),
    ("verifiers", _VROOT + "/verifiers"),
    ("verifiers.envs", _VROOT + "/verifiers/envs"),
]:
    if _pkg not in sys.modules:
        _mod = _pytypes.ModuleType(_pkg)
        _mod.__path__ = [_subdir]
        sys.modules[_pkg] = _mod

from verifiers.envs.multi_actor_kernel import (
    KernelState,
    StaticSchedule,
    TurnSlot,
    apply_action,
)

from prime_rl.orchestrator.multi_actor import (
    EpisodeResult,
    EpisodeSpec,
    EpisodeStart,
    Member,
    MemberResult,
    PolicyHandle,
    TurnReq,
    TurnResp,
    run_episode,
    run_episode_group,
)

# ---------------------------------------------------------------------------
# Samplers (orchestrator-side — unchanged)
# ---------------------------------------------------------------------------


class FakeBinder:
    def __init__(self):
        self.calls = []

    async def resolve(self, sample, member, request):
        self.calls.append((sample.episode_id, member.member_id, request.turn_id))
        return PolicyHandle(
            policy_slot=f"slot-{member.member_id}",
            client={"member": member.member_id},
            model_name="model",
        )


class FakeSampler:
    def __init__(self):
        self.calls = []

    async def __call__(self, request, policy):
        self.calls.append((request.member_id, request.turn_id, policy.policy_slot))
        await asyncio.sleep(0)
        return TurnResp(
            episode_id=request.episode_id,
            member_id=request.member_id,
            turn_id=request.turn_id,
            content=[],
        )


class WrongRequestIdSampler(FakeSampler):
    async def __call__(self, request, policy):
        return TurnResp(
            episode_id=request.episode_id,
            member_id=request.member_id,
            turn_id="wrong-request",
            content=[],
        )


class WrongMemberSampler(FakeSampler):
    async def __call__(self, request, policy):
        member_id = "B" if request.member_id == "A" else "A"
        return TurnResp(
            episode_id=request.episode_id,
            member_id=member_id,
            turn_id=request.turn_id,
            content=[],
        )


class WrongEpisodeResponseSampler(FakeSampler):
    async def __call__(self, request, policy):
        return TurnResp(
            episode_id="wrong-episode",
            member_id=request.member_id,
            turn_id=request.turn_id,
            content=[],
        )


# ---------------------------------------------------------------------------
# Envs backed by the kernel
# ---------------------------------------------------------------------------

SCHEDULE = StaticSchedule((
    TurnSlot(slot_id=0, actors=("A", "B"), phase="opening"),
    TurnSlot(slot_id=1, actors=("A",), phase="rebuttal"),
))

MEMBERS = [
    Member(member_id="A", role_id="debater", seat_id="A"),
    Member(member_id="B", role_id="debater", seat_id="B"),
]


class HappyPathEnv:
    def __init__(self):
        self.ready_turn_batches = []
        self._sessions: dict[str, KernelState] = {}

    def _make_turn_reqs(self, episode_id: str, state: KernelState) -> list[TurnReq]:
        slot = SCHEDULE.current_slot(state)
        if slot is None:
            return []
        return [
            TurnReq(
                episode_id=episode_id,
                member_id=a,
                turn_id=f"{episode_id}-{slot.slot_id}{a}",
                prompt=[],
            )
            for a in slot.actors
        ]

    def start_episode(self, example, sample_index):
        episode_id = f"episode-{sample_index}"
        state = KernelState(slot_index=0)
        self._sessions[episode_id] = state

        episode = EpisodeSpec(
            base_example_id=example["example_id"],
            episode_id=episode_id,
            input=example,
        )
        return EpisodeStart(
            episode=episode,
            members=list(MEMBERS),
            ready_turns=self._make_turn_reqs(episode_id, state),
        )

    async def submit_ready_turns(self, responses):
        episode_id = responses[0].episode_id
        state = self._sessions[episode_id]
        self.ready_turn_batches.append([r.turn_id for r in responses])

        for resp in responses:
            result = apply_action(state, SCHEDULE, resp.member_id, resp.content or "", resp.token_count or 0)
            state = result.new_state

        self._sessions[episode_id] = state
        return self._make_turn_reqs(episode_id, state)

    async def finalize_episode(self, episode_id):
        self._sessions.pop(episode_id)
        return EpisodeResult(
            base_example_id=1,
            episode_id=episode_id,
            members=[
                MemberResult(member_id="A", role_id="debater", seat_id="A", trajectory=[]),
                MemberResult(member_id="B", role_id="debater", seat_id="B", trajectory=[]),
            ],
            logs={"ready_turn_batches": list(self.ready_turn_batches)},
        )


# Error envs: inherit kernel-backed base, override one method to inject bad data.


class DuplicateMemberEnv(HappyPathEnv):
    def start_episode(self, example, sample_index):
        start = super().start_episode(example, sample_index)
        return EpisodeStart(
            episode=start.episode,
            members=[
                Member(member_id="A", role_id="debater", seat_id="A"),
                Member(member_id="A", role_id="debater", seat_id="B"),
            ],
            ready_turns=[],
        )


class DuplicateFrontierMemberEnv(HappyPathEnv):
    def start_episode(self, example, sample_index):
        start = super().start_episode(example, sample_index)
        return EpisodeStart(
            episode=start.episode,
            members=list(MEMBERS),
            ready_turns=[
                TurnReq(episode_id=start.episode.episode_id, member_id="A",
                         turn_id=f"{start.episode.episode_id}-0A", prompt=[]),
                TurnReq(episode_id=start.episode.episode_id, member_id="A",
                         turn_id=f"{start.episode.episode_id}-0A-dup", prompt=[]),
            ],
        )


class WrongRequestEpisodeEnv(HappyPathEnv):
    def start_episode(self, example, sample_index):
        start = super().start_episode(example, sample_index)
        return EpisodeStart(
            episode=start.episode,
            members=[Member(member_id="A", role_id="debater", seat_id="A")],
            ready_turns=[TurnReq(episode_id="wrong-episode", member_id="A", turn_id="req-0", prompt=[])],
        )


class WrongNextRequestEpisodeEnv(HappyPathEnv):
    async def submit_ready_turns(self, responses):
        return [TurnReq(episode_id="wrong-episode", member_id="A", turn_id="late", prompt=[])]


class DuplicateSampleIdEnv(HappyPathEnv):
    def start_episode(self, example, sample_index):
        state = KernelState(slot_index=0)
        self._sessions["same-episode"] = state
        episode = EpisodeSpec(base_example_id=example["example_id"], episode_id="same-episode", input=example)
        return EpisodeStart(episode=episode, members=list(MEMBERS), ready_turns=[])


class MissingMemberOutputEnv(HappyPathEnv):
    async def finalize_episode(self, episode_id):
        self._sessions.pop(episode_id, None)
        return EpisodeResult(
            base_example_id=1,
            episode_id=episode_id,
            members=[MemberResult(member_id="A", role_id="debater", seat_id="A", trajectory=[])],
        )


class WrongBaseExampleOutputEnv(HappyPathEnv):
    async def finalize_episode(self, episode_id):
        self._sessions.pop(episode_id, None)
        return EpisodeResult(
            base_example_id=999,
            episode_id=episode_id,
            members=[
                MemberResult(member_id="A", role_id="debater", seat_id="A", trajectory=[]),
                MemberResult(member_id="B", role_id="debater", seat_id="B", trajectory=[]),
            ],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_episode_happy_path():
    async def run():
        env = HappyPathEnv()
        binder = FakeBinder()
        sampler = FakeSampler()

        output = await run_episode(
            env=env,
            example={"example_id": 1},
            sample_index=0,
            policy_binder=binder,
            sample_member=sampler,
        )

        assert output.episode_id == "episode-0"
        assert env.ready_turn_batches == [["episode-0-0A", "episode-0-0B"], ["episode-0-1A"]]
        assert binder.calls == [
            ("episode-0", "A", "episode-0-0A"),
            ("episode-0", "B", "episode-0-0B"),
            ("episode-0", "A", "episode-0-1A"),
        ]
        assert sampler.calls == [
            ("A", "episode-0-0A", "slot-A"),
            ("B", "episode-0-0B", "slot-B"),
            ("A", "episode-0-1A", "slot-A"),
        ]

    asyncio.run(run())


def test_run_episode_rejects_duplicate_members():
    async def run():
        try:
            await run_episode(DuplicateMemberEnv(), {"example_id": 1}, 0, FakeBinder(), FakeSampler())
        except ValueError as exc:
            assert "Duplicate member_id" in str(exc)
        else:
            raise AssertionError("Expected duplicate member ids to fail")

    asyncio.run(run())


def test_run_episode_rejects_duplicate_member_in_ready_turns():
    async def run():
        try:
            await run_episode(DuplicateFrontierMemberEnv(), {"example_id": 1}, 0, FakeBinder(), FakeSampler())
        except ValueError as exc:
            assert "Duplicate member_id in ready_turns" in str(exc)
        else:
            raise AssertionError("Expected duplicate ready-turn member ids to fail")

    asyncio.run(run())


def test_run_episode_rejects_request_for_wrong_episode():
    async def run():
        try:
            await run_episode(WrongRequestEpisodeEnv(), {"example_id": 1}, 0, FakeBinder(), FakeSampler())
        except ValueError as exc:
            assert "Request episode_id does not match" in str(exc)
        else:
            raise AssertionError("Expected mismatched request episode id to fail")

    asyncio.run(run())


def test_run_episode_rejects_new_request_for_wrong_episode():
    async def run():
        try:
            await run_episode(WrongNextRequestEpisodeEnv(), {"example_id": 1}, 0, FakeBinder(), FakeSampler())
        except ValueError as exc:
            assert "different episode" in str(exc)
        else:
            raise AssertionError("Expected mismatched next request episode id to fail")

    asyncio.run(run())


def test_run_episode_rejects_unknown_turn_id_response():
    async def run():
        try:
            await run_episode(HappyPathEnv(), {"example_id": 1}, 0, FakeBinder(), WrongRequestIdSampler())
        except ValueError as exc:
            assert "Mismatched turn_id in response" in str(exc)
        else:
            raise AssertionError("Expected unknown turn_id to fail")

    asyncio.run(run())


def test_run_episode_rejects_mismatched_member_response():
    async def run():
        try:
            await run_episode(HappyPathEnv(), {"example_id": 1}, 0, FakeBinder(), WrongMemberSampler())
        except ValueError as exc:
            assert "Response member_id does not match" in str(exc)
        else:
            raise AssertionError("Expected mismatched member_id to fail")

    asyncio.run(run())


def test_run_episode_rejects_response_for_wrong_episode():
    async def run():
        try:
            await run_episode(HappyPathEnv(), {"example_id": 1}, 0, FakeBinder(), WrongEpisodeResponseSampler())
        except ValueError as exc:
            assert "Response episode_id does not match" in str(exc)
        else:
            raise AssertionError("Expected mismatched response episode id to fail")

    asyncio.run(run())


def test_run_episode_group_runs_multiple_samples():
    async def run():
        outputs = await run_episode_group(
            env=HappyPathEnv(),
            example={"example_id": 1},
            rollouts_per_example=2,
            policy_binder=FakeBinder(),
            sample_member=FakeSampler(),
        )
        assert [output.episode_id for output in outputs] == ["episode-0", "episode-1"]

    asyncio.run(run())


def test_run_episode_group_rejects_duplicate_episode_ids():
    async def run():
        try:
            await run_episode_group(
                env=DuplicateSampleIdEnv(),
                example={"example_id": 1},
                rollouts_per_example=2,
                policy_binder=FakeBinder(),
                sample_member=FakeSampler(),
            )
        except ValueError as exc:
            assert "Duplicate episode_id" in str(exc)
        else:
            raise AssertionError("Expected duplicate episode_id to fail")

    asyncio.run(run())


def test_run_episode_rejects_missing_member_outputs():
    async def run():
        try:
            await run_episode(MissingMemberOutputEnv(), {"example_id": 1}, 0, FakeBinder(), FakeSampler())
        except ValueError as exc:
            assert "member ids do not match session members" in str(exc)
        else:
            raise AssertionError("Expected missing member outputs to fail")

    asyncio.run(run())


def test_run_episode_rejects_wrong_base_example_id():
    async def run():
        try:
            await run_episode(WrongBaseExampleOutputEnv(), {"example_id": 1}, 0, FakeBinder(), FakeSampler())
        except ValueError as exc:
            assert "base_example_id does not match" in str(exc)
        else:
            raise AssertionError("Expected wrong base_example_id to fail")

    asyncio.run(run())
