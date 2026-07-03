import pytest

pytest.importorskip("libero")

from lerobot.envs.libero import (
    LiberoEnv,  # noqa: E402
    create_libero_envs,  # noqa: E402
)


class _FakeTask:
    def __init__(self, name: str, language: str):
        self.name = name
        self.language = language
        self.problem_folder = "fake_problem"
        self.init_states_file = "fake_init_states.pruned_init"
        self.bddl_file = "fake.bddl"


class _FakeTaskSuite:
    def __init__(self, languages: list[str]):
        self.tasks = [_FakeTask(f"task_{i}", lang) for i, lang in enumerate(languages)]

    def get_task(self, task_id: int) -> _FakeTask:
        return self.tasks[task_id]


def _make_libero_env(task_id: int, languages: list[str], task_language_override: str | None) -> LiberoEnv:
    return LiberoEnv(
        task_suite=_FakeTaskSuite(languages),
        task_id=task_id,
        task_suite_name="libero_spatial",
        init_states=False,
        task_language_override=task_language_override,
    )


def test_task_description_falls_back_to_libero_language_by_default():
    env = _make_libero_env(0, ["pick up the bowl"], task_language_override=None)
    assert env.task_description == "pick up the bowl"


def test_task_description_uses_override_when_provided():
    env = _make_libero_env(0, ["pick up the bowl"], task_language_override="nhấc cái bát lên")
    assert env.task_description == "nhấc cái bát lên"


def test_create_libero_envs_threads_per_task_overrides(monkeypatch):
    fake_suite = _FakeTaskSuite(["pick up the bowl", "open the drawer"])
    monkeypatch.setattr("lerobot.envs.libero._get_suite", lambda name: fake_suite)

    out = create_libero_envs(
        task="libero_spatial",
        n_envs=1,
        init_states=False,
        env_cls=list,
        gym_kwargs={"task_language_overrides": {"libero_spatial": {"0": "nhấc cái bát lên"}}},
    )

    env_task0 = out["libero_spatial"][0][0]()
    assert env_task0.task_description == "nhấc cái bát lên"

    env_task1 = out["libero_spatial"][1][0]()
    assert env_task1.task_description == "open the drawer"
