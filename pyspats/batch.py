"""Independent field/trait jobs with explicit failure reporting."""

from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor
from .core import SpATS, fit_trial


@dataclass
class TrialFit:
    """One job's identity, fitted model (if successful), and error (if failed)."""

    key: tuple
    response: str
    model: SpATS | None = None
    error: str | None = None

    @property
    def ok(self):
        return self.error is None and self.model is not None and self.model.converged


def _run(task):
    key, response, data, options, on_error = task
    try:
        return TrialFit(
            key,
            response,
            fit_trial(
                data=data,
                response=response,
                environment_id=key[0] if len(key) == 1 else key,
                **options,
            ),
        )
    except Exception as exc:
        if on_error == "raise":
            raise
        return TrialFit(key, response, error=f"{type(exc).__name__}: {exc}")


def fit_trials(*, data, by, responses, workers=1, on_error="raise", **options):
    """Yield independent fits by field identifier and trait, in input group order.

    ``by`` and ``responses`` accept a column name or list of names. Missing
    field identifiers raise an error. ``workers=1`` streams jobs sequentially;
    larger values use processes, with at most ``2*workers`` jobs submitted at
    once. Use a main guard when calling from scripts on spawn-based platforms.
    Control BLAS threads in the job environment to avoid oversubscription.

    ``on_error='record'`` yields error text alongside successful results;
    numerical nonconvergence is represented by ``record.ok == False`` while
    retaining the model and iteration history. No rows or failed jobs vanish.
    The model's environment_id is set from the by-column values (a tuple for
    multiple columns); do not pass environment_id separately. These are separate
    single-field models, not a joint multi-environment analysis.
    """
    if "environment_id" in options:
        raise ValueError(
            "fit_trials derives environment_id from by; do not supply it separately"
        )
    by = [by] if isinstance(by, str) else list(by)
    responses = [responses] if isinstance(responses, str) else list(responses)
    if not by or not responses or len(set(responses)) != len(responses):
        raise ValueError("by and responses must be nonempty; responses must be unique")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be 'raise' or 'record'")
    if data[by].isna().any().any():
        raise ValueError("Missing field identifiers in by columns")
    missing = [c for c in responses if c not in data]
    if missing:
        raise ValueError(f"Missing response columns: {missing}")

    def jobs():
        for key, group in data.groupby(by, sort=False, observed=True):
            key = key if isinstance(key, tuple) else (key,)
            for response in responses:
                yield key, response, group, options, on_error

    if workers == 1:
        for task in jobs():
            yield _run(task)
    else:
        from collections import deque

        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = deque()
            source = iter(jobs())
            for task in source:
                pending.append(pool.submit(_run, task))
                if len(pending) >= 2 * workers:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()
