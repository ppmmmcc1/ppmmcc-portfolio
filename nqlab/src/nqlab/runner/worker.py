"""Worker subprocess entry point: ``python -m nqlab.runner.worker <job_id>``."""

from __future__ import annotations

import os
import sys
import traceback


def main(argv: list[str]) -> int:
    from nqlab.log import setup_logging
    from nqlab.results.ledger import LedgerEntry, TrialLedger
    from nqlab.runner.jobs import JOBS
    from nqlab.runner.queue import JobQueue
    from nqlab.settings import get_settings

    setup_logging()
    jid = argv[0]
    n_jobs = int(argv[1]) if len(argv) > 1 else 1
    settings = get_settings()
    q = JobQueue(settings.path("queue_db"))
    job = q.get(jid)
    q.start(jid, os.getpid())
    try:
        result = JOBS[job["type"]](settings, q.get(jid), q, n_jobs)
    except Exception as exc:  # crash isolation: log as errored in the queue AND the ledger
        tb = traceback.format_exc()
        q.fail(jid, tb)
        try:
            p = job["payload"]
            TrialLedger(settings.path("results_db")).append([LedgerEntry(
                f"job:{jid}", "diagnostic", "errored", family_id=p.get("family_id"), strategy=p.get("strategy"),
                params=p, note=f"queue job {job['type']} errored: {type(exc).__name__}: {exc}"[:500])])
        except Exception:
            pass
        print(tb, file=sys.stderr)
        return 2
    q.finish(jid, result)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
