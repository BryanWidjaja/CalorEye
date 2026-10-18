import argparse

from caloreye.data.triage_probe import main as probe_main

def main(argv: list[str] | None = None) -> list[str]:
    parser = argparse.ArgumentParser(prog="caloreye data triage")
    parser.add_argument("step", choices=["probe", "score", "agree", "sheets", "calibration-sheets", "decide", "apply", "undo"])
    arguments, remaining = parser.parse_known_args(argv)

    if arguments.step == "probe":
        return probe_main(["probe", *remaining])

    if arguments.step == "sheets":
        from caloreye.data.triage_sheets import main as sheets_main

        return sheets_main(remaining)

    if arguments.step == "calibration-sheets":
        from caloreye.data.triage_calibrate import main as calibrate_main

        return calibrate_main(remaining)

    if arguments.step == "agree":
        from caloreye.data.triage_agree import main as agree_main

        return agree_main(remaining)

    if arguments.step == "decide":
        from caloreye.data.triage_decide import main as decide_main

        return decide_main(remaining)

    if arguments.step in ("apply", "undo"):
        from caloreye.data.triage_apply import main as apply_main

        return apply_main([arguments.step, *remaining])

    from caloreye.data.triage_score import main as score_main

    return score_main(remaining)
