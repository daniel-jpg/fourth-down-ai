import contextlib
import importlib.util
import io

import numpy as np
import pandas as pd


# =========================================================
# Load the frozen decision engine quietly.
# =========================================================

buffer = io.StringIO()

with contextlib.redirect_stdout(buffer):

    spec = importlib.util.spec_from_file_location(
        "fourth_down_engine",
        "src/42_build_decision_engine.py",
    )

    engine = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        engine
    )


# =========================================================
# Input helpers
# =========================================================

def ask_int(
    prompt,
    default,
    minimum=None,
    maximum=None,
):

    while True:

        raw = input(
            f"{prompt} [{default}]: "
        ).strip()

        if raw == "":
            value = int(default)

        else:

            try:
                value = int(raw)

            except ValueError:
                print(
                    "Please enter an integer."
                )
                continue


        if (
            minimum is not None
            and
            value < minimum
        ):

            print(
                f"Minimum is {minimum}."
            )
            continue


        if (
            maximum is not None
            and
            value > maximum
        ):

            print(
                f"Maximum is {maximum}."
            )
            continue


        return value


def ask_float(
    prompt,
    default,
    minimum=None,
    maximum=None,
):

    while True:

        raw = input(
            f"{prompt} [{default}]: "
        ).strip()

        if raw == "":
            value = float(default)

        else:

            try:
                value = float(raw)

            except ValueError:
                print(
                    "Please enter a number."
                )
                continue


        if (
            minimum is not None
            and
            value < minimum
        ):

            print(
                f"Minimum is {minimum}."
            )
            continue


        if (
            maximum is not None
            and
            value > maximum
        ):

            print(
                f"Maximum is {maximum}."
            )
            continue


        return value


def ask_yes_no(
    prompt,
    default=True,
):

    default_text = (
        "Y"
        if default
        else "N"
    )

    while True:

        raw = input(
            f"{prompt} [{default_text}]: "
        ).strip().lower()


        if raw == "":

            return default


        if raw in [
            "y",
            "yes",
        ]:

            return True


        if raw in [
            "n",
            "no",
        ]:

            return False


        print(
            "Enter y or n."
        )


def ask_roof():

    valid = {
        "1": "outdoors",
        "2": "dome",
        "3": "closed",
        "4": "open",
    }


    print(
        "\nRoof:"
    )

    print(
        "  1. outdoors"
    )

    print(
        "  2. dome"
    )

    print(
        "  3. closed"
    )

    print(
        "  4. open"
    )


    while True:

        raw = input(
            "Choose roof [1]: "
        ).strip()


        if raw == "":
            raw = "1"


        if raw in valid:

            return valid[
                raw
            ]


        print(
            "Enter 1, 2, 3, or 4."
        )


# =========================================================
# Clock conversion
# =========================================================

def game_seconds_from_clock(
    qtr,
    minutes,
    seconds,
):

    quarter_seconds = (
        minutes * 60
        +
        seconds
    )


    return float(

        (
            4 - qtr
        )
        *
        900

        +

        quarter_seconds

    )


# =========================================================
# Display helpers
# =========================================================

def display_action_table(
    result,
):

    results = (
        result[
            "results"
        ]
        .copy()
    )


    production = results[
        results[
            "eligible"
        ]
    ].copy()


    production[
        "Expected WP"
    ] = (

        100.0
        *
        production[
            "expected_win_probability"
        ]

    ).round(2)


    production[
        "MC SE"
    ] = (

        100.0
        *
        production[
            "mc_se"
        ]

    ).round(3)


    production = production.rename(
        columns={
            "decision":
                "Action",
            "detail":
                "Model detail",
        }
    )


    print(
        "\nPRODUCTION ACTION RANKING"
    )

    print(
        production[[
            "Action",
            "Expected WP",
            "MC SE",
            "Model detail",
        ]]
        .to_string(
            index=False
        )
    )


    experimental = results[
        ~results[
            "eligible"
        ]
    ].copy()


    if len(
        experimental
    ) > 0:

        experimental = (
            experimental[
                experimental[
                    "action"
                ]
                .str.startswith(
                    "FAKE_"
                )
            ]
        )


    if len(
        experimental
    ) > 0:

        print(
            "\nEXPERIMENTAL / NON-PRODUCTION ACTIONS"
        )

        for _, row in (
            experimental.iterrows()
        ):

            print(
                f"  {row['decision']}: "
                f"{row['detail']}"
            )


# =========================================================
# Decision interpretation
# =========================================================

GO_PLAYCALL_CLOSE_THRESHOLD_PP = 1.0


def print_decision_interpretation(
    results,
):

    production = (
        results[
            results["eligible"]
        ]
        .copy()
    )

    production = production[
        ~production[
            "action"
        ]
        .str.startswith(
            "FAKE_"
        )
    ]


    go = (
        production[
            production[
                "action"
            ]
            .isin([
                "NORMAL_GO_PASS",
                "NORMAL_GO_RUN",
            ])
        ]
        .sort_values(
            "expected_win_probability",
            ascending=False,
        )
    )


    strategy_rows = []


    if len(go) > 0:

        strategy_rows.append({
            "strategy":
                "GO FOR IT",

            "expected_win_probability":
                float(
                    go.iloc[0][
                        "expected_win_probability"
                    ]
                ),
        })


    for action, label in [
        (
            "FIELD_GOAL",
            "FIELD GOAL",
        ),
        (
            "PUNT",
            "PUNT",
        ),
    ]:

        rows = production[
            production[
                "action"
            ]
            == action
        ]

        if len(rows) > 0:

            strategy_rows.append({
                "strategy":
                    label,

                "expected_win_probability":
                    float(
                        rows.iloc[0][
                            "expected_win_probability"
                        ]
                    ),
            })


    strategies = (
        pd.DataFrame(
            strategy_rows
        )
        .sort_values(
            "expected_win_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    if len(strategies) == 0:

        return


    best = strategies.iloc[0]

    print(
        "\nRECOMMENDATION:"
    )

    print(
        best["strategy"]
    )
    print(
    f"Expected WP: "
    f"{100.0 * best['expected_win_probability']:.2f}%"
)


    if len(strategies) >= 2:

        second = (
            strategies.iloc[1]
        )

        strategy_edge = (
            100.0
            *
            (
                best[
                    "expected_win_probability"
                ]
                -
                second[
                    "expected_win_probability"
                ]
            )
        )


        print(
            f"\nADVANTAGE OVER "
            f"{second['strategy']}:"
        )

        print(
            f"+{strategy_edge:.2f} pp"
        )


        if strategy_edge < 0.5:

            print(
                "Interpretation: "
                "extremely close strategic call."
            )

        elif strategy_edge < 1.0:

            print(
                "Interpretation: "
                "close strategic call."
            )

        elif strategy_edge < 2.0:

            print(
                "Interpretation: "
                "moderate model preference."
            )

        else:

            print(
                "Interpretation: "
                "stronger model preference."
            )


    if (
        best["strategy"]
        == "GO FOR IT"
        and
        len(go) >= 2
    ):

        pass_rows = go[
            go["action"]
            == "NORMAL_GO_PASS"
        ]

        run_rows = go[
            go["action"]
            == "NORMAL_GO_RUN"
        ]


        if (
            len(pass_rows) > 0
            and
            len(run_rows) > 0
        ):

            pass_wp = float(
                pass_rows.iloc[0][
                    "expected_win_probability"
                ]
            )

            run_wp = float(
                run_rows.iloc[0][
                    "expected_win_probability"
                ]
            )

            playcall_edge = (
                100.0
                *
                abs(
                    pass_wp
                    -
                    run_wp
                )
            )


            print(
                "\nPLAY CALL:"
            )


            if pass_wp > run_wp:

                preferred_call = (
                    "PASS/DROPBACK"
                )

            else:

                preferred_call = (
                    "RUN"
                )


            if (
                playcall_edge
                <
                GO_PLAYCALL_CLOSE_THRESHOLD_PP
            ):

                print(
                    f"Slight lean: "
                    f"{preferred_call} "
                    f"(+{playcall_edge:.2f} pp)"
                )

                print(
                    "Run vs. pass is too close "
                    "to distinguish reliably."
                )

            else:

                if pass_wp > run_wp:

                    preferred_call = (
                        "PASS/DROPBACK"
                    )

                    other_call = (
                        "RUN"
                    )

                else:

                    preferred_call = (
                        "RUN"
                    )

                    other_call = (
                        "PASS/DROPBACK"
                    )


                print(
                    f"Model leans "
                    f"{preferred_call} over "
                    f"{other_call} by "
                    f"{playcall_edge:.2f} pp."
                )


# =========================================================
# Main interactive demo
# =========================================================

def main():

    print(
        "\n=============================================="
    )

    print(
        "       NFL FOURTH-DOWN DECISION AI"
    )

    print(
        "=============================================="
    )


    print(
        "\nEnter the current fourth-down situation."
    )

    print(
        "Press Enter to use the example defaults."
    )


    # -----------------------------------------------------
    # Default:
    # 4th-and-2 at opponent 38
    # Q4 7:00
    # offense down 3
    # -----------------------------------------------------

    qtr = ask_int(
        "\nQuarter",
        4,
        minimum=1,
        maximum=4,
    )


    minutes = ask_int(
        "Minutes remaining in quarter",
        7,
        minimum=0,
        maximum=15,
    )


    seconds = ask_int(
        "Seconds remaining",
        0,
        minimum=0,
        maximum=59,
    )


    ydstogo = ask_float(
        "\nYards to go",
        2,
        minimum=0.1,
        maximum=99,
    )


    print(
        "\nField position uses yards from the "
        "opponent end zone."
    )

    print(
        "Examples:"
    )

    print(
        "  opponent 10 -> 10"
    )

    print(
        "  opponent 38 -> 38"
    )

    print(
        "  midfield    -> 50"
    )

    print(
        "  own 25      -> 75"
    )


    yardline_100 = ask_float(
        "Yards from opponent end zone",
        38,
        minimum=1,
        maximum=99,
    )


    score_differential = ask_int(
        "\nCurrent offense score differential",
        -3,
        minimum=-60,
        maximum=60,
    )


    posteam_timeouts = ask_int(
        "Offense timeouts remaining",
        3,
        minimum=0,
        maximum=3,
    )


    defteam_timeouts = ask_int(
        "Defense timeouts remaining",
        3,
        minimum=0,
        maximum=3,
    )


    offense_is_home = ask_yes_no(
        "\nIs the current offense the HOME team?",
        True,
    )


    roof = ask_roof()


    game_seconds = (
        game_seconds_from_clock(
            qtr,
            minutes,
            seconds,
        )
    )


    state = {

        "qtr":
            qtr,

        "game_seconds_remaining":
            game_seconds,

        "yardline_100":
            yardline_100,

        "ydstogo":
            ydstogo,

        # From current offense's perspective.
        "score_differential":
            score_differential,

        "posteam_timeouts_remaining":
            posteam_timeouts,

        "defteam_timeouts_remaining":
            defteam_timeouts,

        "is_home":
            int(
                offense_is_home
            ),

        "roof":
            roof,
    }


    print(
        "\nRunning 4,000 simulations per "
        "eligible action..."
    )


    result = engine.recommend(

        state,

        n_simulations=4000,

        seed=42,
    )


    print(
        "\n=============================================="
    )

    print(
        "                 RESULT"
    )

    print(
        "=============================================="
    )


    print(
        "\nSituation:"
    )

    print(
        f"4th-and-{ydstogo:g}, "
        f"yardline_100={yardline_100:g}, "
        f"Q{qtr} "
        f"{minutes}:{seconds:02d}"
    )


    if score_differential > 0:

        score_text = (
            f"offense leads by "
            f"{score_differential}"
        )

    elif score_differential < 0:

        score_text = (
            f"offense trails by "
            f"{abs(score_differential)}"
        )

    else:

        score_text = (
            "game tied"
        )


    print(
        score_text
    )


    print(
        "\nCurrent model WP:"
    )

    print(
        f"{100 * result['pre_decision_wp']:.2f}%"
    )


    print_decision_interpretation(
        result[
            "results"
        ]
    )

    display_action_table(
        result
    )


    print(
        "\nNotes:"
    )

    print(
        "  - Expected WP is model-estimated."
    )

    print(
        "  - MC SE measures simulation noise only, "
        "not total model uncertainty."
    )

    print(
        "  - Fake punts and fake field goals are "
        "experimental-only."
    )

    print(
        "  - Kicker-specific context defaults to "
        "the development-data median unless supplied "
        "programmatically."
    )

    print(
        "  - This is a decision-support research model, "
        "not a causal guarantee."
    )


if __name__ == "__main__":

    main()