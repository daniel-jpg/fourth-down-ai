import math
import contextlib
import importlib.util
import io
import pandas as pd
import streamlit as st
import altair as alt

def format_field_position(yardline_100: int) -> str:
    if yardline_100 == 50:
        return "midfield"
    elif yardline_100 < 50:
        return f"opponent {yardline_100}"
    else:
        return f"own {100 - yardline_100}"


ROOF_LABELS = {
    "outdoors":
        "Outdoor / open-air",

    "dome":
        "Fixed dome",

    "closed":
        "Retractable roof — closed",

    "open":
        "Retractable roof — open",
}

# =========================================================
# App configuration
# =========================================================

st.set_page_config(
    page_title="NFL Fourth-Down Decision AI",
    page_icon="🏈",
    layout="wide",
)

# =========================================================
# Interactive football field component
# =========================================================

FIELD_HTML = """
<div class="field-shell">

    <div class="los-control">
        <label for="los-input">
            Line of scrimmage
        </label>

        <div class="los-input-row">
            <input
                id="los-input"
                type="number"
                min="1"
                max="99"
                step="1"
            >

            <span>
                yards from opponent end zone
            </span>
        </div>
    </div>

    <div
        id="field"
        class="football-field"
        aria-label="Football field position control"
    >

        <div class="yard-line goal" style="left:0%"></div>
        <div class="yard-line" style="left:10%"></div>
        <div class="yard-line" style="left:20%"></div>
        <div class="yard-line" style="left:30%"></div>
        <div class="yard-line" style="left:40%"></div>
        <div class="yard-line midfield" style="left:50%"></div>
        <div class="yard-line" style="left:60%"></div>
        <div class="yard-line" style="left:70%"></div>
        <div class="yard-line" style="left:80%"></div>
        <div class="yard-line" style="left:90%"></div>
        <div class="yard-line goal" style="left:100%"></div>

        <div class="yard-label left-edge" style="left:0%">G</div>
        <div class="yard-label" style="left:10%">10</div>
        <div class="yard-label" style="left:20%">20</div>
        <div class="yard-label" style="left:30%">30</div>
        <div class="yard-label" style="left:40%">40</div>
        <div class="yard-label" style="left:50%">50</div>
        <div class="yard-label" style="left:60%">40</div>
        <div class="yard-label" style="left:70%">30</div>
        <div class="yard-label" style="left:80%">20</div>
        <div class="yard-label" style="left:90%">10</div>
        <div class="yard-label right-edge" style="left:100%">G</div>

        <div
            id="first-down"
            class="first-down-line"
        ></div>

        <div
            id="ball"
            class="ball-marker"
            role="slider"
            tabindex="0"
            aria-label="Line of scrimmage"
            aria-valuemin="1"
            aria-valuemax="99"
        >
            &lt;
        </div>

    </div>

    <div class="field-info">
        <span id="direction-info"></span>
        <span id="field-status"></span>
    </div>

</div>
"""

FIELD_CSS = """
html,
body {
    margin: 0 !important;
    padding: 0 !important;
    overflow: hidden !important;
}

* {
    box-sizing: border-box;
}

.field-shell {
    width: 100%;
    max-width: 100%;
    overflow: hidden;
    padding: 0 8px;
    font-family: var(--st-font);
    color: var(--st-text-color);
}

.los-control {
    margin-bottom: 10px;
}

.los-control label {
    display: block;
    margin-bottom: 5px;
    font-size: 13px;
    font-weight: 600;
}

.los-input-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
}

.los-input-row input {
    width: 85px;
    padding: 7px 9px;
    border-radius: 6px;
    border: 1px solid var(--st-border-color);
    background: transparent;
    color: var(--st-text-color);
    font-size: 15px;
}

.los-input-row span {
    font-size: 12px;
    opacity: 0.65;
}

.football-field {
    position: relative;
    width: 100%;
    max-width: 100%;
    height: 120px;
    overflow: hidden;
    user-select: none;
    touch-action: none;
    cursor: crosshair;
}

.yard-line {
    position: absolute;
    top: 38px;
    bottom: 12px;
    width: 1px;
    background: color-mix(
        in srgb,
        var(--st-text-color) 30%,
        transparent
    );
}

.yard-line.midfield {
    width: 2px;
    background: color-mix(
        in srgb,
        var(--st-text-color) 55%,
        transparent
    );
}

.yard-line.goal {
    width: 3px;
    background: color-mix(
        in srgb,
        var(--st-text-color) 70%,
        transparent
    );
}

.yard-label {
    position: absolute;
    top: 5px;
    transform: translateX(-50%);
    font-size: 13px;
    font-weight: 600;
    opacity: 0.85;
}

.yard-label.left-edge {
    transform: none;
}

.yard-label.right-edge {
    transform: translateX(-100%);
}

.first-down-line {
    position: absolute;
    top: 36px;
    bottom: 10px;
    width: 4px;
    background: #f5c518;
    transform: translateX(-50%);
    border-radius: 3px;
    pointer-events: none;
}

.ball-marker {
    position: absolute;
    top: 68px;
    transform: translate(-50%, -50%);
    font-family: Menlo, Monaco, Consolas, monospace;
    font-size: 38px;
    font-weight: 900;
    line-height: 1;
    color: #ff4b4b;
    cursor: grab;
    padding: 4px;
    background: transparent;
    border: none;
    outline: none;
}

.ball-marker:focus-visible {
    outline: 2px solid #ff4b4b;
    outline-offset: 3px;
    border-radius: 4px;
}

.ball-marker.dragging {
    cursor: grabbing;
}

.field-info {
    display: flex;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 5px 16px;
    margin-top: 4px;
    font-size: 13px;
    opacity: 0.7;
    min-width: 0;
}

#field-status {
    overflow-wrap: anywhere;
}
"""

FIELD_JS = """
export default function(component) {

    const {
        parentElement,
        data,
        setStateValue
    } = component;

    const field =
        parentElement.querySelector("#field");

    const ball =
        parentElement.querySelector("#ball");

    const firstDown =
        parentElement.querySelector("#first-down");

    const directionInfo =
        parentElement.querySelector("#direction-info");

    const status =
        parentElement.querySelector("#field-status");

    const losInput =
        parentElement.querySelector("#los-input");


    const clamp = (
        value,
        low,
        high
    ) => {

        return Math.max(
            low,
            Math.min(
                high,
                value
            )
        );
    };


    const formatPosition = (
        yardline
    ) => {

        yardline =
            Math.round(
                yardline
            );

        if (
            yardline === 50
        ) {
            return "midfield";
        }

        if (
            yardline < 50
        ) {
            return (
                `opponent ${yardline}`
            );
        }

        return (
            `own ${100 - yardline}`
        );
    };


    let yardline = clamp(
        Math.round(
            Number(
                data?.yardline_100 ?? 30
            )
        ),
        1,
        99
    );


    const yardsToGo =
        Math.max(
            0.1,
            Number(
                data?.ydstogo ?? 1
            )
        );


    const render = () => {

        const firstDownYardline =
            Math.max(
                0,
                yardline
                -
                yardsToGo
            );
        const fgDistance =
    yardline + 18;


    if (fgDistance <= 60) {

        directionInfo.textContent =
            `← offense attacking left`
            +
            ` • Estimated FG: ${fgDistance} yd`;

    } else if (fgDistance <= 67) {

        directionInfo.textContent =
            `← offense attacking left`
            +
            ` • Estimated FG: ${fgDistance} yd`
            +
            ` — extreme distance (sparse-data tail)`;

    } else if (fgDistance <= 70) {

        directionInfo.textContent =
            `← offense attacking left`
            +
            ` • Estimated FG: ${fgDistance} yd`
            +
            ` — extreme distance (outside observed support)`;

    } else {

        directionInfo.textContent =
            `← offense attacking left`
            +
            ` • Estimated FG: ${fgDistance} yd`
            +
            ` — outside model range`;
    }

        ball.style.left =
            `${yardline}%`;

        firstDown.style.left =
            `${firstDownYardline}%`;

        losInput.value =
            yardline;

        ball.setAttribute(
            "aria-valuenow",
            yardline
        );


        let firstDownText;

        if (
            firstDownYardline <= 0
        ) {

            firstDownText =
                "goal line";

        } else {

            firstDownText =
                formatPosition(
                    firstDownYardline
                );
        }


        status.textContent =
            `Ball: ${formatPosition(yardline)}`
            +
            ` • Line to gain: ${firstDownText}`;
    };


    const commit = (
        newYardline
    ) => {

        yardline = clamp(
            Math.round(
                Number(
                    newYardline
                )
            ),
            1,
            99
        );

        render();

        setStateValue(
            "yardline_100",
            yardline
        );
    };


    const yardlineFromPointer = (
        event
    ) => {

        const rect =
            field.getBoundingClientRect();

        const fraction =
            (
                event.clientX
                -
                rect.left
            )
            /
            rect.width;

        return clamp(
            Math.round(
                fraction
                *
                100
            ),
            1,
            99
        );
    };


    let dragging = false;


    ball.onpointerdown = (
        event
    ) => {

        dragging = true;

        ball.classList.add(
            "dragging"
        );

        ball.setPointerCapture(
            event.pointerId
        );
    };


    ball.onpointermove = (
        event
    ) => {

        if (
            !dragging
        ) {
            return;
        }

        yardline =
            yardlineFromPointer(
                event
            );

        render();
    };


    ball.onpointerup = (
        event
    ) => {

        if (
            !dragging
        ) {
            return;
        }

        dragging = false;

        ball.classList.remove(
            "dragging"
        );

        commit(
            yardlineFromPointer(
                event
            )
        );
    };


    ball.onpointercancel = () => {

        dragging = false;

        ball.classList.remove(
            "dragging"
        );
    };


    field.onclick = (
        event
    ) => {

        if (
            event.target === ball
        ) {
            return;
        }

        commit(
            yardlineFromPointer(
                event
            )
        );
    };


    losInput.oninput = () => {

        const value =
            Number(
                losInput.value
            );

        if (
            Number.isFinite(
                value
            )
        ) {

            yardline = clamp(
                Math.round(
                    value
                ),
                1,
                99
            );

            render();
        }
    };


    losInput.onchange = () => {

        commit(
            losInput.value
        );
    };


    losInput.onkeydown = (
        event
    ) => {

        if (
            event.key === "Enter"
        ) {

            commit(
                losInput.value
            );

            losInput.blur();
        }
    };


    ball.onkeydown = (
        event
    ) => {

        if (
            event.key !==
            "ArrowLeft"
            &&
            event.key !==
            "ArrowRight"
        ) {
            return;
        }

        event.preventDefault();

        if (
            event.key ===
            "ArrowLeft"
        ) {

            commit(
                yardline - 1
            );

        } else {

            commit(
                yardline + 1
            );
        }
    };


    render();
}
"""

football_field_component = (
    st.components.v2.component(
        "football_field_position",
        html=FIELD_HTML,
        css=FIELD_CSS,
        js=FIELD_JS,
    )
)


def draggable_football_field(
    ydstogo,
    default=30,
    key="football_field",
):

    component_state = (
        st.session_state.get(
            key,
            {},
        )
    )

    try:

        current_ball = int(
            component_state.get(
                "yardline_100",
                default,
            )
        )

    except AttributeError:

        current_ball = int(
            getattr(
                component_state,
                "yardline_100",
                default,
            )
        )


    result = football_field_component(
        data={
            "yardline_100":
                current_ball,

            "ydstogo":
                float(ydstogo),
        },

        default={
            "yardline_100":
                current_ball,
        },

        on_yardline_100_change=
            lambda: None,

        key=key,

        width="stretch",

        height=230,
    )


    return int(
        result.yardline_100
    )

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1200px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.10);
        padding: 18px;
        border-radius: 12px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.95rem;
    }

    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# =========================================================
# Load frozen decision engine
# =========================================================

@st.cache_resource
def load_engine():

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

    return engine


engine = load_engine()


# =========================================================
# Helpers
# =========================================================

def game_seconds_from_clock(
    qtr,
    minutes,
    seconds,
):

    return float(
        (4 - qtr) * 900
        + minutes * 60
        + seconds
    )

def parse_game_clock(
    clock_text,
):

    try:

        parts = (
            clock_text
            .strip()
            .split(":")
        )

        if len(parts) != 2:
            return None

        minutes = int(
            parts[0]
        )

        seconds = int(
            parts[1]
        )

        if not (
            0 <= minutes <= 15
        ):
            return None

        if not (
            0 <= seconds <= 59
        ):
            return None

        return (
            minutes,
            seconds,
        )

    except ValueError:

        return None

def build_strategy_summary(
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

            "wp":
                float(
                    go.iloc[0][
                        "expected_win_probability"
                    ]
                ),

            "mc_se":
                float(
                    go.iloc[0][
                        "mc_se"
                    ]
                ),

            "source_action":
                go.iloc[0][
                    "action"
                ],
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
            production["action"]
            == action
        ]

        if len(rows) > 0:

            strategy_rows.append({
                "strategy":
                    label,

                "wp":
                    float(
                        rows.iloc[0][
                            "expected_win_probability"
                        ]
                    ),

                "mc_se":
                    float(
                        rows.iloc[0][
                            "mc_se"
                        ]
                    ),

                "source_action":
                    action,
            })


    strategies = (
        pd.DataFrame(
            strategy_rows
        )
        .sort_values(
            "wp",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    best = strategies.iloc[0]

    second = (
        strategies.iloc[1]
        if len(strategies) >= 2
        else None
    )


    return (
        production,
        go,
        best,
        second,
    )



# =========================================================
# Adaptive Monte Carlo recommendation.
#
# Validation-derived procedure:
#
#   300 simulations/action
#   -> if top-two action gap < 1.96 combined MC SE,
#      rerun from scratch at 1,200 simulations/action.
#
# The fixed-budget engine API remains unchanged.
# =========================================================

ADAPTIVE_BASE_SIMULATIONS = 300
ADAPTIVE_HIGH_SIMULATIONS = 1200
ADAPTIVE_Z_THRESHOLD = 1.96

ADAPTIVE_BASE_SEED = 42
ADAPTIVE_HIGH_SEED = 43


def combined_mc_gap_z(
    best_value,
    best_se,
    second_value,
    second_se,
):

    values = [
        best_value,
        best_se,
        second_value,
        second_se,
    ]

    if not all(
        math.isfinite(float(value))
        for value in values
    ):

        return math.nan


    gap = float(
        best_value
        -
        second_value
    )


    combined_se = math.sqrt(
        float(best_se) ** 2
        +
        float(second_se) ** 2
    )


    if combined_se > 0.0:

        return (
            gap
            /
            combined_se
        )


    if gap > 0.0:

        return math.inf


    return 0.0


def action_gap_z(results):

    eligible = (
        results[
            results["eligible"]
        ]
        .copy()
    )


    eligible = eligible[
        ~eligible[
            "action"
        ]
        .str.startswith(
            "FAKE_"
        )
    ]


    eligible = (
        eligible
        .sort_values(
            "expected_win_probability",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    if len(eligible) < 2:

        return math.inf


    best = eligible.iloc[0]
    second = eligible.iloc[1]


    return combined_mc_gap_z(

        float(
            best[
                "expected_win_probability"
            ]
        ),

        float(
            best["mc_se"]
        ),

        float(
            second[
                "expected_win_probability"
            ]
        ),

        float(
            second["mc_se"]
        ),
    )


def strategy_gap_z(
    best,
    second,
):

    if second is None:

        return math.inf


    return combined_mc_gap_z(

        float(best["wp"]),
        float(best["mc_se"]),

        float(second["wp"]),
        float(second["mc_se"]),
    )


def go_playcall_gap_z(go):

    pass_row = go[
        go["action"]
        ==
        "NORMAL_GO_PASS"
    ]

    run_row = go[
        go["action"]
        ==
        "NORMAL_GO_RUN"
    ]


    if (
        len(pass_row) == 0
        or
        len(run_row) == 0
    ):

        return math.inf


    pass_row = pass_row.iloc[0]
    run_row = run_row.iloc[0]


    if (
        float(
            pass_row[
                "expected_win_probability"
            ]
        )
        >=
        float(
            run_row[
                "expected_win_probability"
            ]
        )
    ):

        first = pass_row
        second = run_row

    else:

        first = run_row
        second = pass_row


    return combined_mc_gap_z(

        float(
            first[
                "expected_win_probability"
            ]
        ),

        float(
            first["mc_se"]
        ),

        float(
            second[
                "expected_win_probability"
            ]
        ),

        float(
            second["mc_se"]
        ),
    )


def adaptive_recommend(
    state,
):

    initial = engine.recommend(

        state,

        n_simulations=
            ADAPTIVE_BASE_SIMULATIONS,

        seed=
            ADAPTIVE_BASE_SEED,
    )


    initial_z = action_gap_z(
        initial["results"]
    )


    escalate = (
        (
            not math.isfinite(
                initial_z
            )
        )
        or
        (
            initial_z
            <
            ADAPTIVE_Z_THRESHOLD
        )
    )


    if escalate:

        result = engine.recommend(

            state,

            n_simulations=
                ADAPTIVE_HIGH_SIMULATIONS,

            seed=
                ADAPTIVE_HIGH_SEED,
        )

        simulations = (
            ADAPTIVE_HIGH_SIMULATIONS
        )

    else:

        result = initial

        simulations = (
            ADAPTIVE_BASE_SIMULATIONS
        )


    result[
        "adaptive_escalated"
    ] = escalate

    result[
        "adaptive_simulations_per_action"
    ] = simulations

    result[
        "initial_action_gap_z"
    ] = initial_z


    return result


# =========================================================
# Header
# =========================================================

st.title(
    "🏈 NFL Fourth-Down Decision AI"
)

st.caption(
    "In-game decision support based only on information "
    "available at the moment of the fourth-down decision."
)


# =========================================================
# Input panel
# =========================================================

st.subheader(
    "Game Situation"
)


col1, col2, col3 = st.columns(3)


ot_format = None
ot_phase = None
ot_period = 1


with col1:

    qtr = st.radio(
        "Quarter",
        options=[
            1,
            2,
            3,
            4,
            5,
        ],
        index=3,
        horizontal=True,
        format_func=lambda value:
            "OT"
            if value == 5
            else str(value),
    )

    clock_text = st.text_input(
        "Time remaining (MM:SS)",
        value="07:38",
        max_chars=5,
    )

with col2:

    ydstogo = st.number_input(
        "Yards to go",
        min_value=1,
        max_value=99,
        value=3,
        step=1,
    )

    score_col1, score_col2 = st.columns(2)

    with score_col1:

        offense_score = st.number_input(
            "Offense score",
            min_value=0,
            max_value=100,
            value=24,
            step=1,
        )

    with score_col2:

        defense_score = st.number_input(
            "Defense score",
            min_value=0,
            max_value=100,
            value=27,
            step=1,
        )


    score_differential = (
        offense_score
        -
        defense_score
    )


    if score_differential > 0:

        score_status = (
            f"Offense leading by "
            f"{score_differential}"
        )

    elif score_differential < 0:

        score_status = (
            f"Offense trailing by "
            f"{abs(score_differential)}"
        )

    else:

        score_status = (
            "Game tied"
        )


    st.caption(
        score_status
    )


if qtr == 5:

    (
        ot_format_col,
        ot_possession_col,
    ) = st.columns(
        [1, 2]
    )

    with ot_format_col:

        ot_format = st.radio(
            "Overtime format",
            options=[
                "REGULAR_SEASON",
                "POSTSEASON",
            ],
            format_func=lambda value: {
                "REGULAR_SEASON":
                    "Regular season",
                "POSTSEASON":
                    "Playoffs",
            }[value],
            horizontal=True,
        )

    with ot_possession_col:

        ot_phase = st.radio(
            "OT possession",
            options=[
                "OPENING",
                "RESPONSE",
                "SUDDEN_DEATH",
            ],
            format_func=lambda value: {
                "OPENING":
                    "Opening possession",
                "RESPONSE":
                    "Response possession",
                "SUDDEN_DEATH":
                    "Sudden death",
            }[value],
            horizontal=True,
        )


with col3:

    regular_season_ot = (
        qtr == 5
        and
        ot_format == "REGULAR_SEASON"
    )

    timeout_options = (
        [0, 1, 2]
        if regular_season_ot
        else [0, 1, 2, 3]
    )

    timeout_default_index = (
        2
        if regular_season_ot
        else 3
    )

    offense_timeouts = st.radio(
        "Offense timeouts",
        options=timeout_options,
        index=timeout_default_index,
        horizontal=True,
    )

    defense_timeouts = st.radio(
        "Defense timeouts",
        options=timeout_options,
        index=timeout_default_index,
        horizontal=True,
    )


parsed_clock = (
    parse_game_clock(
        clock_text
    )
)


clock_valid = (
    parsed_clock is not None
)


if clock_valid:

    minutes, seconds = (
        parsed_clock
    )

    if qtr == 5:

        ot_clock_limit = (
            600
            if ot_format == "REGULAR_SEASON"
            else 900
        )

        if (
            (
                minutes * 60
                +
                seconds
            )
            >
            ot_clock_limit
        ):

            clock_valid = False

            if ot_format == "REGULAR_SEASON":

                st.error(
                    "Regular-season OT clock "
                    "cannot exceed 10:00."
                )

            else:

                st.error(
                    "Playoff OT clock "
                    "cannot exceed 15:00."
                )

else:

    minutes = 0
    seconds = 0

    st.error(
        "Enter the clock as MM:SS, "
        "for example 07:38."
    )


situation_valid = (
    clock_valid
)


if (
    clock_valid
    and
    qtr == 5
):

    ot_seconds_remaining = (
        minutes * 60
        +
        seconds
    )

    if ot_seconds_remaining == 0:

        situation_valid = False

        st.error(
            "A fourth-down decision cannot occur "
            "after the OT period clock has expired."
        )

    elif (
        ot_phase == "OPENING"
        and
        score_differential != 0
    ):

        situation_valid = False

        st.error(
            "The opening OT possession must begin "
            "with the score tied."
        )

    elif (
        ot_phase == "RESPONSE"
        and
        score_differential > 0
    ):

        situation_valid = False

        st.error(
            "The responding offense cannot already "
            "be leading in a live fourth-down state."
        )

    elif (
        ot_phase == "SUDDEN_DEATH"
        and
        score_differential != 0
    ):

        situation_valid = False

        st.error(
            "Sudden death can only have a live "
            "fourth-down decision while the game is tied."
        )


st.markdown(
    "#### Ball Position"
)

yardline_100 = (
    draggable_football_field(
        ydstogo=ydstogo,
        default=30,
        key="football_field",
    )
)

max_yards_to_go = max(
    1.0,
    float(yardline_100),
)

if (
    float(ydstogo)
    >
    max_yards_to_go
):

    situation_valid = False

    st.error(
        "Yards to go cannot exceed the distance "
        "to the opponent goal line."
    )


col4, col5 = st.columns(2)


with col4:

    game_site = st.radio(
        "Game site",
        options=[
            "Home",
            "Away",
            "Neutral",
        ],
        index=1,
        horizontal=True,
        help=(
            "Relative to the current offense. "
            "Neutral games have no home-field advantage."
        ),
    )


with col5:


    roof = st.selectbox(
        "Stadium / roof",
        options=list(
            ROOF_LABELS.keys()
        ),
        format_func=lambda value:
            ROOF_LABELS[value],
    )


second_half_receiver = None


if qtr == 2:

    second_half_receiver = st.radio(
        "Second-half kickoff receiver",
        options=[
            "Current offense",
            "Current defense",
        ],
        index=None,
        horizontal=True,
        help=(
            "Who is scheduled to receive the opening "
            "kickoff of the third quarter. This is known "
            "from the opening coin toss."
        ),
    )


    if second_half_receiver is None:

        situation_valid = False

        st.caption(
            "Select the second-half kickoff receiver."
        )


# =========================================================
# Optional modern-kickoff sensitivity analysis.
#
# This NEVER replaces the original production recommendation.
# All experimental evaluation happens inside a fresh, private
# engine module so concurrent Streamlit users cannot affect
# each other's production calculations.
# =========================================================

experimental_kickoff_enabled = False
experimental_use_2024_assumptions = False
experimental_touchback_rate = 0.30
experimental_onside_enabled = False
experimental_onside_attempt_probability = 0.25
experimental_onside_recovery_probability = 0.10
experimental_ordinary_only = False
experimental_native_candidate_enabled = False

if qtr == 4:

    with st.expander(
        "2025+ kickoff rule comparison (experimental)",
        expanded=False,
    ):

        st.caption(
            "Compare the official model with a separate simulation using "
            "observed 2025 NFL kickoff outcomes. This is an experimental "
            "scenario, not a validated replacement for the default model."
        )

        experimental_kickoff_enabled = st.checkbox(
            "Show the experimental kickoff-adjusted comparison",
            value=False,
            help=(
                "Defaults to frozen 2025 ordinary kickoffs AND verified "
                "penalty-affected first possessions. The preview is isolated "
                "from the production model and is not an independently "
                "validated 2026+ prediction."
            ),
        )

        if experimental_kickoff_enabled:

            st.caption(
                "Default: 2025 regular-season Q4 nonscoring kickoffs. "
                "538 verified first possessions: 488 ordinary kickoffs "
                "and 50 clean penalty-affected kickoffs. Field position "
                "and elapsed time are sampled together from real plays."
            )

            experimental_use_2024_assumptions = st.checkbox(
                "Advanced: use older 2024 return data with an assumed "
                "touchback rate instead",
                value=False,
            )

            if experimental_use_2024_assumptions:

                assumed_touchback_pct = st.select_slider(
                    "Assumed ordinary kickoff touchback rate",
                    options=[20, 25, 30, 35, 40],
                    value=30,
                    format_func=lambda value: f"{value}%",
                )

                experimental_touchback_rate = (
                    float(assumed_touchback_pct) / 100.0
                )

                st.caption(
                    "Legacy sensitivity mode only. This rate is a "
                    "what-if assumption, not a live forecast."
                )

            else:

                st.caption(
                    "No touchback or penalty rate needs to be guessed. "
                    "The selected 2025 historical outcome distribution "
                    "is not a calibrated 2026+ forecast."
                )

                experimental_ordinary_only = st.checkbox(
                    "Advanced: compare using only the previous 488 "
                    "ordinary kickoffs (exclude penalty-affected snaps)",
                    value=False,
                )

                experimental_native_candidate_enabled = st.checkbox(
                    "Advanced: also show native kickoff-input correction "
                    "(research-only)",
                    value=False,
                    help=(
                        "Uses the original frozen WP model with the native "
                        "Q4 post-score kickoff yardline/distance features. "
                        "It does not use 2025 first-snap donor sampling. "
                        "The official recommendation never changes."
                    ),
                )

                if experimental_native_candidate_enabled:

                    st.caption(
                        "Retrospective 2023–2025 post-score Q4 kicks "
                        "(1,617 observations) had a lower pooled Brier "
                        "score with this candidate (0.11561 to 0.10743), "
                        "but the away-kicking-team subgroup was slightly "
                        "worse. Historical data were previously examined; "
                        "this is not a validated replacement or causal "
                        "fourth-down policy test."
                    )

                if (
                    float(score_differential) + 3.0 < 0
                    and int(minutes)*60 + int(seconds) > 40
                ):

                    experimental_onside_enabled = st.checkbox(
                        "Advanced: include hypothetical onside kicks "
                        "after a made field goal",
                        value=False,
                        help=(
                            "Only applies if the kicking team would remain "
                            "behind after making the FG. Samples historical "
                            "2025 onside first-snap outcomes, but both "
                            "probabilities are explicit what-if assumptions."
                        ),
                    )

                    if experimental_onside_enabled:

                        st.caption(
                            "Hypothetical branch only. A 2025 observational "
                            "sample of 41 attempts included just four "
                            "kicking-team recoveries. Neither the chance "
                            "of attempting nor recovering is validated "
                            "for this particular game."
                        )

                        assumed_attempt_pct = st.select_slider(
                            "Assumed chance of choosing an onside kick "
                            "after a made field goal",
                            options=[0, 5, 10, 15, 20, 25, 30, 40, 50],
                            value=25,
                            format_func=lambda value: f"{value}%",
                        )

                        assumed_recovery_pct = st.select_slider(
                            "Assumed conditional kicking-team recovery chance",
                            options=[0, 5, 10, 15, 20, 30, 50],
                            value=10,
                            format_func=lambda value: f"{value}%",
                        )

                        experimental_onside_attempt_probability = (
                            float(assumed_attempt_pct) / 100.0
                        )
                        experimental_onside_recovery_probability = (
                            float(assumed_recovery_pct) / 100.0
                        )

                if not experimental_onside_enabled:

                    st.caption(
                        "Onside kicks, return touchdowns, actual re-kicks, "
                        "and unmatched first possessions remain excluded. "
                        "Enable the Advanced onside what-if to explore "
                        "one additional hypothetical branch."
                    )


# =========================================================
# Run model
# =========================================================

run_model = st.button(
    "Analyze Fourth Down",
    type="primary",
    width="stretch",
    disabled=not situation_valid,
)


if run_model:

    if qtr == 5:

        game_seconds = (
            int(minutes) * 60
            +
            int(seconds)
        )

    else:

        game_seconds = (
            game_seconds_from_clock(
                qtr,
                int(minutes),
                int(seconds),
            )
        )


    state = {

        "qtr":
            int(qtr),

        "game_seconds_remaining":
            game_seconds,

        "yardline_100":
            yardline_100,

        "ydstogo":
            float(ydstogo),

        "score_differential":
            float(
                score_differential
            ),

        "posteam_timeouts_remaining":
            float(
                offense_timeouts
            ),

        "defteam_timeouts_remaining":
            float(
                defense_timeouts
            ),

        "site":
            game_site.upper(),

        "roof":
            roof,
    }


    if qtr == 2:

        state[
            "second_half_receiver"
        ] = (
            "OFFENSE"
            if
            second_half_receiver
            ==
            "Current offense"
            else
            "DEFENSE"
        )


    if qtr == 5:

        state["ot_phase"] = (
            ot_phase
        )

        state["ot_format"] = (
            ot_format
        )

        state["ot_period"] = (
            ot_period
        )


    with st.spinner(
        "Running adaptive decision simulation..."
    ):

        result = adaptive_recommend(
            state
        )


    (
        production,
        go,
        best,
        second,
    ) = build_strategy_summary(
        result["results"]
    )


    strategy_z = strategy_gap_z(
        best,
        second,
    )


    # =====================================================
    # Situation
    # =====================================================

    st.divider()

    st.subheader(
        "Decision"
    )


    location_text = (
        format_field_position(
            yardline_100
        )
    )

    period_label = (
        "OT"
        if qtr == 5
        else f"Q{qtr}"
    )

    st.write(
        f"**4th & {ydstogo:g}** at "
        f"**{location_text}** · "
        f"{period_label} "
        f"{int(minutes)}:"
        f"{int(seconds):02d} · "
        f"{score_status}"
    )


    # =====================================================
    # Main recommendation
    # =====================================================

    if qtr == 5:

        value_metric_label = (
            "Game value if chosen"
        )

        value_phrase = (
            "estimated game value"
        )

        table_value_label = (
            "Expected game value (%)"
        )

        chart_expander_label = (
            "Show game-value chart"
        )

        chart_axis_label = (
            "Expected game value (%)"
        )

        chart_tooltip_label = (
            "Expected game value"
        )

    else:

        value_metric_label = (
            "Win probability if chosen"
        )

        value_phrase = (
            "estimated win probability"
        )

        table_value_label = (
            "Expected WP (%)"
        )

        chart_expander_label = (
            "Show win-probability chart"
        )

        chart_axis_label = (
            "Expected win probability (%)"
        )

        chart_tooltip_label = (
            "Expected WP"
        )


    recommendation_col, wp_col, edge_col = (
        st.columns(3)
    )


    with recommendation_col:

        st.metric(
            "Recommendation",
            best["strategy"],
        )


    with wp_col:

        st.metric(
            value_metric_label,
            f"{100 * best['wp']:.2f}%",
        )


    if second is not None:

        strategy_edge = (
            100
            *
            (
                best["wp"]
                -
                second["wp"]
            )
        )

        with edge_col:

            st.metric(
                f"Advantage over "
                f"{second['strategy']}",
                f"+{strategy_edge:.2f} pp",
            )


        relative_strategy_edge = (
            (
                best["wp"]
                -
                second["wp"]
            )
            /
            max(
                float(best["wp"]),
                1e-12,
            )
        )


        near_tie = (
            math.isfinite(
                strategy_z
            )
            and
            strategy_z
            <
            ADAPTIVE_Z_THRESHOLD
        )


        if near_tie:

            st.info(
                f"**Near tie:** "
                f"{best['strategy']} has the highest "
                f"{value_phrase}, but only "
                f"{strategy_edge:.2f} percentage points "
                f"above {second['strategy']}. "
                f"Monte Carlo uncertainty does not clearly "
                f"separate these choices."
            )

        else:

            st.info(
                f"**Model recommendation:** "
                f"{best['strategy']} has the highest "
                f"{value_phrase}, "
                f"{strategy_edge:.2f} percentage points "
                f"above {second['strategy']}."
            )

    if qtr != 5:

        st.caption(
            f"Win probability before the fourth-down decision: "
            f"{100 * result['pre_decision_wp']:.2f}%"
        )

    # =====================================================
    # Optional 2025+ kickoff rule scenario, visually separate
    # from the unchanged production recommendation above.
    # =====================================================

    if qtr == 4 and experimental_kickoff_enabled:

        st.divider()
        st.subheader(
            "2025+ kickoff-adjusted comparison (experimental)"
        )

        if experimental_use_2024_assumptions:

            st.caption(
                "Older 2024 kickoff returns, 2025+ touchback placement, "
                f"and a manually assumed {experimental_touchback_rate:.0%} "
                "touchback rate. Sensitivity analysis only."
            )

        else:

            st.caption(
                "Observed 2025 regular-season Q4 nonscoring "
                "kickoff-to-first-snap transitions, including verified "
                "penalty-enforced first possessions by default. "
                "No assumed touchback or penalty rates. Historical "
                "description, NOT a calibrated 2026+ forecast."
            )

        try:

            from src.streamlit_kickoff_preview import (
                preview_recommendation,
                preview_observed_2025_recommendation,
                preview_observed_2025_resolved_recommendation,
                preview_native_kickoff_feature_candidate,
                preview_observed_2025_with_onside,
                observed_2025_onside_summary,
                observed_2025_pool_summary,
                observed_2025_resolved_summary,
                observed_2025_kickoff_coverage,
                action_comparison,
                preview_strategy_rank_with_uncertainty,
                preview_go_playcall_with_uncertainty,
                recommendation_sensitivity,
            )

            with st.spinner(
                "Running a separate experimental kickoff scenario..."
            ):

                if experimental_use_2024_assumptions:

                    scenario_result = preview_recommendation(
                        state,
                        touchback_rate=experimental_touchback_rate,
                    )
                    frozen_2025_details = None
                    frozen_2025_coverage = None
                    frozen_onside_details = None

                else:

                    if experimental_onside_enabled:

                        scenario_result = preview_observed_2025_with_onside(
                            state,
                            attempt_probability=
                                experimental_onside_attempt_probability,
                            recovery_probability=
                                experimental_onside_recovery_probability,
                            include_verified_penalties=
                                not experimental_ordinary_only,
                        )
                        frozen_onside_details = observed_2025_onside_summary()

                    else:

                        if experimental_ordinary_only:

                            scenario_result = preview_observed_2025_recommendation(
                                state,
                            )

                        else:

                            scenario_result = preview_observed_2025_resolved_recommendation(
                                state,
                            )

                        frozen_onside_details = None

                    frozen_2025_details = (
                        observed_2025_pool_summary()
                        if experimental_ordinary_only
                        else observed_2025_resolved_summary()
                    )
                    frozen_2025_coverage = observed_2025_kickoff_coverage()

                if experimental_native_candidate_enabled:

                    native_candidate_result = preview_native_kickoff_feature_candidate(
                        state,
                    )
                    native_candidate_rank = preview_strategy_rank_with_uncertainty(
                        native_candidate_result
                    )
                    native_candidate_table = action_comparison(
                        result,
                        native_candidate_result,
                    )

                else:

                    native_candidate_result = None
                    native_candidate_rank = None
                    native_candidate_table = None

                scenario_table = action_comparison(
                    result,
                    scenario_result,
                )

                scenario_strategy_rank = preview_strategy_rank_with_uncertainty(
                    scenario_result
                )

                scenario_playcall_rank = preview_go_playcall_with_uncertainty(
                    scenario_result
                )

                scenario_sensitivity = recommendation_sensitivity(
                    result,
                    scenario_result,
                )

        except Exception as exc:

            st.warning(
                "The optional kickoff scenario was unavailable: "
                f"{exc}. The production recommendation above "
                "is unaffected."
            )

        else:

            if frozen_2025_details is not None:

                all_kicks = frozen_2025_coverage["all"]
                q4_late_kicks = frozen_2025_coverage["late"]
                exclusions = all_kicks["counts"]
                recorded = frozen_2025_details["kickoffs"]

                if experimental_ordinary_only:

                    st.caption(
                        f"**Previous ordinary-only comparison:** "
                        f"{recorded} matched kickoffs, "
                        f"{recorded/all_kicks['total_kickoffs']:.1%} of "
                        "eligible 2025 Q4 standard-35 events. "
                        "Penalty-affected plays are deliberately excluded."
                    )

                else:

                    st.caption(
                        f"**Expanded 2025 nonscoring sample:** "
                        f"{recorded} verified first possessions "
                        f"({frozen_2025_details['ordinary']} ordinary + "
                        f"{frozen_2025_details['penalty_first_snaps']} "
                        f"penalty-affected), from "
                        f"{frozen_2025_details['games']} games. "
                        f"Coverage: {recorded/all_kicks['total_kickoffs']:.1%} "
                        f"of all {all_kicks['total_kickoffs']} eligible "
                        "2025 Q4 standard-35 kickoffs."
                    )

                st.caption(
                    "**Still unsupported as ordinary nonscoring "
                    "first-possession outcomes:** "
                    f"{exclusions['onside_attempt']} onside attempts, "
                    f"{exclusions['kickoff_return_td']} return touchdown, "
                    "2 actual re-kicks, and "
                    f"{exclusions['other_unmatched']} unmatched plays. "
                    "One 2025 return touchdown at 7:18 remaining "
                    "cannot establish a future game-specific TD rate."
                )

                if frozen_onside_details is not None:

                    st.info(
                        "**Hypothetical onside sensitivity active:** "
                        f"you selected a "
                        f"{experimental_onside_attempt_probability:.0%} chance "
                        "of an onside attempt after a made FG while still "
                        f"trailing, and a "
                        f"{experimental_onside_recovery_probability:.0%} "
                        "kicking-team recovery chance *conditional on "
                        "attempting*. "
                        "Historical field positions and clock changes "
                        f"come from {frozen_onside_details['matched']} "
                        "verified 2025 first-snap transitions, including "
                        f"only {frozen_onside_details['recovered']} "
                        "successful recoveries. "
                        "**These input probabilities are assumptions, "
                        "not calibrated estimates.** "
                        "Verified penalty first possessions are included "
                        "unless the previous 488-only Advanced mode is active. "
                        "Return touchdowns, actual re-kicks and special "
                        "kickoff spots remain excluded."
                    )

                late_resolved = (
                    q4_late_kicks["counts"]["modeled_ordinary"]
                    + (
                        0 if experimental_ordinary_only
                        else frozen_2025_details["late_verified_penalties"]
                    )
                )

                st.caption(
                    f"In the final five minutes (>40 seconds remaining), "
                    f"the selected historical donor pool covers "
                    f"{late_resolved} of "
                    f"{q4_late_kicks['total_kickoffs']} "
                    f"({late_resolved/q4_late_kicks['total_kickoffs']:.1%}) "
                    "kickoffs. These selected historical rates are NOT "
                    "forecasts of future coaching or special-teams outcomes."
                )

                # For a successful 3-point FG, the OPPONENT receives.
                # Their post-score lead can be inferred without peeking
                # at actual future events or claiming a causal sample.
                receiving_margin_if_fg_made = -(
                    float(state["score_differential"]) + 3.0
                )
                if (
                    40 < float(state["game_seconds_remaining"]) <= 300
                    and 1 <= receiving_margin_if_fg_made <= 3
                ):

                    close_kicks = frozen_2025_coverage[
                        "receiver_leads_1to3"
                    ]
                    close_counts = close_kicks["counts"]

                    st.caption(
                        f"**Limited close-game support:** after a made field "
                        f"goal in this situation, the receiver would lead "
                        f"by {receiving_margin_if_fg_made:g}. Among 2025 "
                        f"Q4 kickoffs with a receiver ahead by 1–3 and "
                        f"40–300 seconds left, only "
                        f"{close_counts['modeled_ordinary']} of "
                        f"{close_kicks['total_kickoffs']} were ordinary "
                        "or clean nonscoring possessions; "
                        f"{close_counts['onside_attempt']} were onside. "
                        "This small selected sample does NOT establish "
                        "a game-specific onside attempt or recovery rate."
                    )

            # A RUN-vs-PASS near tie is about the play call inside
            # GO FOR IT. It is not a GO-vs-FG strategy near tie.
            if scenario_sensitivity["model_sensitive"]:

                st.warning(
                    "**Fourth-down strategy is sensitive to the "
                    "kickoff assumptions.** "
                    f"The official model favors "
                    f"{scenario_sensitivity['production_leader']}; "
                    f"the experiment's top strategy is "
                    f"{scenario_sensitivity['scenario_point_leader']}. "
                    "The order changes or the strategy-level margin "
                    "is within approximate Monte Carlo error. "
                    "Neither result has independent 2026+ validation."
                )

            if scenario_strategy_rank["near_tie"]:

                st.metric(
                    "Experimental strategy (not official)",
                    "NO CLEAR STRATEGY",
                )

                st.caption(
                    f"Highest strategy point estimate: "
                    f"{scenario_strategy_rank['leader']} at "
                    f"{scenario_strategy_rank['leader_wp_pct']:.2f}% "
                    f"(best GO action: "
                    f"{scenario_strategy_rank['leader_action']})."
                )

            else:

                st.metric(
                    "Experimental strategy (not official)",
                    scenario_strategy_rank["leader"],
                )

                st.caption(
                    f"Scenario-estimated win probability: "
                    f"{scenario_strategy_rank['leader_wp_pct']:.2f}%. "
                    f"Best supporting action: "
                    f"{scenario_strategy_rank['leader_action']}."
                )

            if scenario_strategy_rank["runner_up"] is not None:

                strategy_gap = scenario_strategy_rank["gap_pp"]

                if scenario_strategy_rank["near_tie"]:

                    st.warning(
                        f"**Strategy too close to call:** "
                        f"{scenario_strategy_rank['leader']} leads "
                        f"{scenario_strategy_rank['runner_up']} "
                        f"by only {strategy_gap:.2f} percentage points. "
                        "Approximate simulation uncertainty alone "
                        "prevents a clear strategy-level ranking. "
                        "That does not quantify model uncertainty."
                    )

                else:

                    st.info(
                        f"**Experimental strategy comparison:** "
                        f"{scenario_strategy_rank['leader']} leads "
                        f"{scenario_strategy_rank['runner_up']} "
                        f"by {strategy_gap:.2f} percentage points. "
                        "This is a scenario estimate, not an "
                        "independently validated decision edge."
                    )

            if (
                scenario_strategy_rank["leader"] == "GO FOR IT"
                and scenario_playcall_rank is not None
            ):

                play_call_gap = scenario_playcall_rank["gap_pp"]

                if scenario_playcall_rank["near_tie"]:

                    st.info(
                        f"**Separate play-call uncertainty:** "
                        f"{scenario_playcall_rank['leader']} leads "
                        f"{scenario_playcall_rank['runner_up']} "
                        f"by only {play_call_gap:.2f} percentage points. "
                        "The simulation cannot reliably distinguish "
                        "RUN from PASS within GO FOR IT. "
                        "This does **not** mean GO FOR IT and "
                        "FIELD GOAL are tied."
                    )

                else:

                    st.caption(
                        f"Within GO FOR IT, the experimental "
                        f"{scenario_playcall_rank['leader']} call "
                        f"leads {scenario_playcall_rank['runner_up']} "
                        f"by {play_call_gap:.2f} percentage points. "
                        "Play-call precision is not independently validated."
                    )

            st.dataframe(
                scenario_table.round(2),
                hide_index=True,
                width="stretch",
            )

            if native_candidate_rank is not None:

                st.markdown(
                    "**Additional model-input comparison (research-only)**"
                )

                st.caption(
                    "Unlike the 538-kickoff experiment above, this "
                    "candidate changes only two kickoff input fields "
                    "in the frozen WP model: yardline_100=35 and "
                    "ydstogo=0 for unresolved normal Q4 post-score "
                    "kickoffs. The original model and weights are "
                    "otherwise identical."
                )

                st.metric(
                    "Native kickoff-input candidate strategy",
                    (
                        "NO CLEAR STRATEGY"
                        if native_candidate_rank["near_tie"]
                        else native_candidate_rank["leader"]
                    ),
                )

                if native_candidate_rank["runner_up"] is not None:

                    st.caption(
                        f"Candidate top: {native_candidate_rank['leader']} "
                        f"at {native_candidate_rank['leader_wp_pct']:.2f}% "
                        f"(supporting play: "
                        f"{native_candidate_rank['leader_action']}); "
                        f"gap over {native_candidate_rank['runner_up']} "
                        f"{native_candidate_rank['gap_pp']:.2f} pp. "
                        "MC-only near-tie: "
                        f"{native_candidate_rank['near_tie']}. "
                        "These are research predictions, not validated "
                        "advantages."
                    )

                st.dataframe(
                    native_candidate_table.rename(columns={
                        "2025+ scenario WP (%)": "Native input WP (%)",
                        "Scenario MC SE (pp)": "Native MC SE (pp)",
                        "Scenario minus production (pp)":
                            "Native minus official (pp)",
                    }).round(2),
                    hide_index=True,
                    width="stretch",
                )

                st.caption(
                    "The 2023–2025 retrospective kickoff calibration "
                    "showed overall improvement but a small deterioration "
                    "for away kicking teams. Neither that study nor this "
                    "simulation establishes real-world fourth-down "
                    "counterfactuals. The official selection above "
                    "remains unchanged."
                )

            st.caption(
                "Scenario runs 2,400 simulations per action in a "
                "private engine instance. MC SE measures simulation "
                "noise only; it does not include model or kickoff "
                "assumption uncertainty. This is an experimental "
                "ordinary-kickoff or explicitly assumed onside sensitivity "
                "analysis, not a validated 2026+ policy or retrained WP model."
            )

    # =====================================================
    # GO play call
    # =====================================================

    if (
        best["strategy"]
        == "GO FOR IT"
        and
        len(go) >= 2
    ):

        pass_row = go[
            go["action"]
            == "NORMAL_GO_PASS"
        ]

        run_row = go[
            go["action"]
            == "NORMAL_GO_RUN"
        ]


        if (
            len(pass_row) > 0
            and
            len(run_row) > 0
        ):

            pass_wp = float(
                pass_row.iloc[0][
                    "expected_win_probability"
                ]
            )

            run_wp = float(
                run_row.iloc[0][
                    "expected_win_probability"
                ]
            )


            playcall_edge = (
                100
                *
                abs(
                    pass_wp
                    -
                    run_wp
                )
            )


            playcall_z = (
                go_playcall_gap_z(
                    go
                )
            )


            playcall_uncertain = (
                math.isfinite(
                    playcall_z
                )
                and
                playcall_z
                <
                ADAPTIVE_Z_THRESHOLD
            )


            if pass_wp > run_wp:

                preferred_call = (
                    "PASS / DROPBACK"
                )

            else:

                preferred_call = (
                    "RUN"
                )


            st.subheader(
                "Play Call"
            )


            if playcall_uncertain:

                st.info(
                    f"**Slight lean: "
                    f"{preferred_call}** "
                    f"(+{playcall_edge:.2f} pp)\n\n"
                    f"Monte Carlo uncertainty does not "
                    f"clearly separate run and pass."
                )

            else:

                st.success(
                    f"**{preferred_call}** "
                    f"has a "
                    f"{playcall_edge:.2f} pp "
                    f"model advantage."
                )


    # =====================================================
    # Action values
    # =====================================================

    action_lookup = {
        row["action"]: row
        for _, row in production.iterrows()
    }

    st.subheader(
        "Compare Actions"
    )

    a1, a2, a3, a4 = st.columns(4)

    with a1:

        row = action_lookup.get(
            "NORMAL_GO_PASS"
        )

        if row is not None:

            st.metric(
                "PASS / DROPBACK",
                f"{100 * row['expected_win_probability']:.2f}%",
            )

    with a2:

        row = action_lookup.get(
            "NORMAL_GO_RUN"
        )

        if row is not None:

            st.metric(
                "RUN",
                f"{100 * row['expected_win_probability']:.2f}%",
            )

        else:

            st.metric(
                "RUN",
                "Unavailable",
            )

    with a3:

        row = action_lookup.get(
            "FIELD_GOAL"
        )

        if row is not None:

            st.metric(
                "FIELD GOAL",
                f"{100 * row['expected_win_probability']:.2f}%",
            )

        else:

            st.metric(
                "FIELD GOAL",
                "Unavailable",
            )

    with a4:

        row = action_lookup.get(
            "PUNT"
        )

        if row is not None:

            st.metric(
                "PUNT",
                f"{100 * row['expected_win_probability']:.2f}%",
            )

        else:

            st.metric(
                "PUNT",
                "Unavailable",
            )

    display = production[
        [
            "decision",
            "expected_win_probability",
            "mc_se",
            "detail",
        ]
    ].copy()


    display["Expected WP"] = (
        100
        *
        display[
            "expected_win_probability"
        ]
    ).round(2)


    display["MC SE"] = (
        100
        *
        display[
            "mc_se"
        ]
    ).round(3)


    display = display.rename(
        columns={
            "decision":
                "Action",

            "detail":
                "Model Detail",
        }
    )


    display = display[
        [
            "Action",
            "Expected WP",
            "MC SE",
            "Model Detail",
        ]
    ]


    with st.expander(
        "Show detailed model output",
        expanded=False,
    ):

        st.dataframe(
            display,
            hide_index=True,
            width="stretch",
            column_config={
                "Expected WP":
                    table_value_label,

                "MC SE":
                    "MC SE (pp)",
            },
        )

        st.caption(
            "MC SE measures Monte Carlo simulation noise, "
            "not total model uncertainty."
        )


    with st.expander(
        chart_expander_label,
        expanded=False,
    ):

        chart_data = display[
            [
                "Action",
                "Expected WP",
            ]
        ].copy()

        chart_data["Action"] = (
            chart_data["Action"]
            .replace({
                "GO — PASS/DROPBACK":
                    "Pass / Dropback",

                "GO — RUN":
                    "Run",

                "FIELD GOAL":
                    "Field Goal",

                "PUNT":
                    "Punt",
            })
        )

        chart_data = (
            chart_data
            .sort_values(
                "Expected WP",
                ascending=True,
            )
        )

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Expected WP:Q",
                    title=chart_axis_label,
                ),
                y=alt.Y(
                    "Action:N",
                    title=None,
                    sort="-x",
                    axis=alt.Axis(
                        labelAngle=0,
                    ),
                ),
                tooltip=[
                    alt.Tooltip(
                        "Action:N",
                        title="Action",
                    ),
                    alt.Tooltip(
                        "Expected WP:Q",
                        title=chart_tooltip_label,
                        format=".2f",
                    ),
                ],
            )
            .properties(
                height=220,
            )
        )

        st.altair_chart(
            chart,
            width="stretch",
        )


    unavailable = result[
        "results"
    ][
        ~result[
            "results"
        ]["eligible"]
    ].copy()

    unavailable = unavailable[
        ~unavailable[
            "action"
        ]
        .str.startswith(
            "FAKE_"
        )
    ]

    if len(unavailable) > 0:

        st.warning(
            "Some normal actions are unavailable "
            "in this situation."
        )

        for _, row in unavailable.iterrows():

            st.write(
                f"**{row['decision']}** — "
                f"{row['detail']}"
            )
    # =====================================================
    # Experimental fakes
    # =====================================================

    experimental = result[
        "results"
    ][
        ~result[
            "results"
        ][
            "eligible"
        ]
    ].copy()


    experimental = experimental[
        experimental[
            "action"
        ]
        .str.startswith(
            "FAKE_"
        )
    ]


    if len(experimental) > 0:

        with st.expander(
            "Experimental fake plays"
        ):

            for _, row in (
                experimental.iterrows()
            ):

                st.write(
                    f"**{row['decision']}** — "
                    f"{row['detail']}"
                )


    # =====================================================
    # Notes
    # =====================================================

    with st.expander(
        "Model notes"
    ):

        if qtr == 5:

            st.write(
                "- OT game value uses 1.0 for a win, "
                "0.5 for a tie at expiration, and 0.0 "
                "for a loss; nonterminal continuation "
                "values are model-estimated."
            )

        else:

            st.write(
                "- Expected WP is model-estimated."
            )

        st.write(
            "- MC SE measures simulation noise only, "
            "not total model uncertainty."
        )

        st.write(
            "- Fake punts and fake field goals are "
            "experimental-only."
        )

        st.write(
            "- Kicker-specific context defaults to "
            "the development-data median unless "
            "supplied programmatically."
        )

        st.write(
            "- This is a decision-support research "
            "model, not a causal guarantee."
        )
