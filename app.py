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


    if (fgDistance <= 66) {

        directionInfo.textContent =
            `← offense attacking left`
            +
            ` • Estimated FG: ${fgDistance} yd`;

    } else if (fgDistance <= 70) {

        directionInfo.textContent =
            `← offense attacking left`
            +
            ` • Estimated FG: ${fgDistance} yd`
            +
            ` — extreme distance (extrapolated)`;

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

    ot_phase = None

    if qtr == 5:

        st.caption(
            "Regular-season overtime"
        )

        ot_phase = st.selectbox(
            "OT phase",
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


with col3:

    timeout_options = (
        [0, 1, 2]
        if qtr == 5
        else [0, 1, 2, 3]
    )

    offense_timeouts = st.radio(
        "Offense timeouts",
        options=timeout_options,
        index=(
            2
            if qtr == 5
            else 3
        ),
        horizontal=True,
    )

    defense_timeouts = st.radio(
        "Defense timeouts",
        options=timeout_options,
        index=2,
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

    if (
        qtr == 5
        and
        (
            minutes * 60
            +
            seconds
        )
        > 600
    ):

        clock_valid = False

        st.error(
            "Regular-season OT clock "
            "cannot exceed 10:00."
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
            "after regular-season OT has expired."
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


    with st.spinner(
        "Running 4,000 simulations per eligible action..."
    ):

        result = engine.recommend(
            state,
            n_simulations=4000,
            seed=42,
        )


    (
        production,
        go,
        best,
        second,
    ) = build_strategy_summary(
        result["results"]
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


        if strategy_edge < 1.0:

            st.info(
                f"**Near tie:** "
                f"{best['strategy']} has the highest "
                f"{value_phrase}, but only "
                f"{strategy_edge:.2f} percentage points "
                f"above {second['strategy']}. "
                f"The model does not strongly distinguish "
                f"between these choices."
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


            if playcall_edge < 1.0:

                st.info(
                    f"**Slight lean: "
                    f"{preferred_call}** "
                    f"(+{playcall_edge:.2f} pp)\n\n"
                    f"Run vs. pass is too close "
                    f"to distinguish reliably."
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
