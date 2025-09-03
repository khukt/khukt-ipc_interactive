
import streamlit as st
import random
import plotly.graph_objects as go

st.set_page_config(page_title="IPC Lecture — Plotly Animations (R/RR/RRA)", page_icon="🎞️", layout="wide")
st.title("🎞️ IPC & RPC — Plotly Animations")

X_CLIENT = 0.2
X_SERVER = 0.8

def base_fig(title=""):
    fig = go.Figure()
    # Lifelines
    fig.add_shape(type="line", x0=X_CLIENT, y0=0.9, x1=X_CLIENT, y1=0.1,
                  line=dict(color="black", width=2, dash="dash"))
    fig.add_shape(type="line", x0=X_SERVER, y0=0.9, x1=X_SERVER, y1=0.1,
                  line=dict(color="black", width=2, dash="dash"))
    # Headers
    fig.add_shape(type="rect", x0=X_CLIENT-0.1, y0=0.92, x1=X_CLIENT+0.1, y1=0.97, line=dict(color="black"), fillcolor="white")
    fig.add_shape(type="rect", x0=X_SERVER-0.1, y0=0.92, x1=X_SERVER+0.1, y1=0.97, line=dict(color="black"), fillcolor="white")
    fig.add_annotation(x=X_CLIENT, y=0.945, text="Client", showarrow=False)
    fig.add_annotation(x=X_SERVER, y=0.945, text="Server", showarrow=False)
    # Axes/size
    fig.update_xaxes(visible=False, range=[0,1])
    fig.update_yaxes(visible=False, range=[0,1])
    fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=460, title=title, transition={"duration":0})
    return fig

def msg_frames(n_msgs=6, p_drop=0.2, p_dup=0.1, p_reorder=0.2, speed=1.0, seed=42):
    rnd = random.Random(seed)
    lane_gap = 0.11
    y = 0.83
    plan = []
    for seq in range(1, n_msgs+1):
        copies = 1 + (1 if rnd.random() < p_dup else 0)
        for c in range(copies):
            drop = rnd.random() < p_drop
            jitter = rnd.random() * p_reorder
            plan.append({"seq": seq, "copy": c, "drop": drop, "jitter": jitter, "y": y})
        y -= lane_gap
    plan.sort(key=lambda m: m["jitter"])

    frames = []
    legend_req = go.Scatter(x=[None], y=[None], mode="markers", name="Request", marker=dict(symbol="triangle-right", size=14, color="black"))
    legend_rep = go.Scatter(x=[None], y=[None], mode="markers", name="Reply", marker=dict(symbol="triangle-left", size=14, color="black"))
    legend_drop = go.Scatter(x=[None], y=[None], mode="markers", name="Drop ✕", marker=dict(symbol="x-thin", size=14, color="crimson"))
    moving = go.Scatter(x=[X_CLIENT], y=[0.83], mode="markers",
                        marker=dict(symbol="triangle-right", size=16, color="black"))
    frames.append(go.Frame(name="start", data=[legend_req, legend_rep, legend_drop, moving]))

    tname = 0
    for item in plan:
        steps = int(20 * speed)
        for k in range(steps+1):
            t = k/steps
            x = X_CLIENT + (X_SERVER - X_CLIENT) * t
            y = item["y"]
            opac = 0.35 + 0.65*(1.0 - t) if item["drop"] else 1.0
            m = go.Scatter(x=[x], y=[y], mode="markers",
                           marker=dict(symbol="triangle-right", size=16, color="black", opacity=opac), showlegend=False)
            frames.append(go.Frame(name=f"f{tname}", data=[legend_req, legend_rep, legend_drop, m]))
            tname += 1
        if item["drop"]:
            d = go.Scatter(x=[X_SERVER-0.02], y=[item["y"]], mode="markers",
                           marker=dict(symbol="x-thin", size=18, color="crimson"), showlegend=False)
            frames.append(go.Frame(name=f"drop{tname}", data=[legend_req, legend_rep, legend_drop, d]))
            tname += 1
        else:
            steps = int(20 * speed)
            for k in range(steps+1):
                t = k/steps
                x = X_SERVER - (X_SERVER - X_CLIENT) * t
                y = item["y"] - 0.05
                m = go.Scatter(x=[x], y=[y], mode="markers",
                               marker=dict(symbol="triangle-left", size=16, color="black"), showlegend=False)
                frames.append(go.Frame(name=f"f{tname}", data=[legend_req, legend_rep, legend_drop, m]))
                tname += 1
    return [legend_req, legend_rep, legend_drop, moving], frames

# ---------------- Reliability Animation ----------------

def reliability_frames(proto="RR", calls=4, p_req_loss=0.25, p_rep_loss=0.25, retries=2, speed=1.0, seed=7):
    rnd = random.Random(seed)
    frames = []
    # legends + moving marker
    legend_req = go.Scatter(x=[None], y=[None], mode="markers", name="Request", marker=dict(symbol="triangle-right", size=14, color="black"))
    legend_rep = go.Scatter(x=[None], y=[None], mode="markers", name="Reply", marker=dict(symbol="triangle-left", size=14, color="black"))
    legend_ack = go.Scatter(x=[None], y=[None], mode="markers", name="ACK", marker=dict(symbol="triangle-right", size=14, color="green"))
    legend_drop = go.Scatter(x=[None], y=[None], mode="markers", name="Drop ✕", marker=dict(symbol="x-thin", size=14, color="crimson"))
    moving = go.Scatter(x=[X_CLIENT], y=[0.83], mode="markers",
                        marker=dict(symbol="triangle-right", size=16, color="black"))
    frames.append(go.Frame(name="start", data=[legend_req, legend_rep, legend_ack, legend_drop, moving]))

    y = 0.83
    step = 0.16
    tname = 0

    def move(symbol, ltr=True, yy=0.83, color="black"):
        nonlocal tname
        steps = int(18 * speed)
        for k in range(steps+1):
            t = k/steps
            x = X_CLIENT + (X_SERVER - X_CLIENT) * t if ltr else X_SERVER - (X_SERVER - X_CLIENT) * t
            m = go.Scatter(x=[x], y=[yy], mode="markers",
                           marker=dict(symbol=symbol, size=16, color=color), showlegend=False)
            frames.append(go.Frame(name=f"pf{tname}", data=[legend_req, legend_rep, legend_ack, legend_drop, m]))
            tname += 1

    def cross(yy):
        nonlocal tname
        d = go.Scatter(x=[X_SERVER-0.02], y=[yy], mode="markers",
                       marker=dict(symbol="x-thin", size=18, color="crimson"), showlegend=False)
        frames.append(go.Frame(name=f"px{tname}", data=[legend_req, legend_rep, legend_ack, legend_drop, d]))
        tname += 1

    for call in range(1, calls+1):
        attempts = 0
        got_reply = False
        acked = False
        server_execs = 0
        title = f"{proto} — Call #{call}"
        frames.append(go.Frame(name=f"title{tname}", data=[legend_req, legend_rep, legend_ack, legend_drop, moving],
                               layout=go.Layout(title=title)))
        tname += 1

        if proto == "R":
            attempts = 1
            lost_req = rnd.random() < p_req_loss
            move("triangle-right", True, y, "black")
            if lost_req:
                cross(y)
            else:
                server_execs += 1
            note = f"Server executions: {server_execs}"
            frames.append(go.Frame(name=f"sum{tname}", data=[legend_req, legend_rep, legend_ack, legend_drop, moving],
                                   layout=go.Layout(title=title + "  •  " + note)))
            tname += 1
            y -= step

        elif proto == "RR":
            while attempts <= retries and not got_reply:
                attempts += 1
                lost_req = rnd.random() < p_req_loss
                move("triangle-right", True, y, "black")
                if not lost_req:
                    server_execs += 1
                    lost_rep = rnd.random() < p_rep_loss
                    move("triangle-left", False, y-0.05, "black")
                    if not lost_rep:
                        got_reply = True
                else:
                    cross(y)
                note = f"try {attempts} • server execs: {server_execs}" + (" • reply received" if got_reply else "")
                frames.append(go.Frame(name=f"sum{tname}", data=[legend_req, legend_rep, legend_ack, legend_drop, moving],
                                       layout=go.Layout(title=title + "  •  " + note)))
                tname += 1
                y -= step

        else:  # RRA
            while attempts <= retries and not acked:
                attempts += 1
                lost_req = rnd.random() < p_req_loss
                move("triangle-right", True, y, "black")
                if not lost_req:
                    server_execs += 1
                    lost_rep = rnd.random() < p_rep_loss
                    move("triangle-left", False, y-0.05, "black")
                    if not lost_rep:
                        lost_ack = rnd.random() < p_rep_loss
                        move("triangle-right", True, y-0.10, "green")
                        if not lost_ack:
                            acked = True
                else:
                    cross(y)
                note = f"try {attempts} • server execs: {server_execs}" + (" • ack received" if acked else "")
                frames.append(go.Frame(name=f"sum{tname}", data=[legend_req, legend_rep, legend_ack, legend_drop, moving],
                                       layout=go.Layout(title=title + "  •  " + note)))
                tname += 1
                y -= step

    return [legend_req, legend_rep, legend_ack, legend_drop, moving], frames

# ---------------- UI ----------------

tab1, tab2 = st.tabs(["A) Message Passing", "B) Reliability (R/RR/RRA)"])

with tab1:
    st.subheader("A) Message Passing — drops, duplicates, reorder (slides 32–34)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_msgs = st.number_input("Messages", 1, 20, 6, key="n")
    with c2:
        p_drop = st.slider("Loss", 0.0, 1.0, 0.2, 0.01, key="loss")
    with c3:
        p_dup = st.slider("Duplicate", 0.0, 1.0, 0.1, 0.01, key="dup")
    with c4:
        p_reorder = st.slider("Reorder", 0.0, 1.0, 0.2, 0.01, key="reo")
    with c5:
        speed = st.slider("Speed", 0.5, 3.0, 1.0, 0.1, key="spd")

    if st.button("▶ Build animation", key="build_msg"):
        data, frames = msg_frames(n_msgs, p_drop, p_dup, p_reorder, speed)
        fig = base_fig("Message Passing")
        fig.add_traces(data)
        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(type="buttons",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None, {"frame": {"duration": 70, "redraw": True},
                                                    "fromcurrent": True, "transition": {"duration": 0}}]),
                                  dict(label="Pause", method="animate",
                                       args=[[None], {"mode": "immediate",
                                                      "frame": {"duration": 0, "redraw": False},
                                                      "transition": {"duration": 0}}]),
                              ],
                              direction="left", x=0.5, y=-0.05, xanchor="center", yanchor="top")]
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

with tab2:
    st.subheader("B) Reliability Protocols — R / RR / RRA (slides 36–38)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        proto = st.selectbox("Protocol", ["R", "RR", "RRA"], key="rp_proto")
    with c2:
        calls = st.number_input("Calls", 1, 15, 4, key="rp_calls")
    with c3:
        p_req_loss = st.slider("Req loss", 0.0, 1.0, 0.25, 0.01, key="rp_rloss")
    with c4:
        p_rep_loss = st.slider("Reply/ACK loss", 0.0, 1.0, 0.25, 0.01, key="rp_reploss")
    with c5:
        speed2 = st.slider("Speed", 0.5, 3.0, 1.0, 0.1, key="rp_spd")

    if st.button("▶ Build animation", key="build_rel"):
        data, frames = reliability_frames(proto, calls, p_req_loss, p_rep_loss, retries=2, speed=speed2)
        title = dict(R="R (request only)", RR="RR (req/reply + retries)", RRA="RRA (req/reply/ack)")[proto]
        fig = base_fig(title)
        fig.add_traces(data)
        fig.frames = frames
        fig.update_layout(
            updatemenus=[dict(type="buttons",
                              buttons=[
                                  dict(label="Play", method="animate",
                                       args=[None, {"frame": {"duration": 70, "redraw": True},
                                                    "fromcurrent": True, "transition": {"duration": 0}}]),
                                  dict(label="Pause", method="animate",
                                       args=[[None], {"mode": "immediate",
                                                      "frame": {"duration": 0, "redraw": False},
                                                      "transition": {"duration": 0}}]),
                              ],
                              direction="left", x=0.5, y=-0.05, xanchor="center", yanchor="top")]
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
