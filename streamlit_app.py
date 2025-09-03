
import streamlit as st
import random
import math
import plotly.graph_objects as go

st.set_page_config(page_title="IPC Lecture — Plotly Animations", page_icon="🎞️", layout="wide")
st.title("🎞️ IPC & RPC — Smooth Plotly Animations")
st.caption("PPT-style lifelines with real, smooth animations (Plotly frames).")

# ---------- Common drawing helpers ----------
X_CLIENT = 0.2
X_SERVER = 0.8
Y_TOP = 0.95
Y_BOTTOM = 0.05

def base_fig(title=""):
    fig = go.Figure()
    # Lifelines
    fig.add_shape(type="line", x0=X_CLIENT, y0=Y_TOP-0.05, x1=X_CLIENT, y1=Y_BOTTOM+0.05,
                  line=dict(color="black", width=2, dash="dash"))
    fig.add_shape(type="line", x0=X_SERVER, y0=Y_TOP-0.05, x1=X_SERVER, y1=Y_BOTTOM+0.05,
                  line=dict(color="black", width=2, dash="dash"))
    # Header boxes
    fig.add_shape(type="rect", x0=X_CLIENT-0.1, y0=Y_TOP-0.03, x1=X_CLIENT+0.1, y1=Y_TOP+0.02,
                  line=dict(color="black"), fillcolor="white")
    fig.add_annotation(x=X_CLIENT, y=Y_TOP-0.005, text="Client", showarrow=False, font=dict(size=14))
    fig.add_shape(type="rect", x0=X_SERVER-0.1, y0=Y_TOP-0.03, x1=X_SERVER+0.1, y1=Y_TOP+0.02,
                  line=dict(color="black"), fillcolor="white")
    fig.add_annotation(x=X_SERVER, y=Y_TOP-0.005, text="Server", showarrow=False, font=dict(size=14))

    fig.update_xaxes(visible=False, range=[0,1])
    fig.update_yaxes(visible=False, range=[0,1])
    fig.update_layout(margin=dict(l=20,r=20,t=40,b=20), height=480, title=title)
    return fig

def make_frames_for_messages(n_msgs=6, p_drop=0.2, p_dup=0.1, p_reorder=0.2, speed=1.0, seed=42):
    rnd = random.Random(seed)
    lane_gap = 0.1
    y = 0.85
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
    data_traces = []  # we'll keep a single moving trace
    # static symbols legend
    legend = go.Scatter(x=[None], y=[None], mode="markers", name="Request",
                        marker=dict(symbol="triangle-right", size=14, color="black"))
    legend2 = go.Scatter(x=[None], y=[None], mode="markers", name="Reply",
                        marker=dict(symbol="triangle-left", size=14, color="black"))
    legend3 = go.Scatter(x=[None], y=[None], mode="markers", name="Drop ✕",
                        marker=dict(symbol="x-thin", size=14, color="crimson"))
    # Build frames
    t_idx = 0
    frames.append(go.Frame(name=f"start_{t_idx}", data=[legend, legend2, legend3]))
    for item in plan:
        # Request frames (client→server)
        steps = int(20 * speed)
        for k in range(steps+1):
            t = k/steps
            x = X_CLIENT + (X_SERVER - X_CLIENT) * t
            y = item["y"]
            # Fade if drop
            opacity = 0.35 + 0.65*(1.0 - t) if item["drop"] else 1.0
            req = go.Scatter(x=[x], y=[y], mode="markers",
                             marker=dict(symbol="triangle-right", size=16, color="black", opacity=opacity),
                             showlegend=False)
            frames.append(go.Frame(name=f"f{t_idx}", data=[legend, legend2, legend3, req]))
            t_idx += 1
        if item["drop"]:
            # show X near server
            dropx = X_SERVER - 0.02
            dxy = go.Scatter(x=[dropx], y=[item["y"]], mode="markers",
                             marker=dict(symbol="x-thin", size=18, color="crimson"), showlegend=False)
            frames.append(go.Frame(name=f"drop_{t_idx}", data=[legend, legend2, legend3, dxy]))
            t_idx += 1
        else:
            # Reply frames (server→client)
            steps = int(20 * speed)
            for k in range(steps+1):
                t = k/steps
                x = X_SERVER - (X_SERVER - X_CLIENT) * t
                y = item["y"] - 0.05
                rep = go.Scatter(x=[x], y=[y], mode="markers",
                                 marker=dict(symbol="triangle-left", size=16, color="black"),
                                 showlegend=False)
                frames.append(go.Frame(name=f"f{t_idx}", data=[legend, legend2, legend3, rep]))
                t_idx += 1
    return frames

def make_frames_for_protocol(proto="RR", calls=4, p_req_loss=0.25, p_rep_loss=0.25, retries=2, speed=1.0, seed=7):
    rnd = random.Random(seed)
    frames = []
    legend_req = go.Scatter(x=[None], y=[None], mode="markers", name="Request",
                            marker=dict(symbol="triangle-right", size=14, color="black"))
    legend_rep = go.Scatter(x=[None], y=[None], mode="markers", name="Reply",
                            marker=dict(symbol="triangle-left", size=14, color="black"))
    legend_ack = go.Scatter(x=[None], y=[None], mode="markers", name="ACK",
                            marker=dict(symbol="triangle-right", size=14, color="green"))
    legend_drop = go.Scatter(x=[None], y=[None], mode="markers", name="Drop ✕",
                             marker=dict(symbol="x-thin", size=14, color="crimson"))
    frames.append(go.Frame(name="start", data=[legend_req, legend_rep, legend_ack, legend_drop]))
    y = 0.85
    step = 0.18
    t_idx = 0
    for call in range(1, calls+1):
        attempts = 0
        got_reply = False
        acked = False
        server_execs = 0

        def move(symbol, left_to_right=True, yy=y, color="black"):
            nonlocal t_idx
            steps = int(18 * speed)
            for k in range(steps+1):
                t = k/steps
                x = X_CLIENT + (X_SERVER - X_CLIENT)*t if left_to_right else X_SERVER - (X_SERVER - X_CLIENT)*t
                m = go.Scatter(x=[x], y=[yy], mode="markers",
                               marker=dict(symbol=symbol, size=16, color=color), showlegend=False)
                frames.append(go.Frame(name=f"p{t_idx}", data=[legend_req, legend_rep, legend_ack, legend_drop, m]))
                t_idx += 1

        def cross(yy=y):
            nonlocal t_idx
            d = go.Scatter(x=[X_SERVER-0.02], y=[yy], mode="markers",
                           marker=dict(symbol="x-thin", size=18, color="crimson"), showlegend=False)
            frames.append(go.Frame(name=f"drop{t_idx}", data=[legend_req, legend_rep, legend_ack, legend_drop, d]))
            t_idx += 1

        # Title frame for each call
        title_ann = f"{proto} — Call #{call}"
        frames.append(go.Frame(name=f"title{t_idx}", data=[legend_req, legend_rep, legend_ack, legend_drop],
                               layout=go.Layout(title=title_ann)))
        t_idx += 1

        if proto == "R":
            attempts = 1
            # Request
            lost_req = rnd.random() < p_req_loss
            move("triangle-right", True, y, "black")
            if lost_req:
                cross(y)
            else:
                server_execs += 1
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
                        move("triangle-right", True, y-0.10, "green")  # ACK client→server
                        if not lost_ack:
                            acked = True
                else:
                    cross(y)
                y -= step

        # summary frame note
        note = f"Server executions this call: {server_execs}"
        frames.append(go.Frame(name=f"sum{t_idx}", data=[legend_req, legend_rep, legend_ack, legend_drop],
                               layout=go.Layout(title=title_ann + "  •  " + note)))
        t_idx += 1

    return frames

# ---------- UI ----------

tab1, tab2 = st.tabs(["A) Message Passing (animated)", "B) Reliability Protocols (animated)"])

with tab1:
    st.subheader("A) Message Passing — drops, duplicates, reorder (like slides 32–34)")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        n_msgs = st.number_input("Messages", 1, 20, 6, key="mp_n")
    with c2:
        p_drop = st.slider("Loss", 0.0, 1.0, 0.2, 0.01, key="mp_loss")
    with c3:
        p_dup = st.slider("Duplicate", 0.0, 1.0, 0.1, 0.01, key="mp_dup")
    with c4:
        p_reorder = st.slider("Reorder", 0.0, 1.0, 0.2, 0.01, key="mp_reo")
    with c5:
        speed = st.slider("Speed", 0.5, 3.0, 1.0, 0.1, key="mp_spd")

    if st.button("▶ Build animation", key="mp_btn"):
        frames = make_frames_for_messages(n_msgs, p_drop, p_dup, p_reorder, speed)
        fig = base_fig("Message Passing")
        fig.frames = frames
        fig.update_layout(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(label="Play", method="animate", args=[None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
                     ],
                     direction="left", x=0.5, y=-0.05, xanchor="center", yanchor="top")
            ]
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)

with tab2:
    st.subheader("B) Reliability Protocols — R, RR, RRA (slides 36–38)")
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

    if st.button("▶ Build animation", key="rp_btn"):
        frames = make_frames_for_protocol(proto, calls, p_req_loss, p_rep_loss, retries=2, speed=speed2)
        title = dict(R="R (request only)", RR="RR (req/reply + retries)", RRA="RRA (req/reply/ack)")[proto]
        fig = base_fig(title)
        fig.frames = frames
        fig.update_layout(
            updatemenus=[
                dict(type="buttons",
                     buttons=[
                         dict(label="Play", method="animate", args=[None, {"frame": {"duration": 60, "redraw": True}, "fromcurrent": True}]),
                         dict(label="Pause", method="animate", args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}}]),
                     ],
                     direction="left", x=0.5, y=-0.05, xanchor="center", yanchor="top")
            ]
        )
        st.plotly_chart(fig, use_container_width=True, theme=None)
