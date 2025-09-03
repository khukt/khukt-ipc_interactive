
import streamlit as st
import random
import plotly.graph_objects as go

st.set_page_config(page_title="IPC Lecture — Plotly Animations (Fixed)", page_icon="🎞️", layout="wide")
st.title("🎞️ IPC & RPC — Plotly Animations (Fixed)")

X_CLIENT = 0.2
X_SERVER = 0.8
Y_TOP = 0.95

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
    # We'll maintain 1 moving trace for the "packet" plus 3 dummy legend traces
    legend_req = go.Scatter(x=[None], y=[None], mode="markers", name="Request", marker=dict(symbol="triangle-right", size=14, color="black"))
    legend_rep = go.Scatter(x=[None], y=[None], mode="markers", name="Reply", marker=dict(symbol="triangle-left", size=14, color="black"))
    legend_drop = go.Scatter(x=[None], y=[None], mode="markers", name="Drop ✕", marker=dict(symbol="x-thin", size=14, color="crimson"))
    moving = go.Scatter(x=[X_CLIENT], y=[0.83], mode="markers",
                        marker=dict(symbol="triangle-right", size=16, color="black"))
    # first frame equals initial data positions
    frames.append(go.Frame(name="start", data=[legend_req, legend_rep, legend_drop, moving]))

    tname = 0
    for item in plan:
        # request move
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
            # drop marker near server
            d = go.Scatter(x=[X_SERVER-0.02], y=[item["y"]], mode="markers",
                           marker=dict(symbol="x-thin", size=18, color="crimson"), showlegend=False)
            frames.append(go.Frame(name=f"drop{tname}", data=[legend_req, legend_rep, legend_drop, d]))
            tname += 1
        else:
            # reply move
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

tab1, tab2 = st.tabs(["A) Message Passing (animated)", "B) Reliability (coming next)"])

with tab1:
    st.subheader("A) Message Passing")
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

    if st.button("▶ Build animation", key="build"):
        data, frames = msg_frames(n_msgs, p_drop, p_dup, p_reorder, speed)
        fig = base_fig("Message Passing")
        # IMPORTANT: set initial data traces to match frame data length
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
